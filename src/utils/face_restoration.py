"""Face restoration utility using GFPGAN via Replicate API."""

import logging
import base64
from typing import Optional
import replicate
from replicate.exceptions import ReplicateError
import requests

logger = logging.getLogger(__name__)


class FaceRestoration:
    """Face restoration using GFPGAN via Replicate.

    This utility enhances and restores faces in images using the GFPGAN model.
    Particularly useful after image-to-image transformations where faces may
    become blurry or distorted.

    Attributes:
        api_key: Replicate API token
        client: Replicate client instance
    """

    # GFPGAN v1.4 model on Replicate
    GFPGAN_MODEL = "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3"

    def __init__(self, api_key: str):
        """Initialize the face restoration utility.

        Args:
            api_key: Replicate API token

        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError("Replicate API key is required for face restoration")

        self.api_key = api_key
        self.client = replicate.Client(api_token=api_key)
        logger.info("Initialized FaceRestoration with GFPGAN model")

    def enhance_faces(
        self,
        image_data: bytes,
        scale: int = 2,
        version: str = "v1.4",
        weight: float = 0.5
    ) -> bytes:
        """Enhance faces in an image using GFPGAN.

        This method detects faces in the image and applies enhancement to improve
        quality, fix distortions, and add realistic details.

        Args:
            image_data: Input image as bytes (PNG or JPEG)
            scale: Upscaling factor (1-4). Higher values increase resolution.
            version: GFPGAN version to use ("v1.3" or "v1.4")
            weight: Fidelity weight (0-1). Controls identity preservation.
                   0 = keep original face (no enhancement)
                   0.5 = balanced (default, recommended)
                   1 = maximum enhancement (may change identity)

        Returns:
            Enhanced image as bytes

        Raises:
            ValueError: If image_data is empty or parameters are out of range
            ConnectionError: If unable to connect to Replicate API
            RuntimeError: If face enhancement fails
        """
        if not image_data:
            raise ValueError("Image data cannot be empty")

        if scale < 1 or scale > 4:
            raise ValueError("Scale must be between 1 and 4")

        if version not in ["v1.3", "v1.4"]:
            raise ValueError("Version must be 'v1.3' or 'v1.4'")

        if weight < 0 or weight > 1:
            raise ValueError("Weight must be between 0 and 1")

        try:
            logger.info(f"Enhancing faces with GFPGAN {version}, scale={scale}, weight={weight}")

            # Convert to base64 data URI
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Detect image format
            if image_data.startswith(b'\x89PNG'):
                mime_type = 'image/png'
            elif image_data.startswith(b'\xff\xd8'):
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'  # default

            data_uri = f"data:{mime_type};base64,{image_base64}"

            # Run GFPGAN model
            logger.debug("Calling Replicate GFPGAN API")
            output = self.client.run(
                self.GFPGAN_MODEL,
                input={
                    "img": data_uri,
                    "version": version,
                    "scale": scale,
                    "weight": weight
                }
            )

            # Download the enhanced image
            image_url = str(output)
            logger.debug(f"Downloading enhanced image from {image_url}")

            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            enhanced_image_data = response.content

            logger.info(f"Successfully enhanced faces ({len(enhanced_image_data)} bytes)")
            return enhanced_image_data

        except ReplicateError as e:
            logger.error(f"Replicate API error during face enhancement: {e}")
            error_msg = str(e)

            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ConnectionError(
                    "Invalid Replicate API token. Please check your REPLICATE_TOKEN."
                ) from e
            elif "rate limit" in error_msg.lower():
                raise RuntimeError(
                    "Rate limit exceeded. Please try again later."
                ) from e
            elif "payment" in error_msg.lower() or "credit" in error_msg.lower():
                raise RuntimeError(
                    "Insufficient Replicate credits. Please add credits to your account."
                ) from e
            else:
                raise RuntimeError(f"Face enhancement failed: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download enhanced image: {e}")
            raise RuntimeError(f"Failed to download enhanced image: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during face enhancement: {e}")
            raise RuntimeError(f"Face enhancement failed: {e}") from e


# Global singleton instance
_face_restoration_instance: Optional[FaceRestoration] = None


def get_face_restoration(api_key: str) -> FaceRestoration:
    """Get or create the global FaceRestoration instance.

    Args:
        api_key: Replicate API token

    Returns:
        FaceRestoration instance
    """
    global _face_restoration_instance

    if _face_restoration_instance is None:
        _face_restoration_instance = FaceRestoration(api_key)

    return _face_restoration_instance


def reset_face_restoration() -> None:
    """Reset the global FaceRestoration instance.

    Useful for testing or when API key changes.
    """
    global _face_restoration_instance
    _face_restoration_instance = None
