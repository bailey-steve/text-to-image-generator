"""Face animation utility using LivePortrait via Replicate API."""

import logging
import base64
from typing import Optional
import replicate
from replicate.exceptions import ReplicateError
import requests

logger = logging.getLogger(__name__)


class FaceAnimator:
    """Face-aware animation using LivePortrait via Replicate.

    This utility animates portrait images with natural facial expressions,
    head movements, blinking, and realistic motion specifically for faces.

    Attributes:
        api_key: Replicate API token
        client: Replicate client instance
    """

    # LivePortrait model on Replicate
    LIVEPORTRAIT_MODEL = "fofr/live-portrait:7c1efc2c4af37288debb6167e87e60da8c55fa952c13e8684bfef5cab1f3fc9f"

    def __init__(self, api_key: str):
        """Initialize the face animator.

        Args:
            api_key: Replicate API token

        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError("Replicate API key is required for face animation")

        self.api_key = api_key
        self.client = replicate.Client(api_token=api_key)
        logger.info("Initialized FaceAnimator with LivePortrait model")

    def animate_face(
        self,
        image_data: bytes,
        expression_scale: float = 1.0,
        head_rotation_scale: float = 1.0,
        blink: bool = True,
        video_length: int = 3
    ) -> bytes:
        """Animate a portrait with natural facial expressions and movements.

        This method takes a portrait image and generates a video with realistic
        facial animation including expressions, head movements, and blinking.

        Args:
            image_data: Input portrait image as bytes (PNG or JPEG)
            expression_scale: Expression intensity (0-2, default 1.0)
                            0 = no expression changes
                            1 = natural expressions
                            2 = exaggerated expressions
            head_rotation_scale: Head movement amount (0-2, default 1.0)
                               0 = no head movement
                               1 = natural movement
                               2 = dramatic movement
            blink: Enable natural blinking (default True)
            video_length: Video duration in seconds (1-10, default 3)

        Returns:
            Animated video as bytes (MP4 format)

        Raises:
            ValueError: If parameters are out of range
            ConnectionError: If unable to connect to Replicate API
            RuntimeError: If animation fails
        """
        if not image_data:
            raise ValueError("Image data cannot be empty")

        if expression_scale < 0 or expression_scale > 2:
            raise ValueError("Expression scale must be between 0 and 2")

        if head_rotation_scale < 0 or head_rotation_scale > 2:
            raise ValueError("Head rotation scale must be between 0 and 2")

        if video_length < 1 or video_length > 10:
            raise ValueError("Video length must be between 1 and 10 seconds")

        try:
            logger.info(
                f"Animating face: expression={expression_scale}, "
                f"rotation={head_rotation_scale}, blink={blink}, length={video_length}s"
            )

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

            # Run LivePortrait model
            logger.debug("Calling Replicate LivePortrait API")
            output = self.client.run(
                self.LIVEPORTRAIT_MODEL,
                input={
                    "image": data_uri,
                    "expression_scale": expression_scale,
                    "head_rotation_scale": head_rotation_scale,
                    "enable_blink": blink,
                    "video_length": video_length
                }
            )

            # Download the video
            video_url = str(output)
            logger.debug(f"Downloading animated video from {video_url}")

            response = requests.get(video_url, timeout=60)
            response.raise_for_status()
            video_data = response.content

            logger.info(f"Successfully animated face ({len(video_data)} bytes)")
            return video_data

        except ReplicateError as e:
            logger.error(f"Replicate API error during face animation: {e}")
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
            elif "no face" in error_msg.lower() or "face not found" in error_msg.lower():
                raise RuntimeError(
                    "No face detected in image. Please use a clear portrait photo."
                ) from e
            else:
                raise RuntimeError(f"Face animation failed: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download animated video: {e}")
            raise RuntimeError(f"Failed to download animated video: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during face animation: {e}")
            raise RuntimeError(f"Face animation failed: {e}") from e


# Global singleton instance
_face_animator_instance: Optional[FaceAnimator] = None


def get_face_animator(api_key: str) -> FaceAnimator:
    """Get or create the global FaceAnimator instance.

    Args:
        api_key: Replicate API token

    Returns:
        FaceAnimator instance
    """
    global _face_animator_instance

    if _face_animator_instance is None:
        _face_animator_instance = FaceAnimator(api_key)

    return _face_animator_instance


def reset_face_animator() -> None:
    """Reset the global FaceAnimator instance.

    Useful for testing or when API key changes.
    """
    global _face_animator_instance
    _face_animator_instance = None
