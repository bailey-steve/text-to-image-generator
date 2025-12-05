"""Image-to-video generation utility using Stable Video Diffusion via Replicate API."""

import logging
import base64
from typing import Optional
import replicate
from replicate.exceptions import ReplicateError
import requests

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Image-to-video generation using Stable Video Diffusion via Replicate.

    This utility animates still images using the Stable Video Diffusion model,
    creating short video clips with natural motion.

    Attributes:
        api_key: Replicate API token
        client: Replicate client instance
    """

    # Stable Video Diffusion model on Replicate
    SVD_MODEL = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"

    def __init__(self, api_key: str):
        """Initialize the video generator.

        Args:
            api_key: Replicate API token

        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError("Replicate API key is required for video generation")

        self.api_key = api_key
        self.client = replicate.Client(api_token=api_key)
        logger.info("Initialized VideoGenerator with SVD model")

    def generate_video(
        self,
        image_data: bytes,
        fps: int = 6,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        decoding_t: int = 7,
        num_frames: int = 14
    ) -> bytes:
        """Generate a video from a still image using Stable Video Diffusion.

        This method takes a still image and generates a short video with natural motion.

        Args:
            image_data: Input image as bytes (PNG or JPEG)
            fps: Frames per second for output video (1-30, default 6)
            motion_bucket_id: Motion intensity (1-255, default 127)
                             Higher values = more motion
            cond_aug: Conditioning augmentation (0-1, default 0.02)
                     Adds randomness to motion
            decoding_t: Decoding timesteps (1-14, default 7)
                       Higher = better quality but slower
            num_frames: Number of frames to generate (14 or 25)

        Returns:
            Video as bytes (MP4 format)

        Raises:
            ValueError: If parameters are out of range
            ConnectionError: If unable to connect to Replicate API
            RuntimeError: If video generation fails
        """
        if not image_data:
            raise ValueError("Image data cannot be empty")

        if fps < 1 or fps > 30:
            raise ValueError("FPS must be between 1 and 30")

        if motion_bucket_id < 1 or motion_bucket_id > 255:
            raise ValueError("Motion bucket ID must be between 1 and 255")

        if cond_aug < 0 or cond_aug > 1:
            raise ValueError("Conditioning augmentation must be between 0 and 1")

        if decoding_t < 1 or decoding_t > 14:
            raise ValueError("Decoding timesteps must be between 1 and 14")

        if num_frames not in [14, 25]:
            raise ValueError("Number of frames must be 14 or 25")

        try:
            logger.info(
                f"Generating video: fps={fps}, motion={motion_bucket_id}, "
                f"frames={num_frames}"
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

            # Run SVD model
            logger.debug("Calling Replicate SVD API")
            output = self.client.run(
                self.SVD_MODEL,
                input={
                    "input_image": data_uri,
                    "fps": fps,
                    "motion_bucket_id": motion_bucket_id,
                    "cond_aug": cond_aug,
                    "decoding_t": decoding_t,
                    "video_length": "14_frames_with_svd" if num_frames == 14 else "25_frames_with_svd_xt"
                }
            )

            # Download the video
            video_url = str(output)
            logger.debug(f"Downloading video from {video_url}")

            response = requests.get(video_url, timeout=60)
            response.raise_for_status()
            video_data = response.content

            logger.info(f"Successfully generated video ({len(video_data)} bytes)")
            return video_data

        except ReplicateError as e:
            logger.error(f"Replicate API error during video generation: {e}")
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
                raise RuntimeError(f"Video generation failed: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download video: {e}")
            raise RuntimeError(f"Failed to download video: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {e}") from e


# Global singleton instance
_video_generator_instance: Optional[VideoGenerator] = None


def get_video_generator(api_key: str) -> VideoGenerator:
    """Get or create the global VideoGenerator instance.

    Args:
        api_key: Replicate API token

    Returns:
        VideoGenerator instance
    """
    global _video_generator_instance

    if _video_generator_instance is None:
        _video_generator_instance = VideoGenerator(api_key)

    return _video_generator_instance


def reset_video_generator() -> None:
    """Reset the global VideoGenerator instance.

    Useful for testing or when API key changes.
    """
    global _video_generator_instance
    _video_generator_instance = None
