"""HuggingFace Inference API backend implementation."""

import logging
from datetime import datetime
from typing import Optional
from huggingface_hub import InferenceClient, model_info
from huggingface_hub.utils import HfHubHTTPError

from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest, GeneratedImage

logger = logging.getLogger(__name__)


class HuggingFaceBackend(BaseBackend):
    """Backend implementation using HuggingFace Inference API.

    This backend uses HuggingFace's serverless inference API to generate images.
    It supports various Stable Diffusion models available on HuggingFace.

    Attributes:
        api_key: HuggingFace API token
        model: The model ID to use for generation
        client: HuggingFace InferenceClient instance
    """

    DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize the HuggingFace backend.

        Args:
            api_key: HuggingFace API token
            model: Optional model ID (defaults to Stable Diffusion 2.1)

        Raises:
            ValueError: If API key is empty
        """
        super().__init__(api_key)

        if not api_key:
            raise ValueError("HuggingFace API key is required")

        self.model = model or self.DEFAULT_MODEL
        self.client = InferenceClient(token=api_key)
        logger.info(f"Initialized HuggingFace backend with model: {self.model}")

    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        """Generate an image using HuggingFace Inference API.

        Supports both text-to-image and image-to-image generation.

        Args:
            request: The generation request with prompt and parameters

        Returns:
            GeneratedImage with the generated image data

        Raises:
            RuntimeError: If image generation fails
            ConnectionError: If unable to connect to HuggingFace API
        """
        try:
            import io
            from PIL import Image

            # Determine if this is image-to-image or text-to-image
            is_img2img = request.init_image is not None

            if is_img2img:
                logger.info(f"Generating image-to-image with prompt: {request.prompt[:50]}...")

                # Convert init_image bytes to PIL Image
                init_image_pil = Image.open(io.BytesIO(request.init_image))

                # Call HuggingFace image-to-image API
                image = self.client.image_to_image(
                    image=init_image_pil,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    model=self.model,
                    guidance_scale=request.guidance_scale,
                    num_inference_steps=request.num_inference_steps,
                    strength=request.strength,
                )
            else:
                logger.info(f"Generating text-to-image with prompt: {request.prompt[:50]}...")

                # Call HuggingFace text-to-image API
                image = self.client.text_to_image(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    model=self.model,
                    guidance_scale=request.guidance_scale,
                    num_inference_steps=request.num_inference_steps,
                    width=request.width,
                    height=request.height,
                )

            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()

            # Create response
            metadata = {
                "model": self.model,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "generation_type": "image-to-image" if is_img2img else "text-to-image",
            }

            # Add type-specific metadata
            if is_img2img:
                metadata["strength"] = request.strength
            else:
                metadata["width"] = request.width
                metadata["height"] = request.height

            result = GeneratedImage(
                image_data=image_data,
                prompt=request.prompt,
                backend=self.name,
                timestamp=datetime.now(),
                metadata=metadata
            )

            logger.info(f"Successfully generated image ({len(image_data)} bytes)")
            return result

        except HfHubHTTPError as e:
            logger.error(f"HuggingFace API error: {e}")
            if e.response.status_code == 401:
                raise ConnectionError(
                    "Invalid HuggingFace API token. Please check your HUGGINGFACE_TOKEN."
                ) from e
            elif e.response.status_code == 429:
                raise RuntimeError(
                    "Rate limit exceeded. Please try again later or upgrade your HuggingFace plan."
                ) from e
            else:
                raise RuntimeError(f"HuggingFace API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during image generation: {e}")
            raise RuntimeError(f"Failed to generate image: {e}") from e

    def health_check(self) -> bool:
        """Check if the HuggingFace API is accessible.

        Returns:
            True if the backend is healthy, False otherwise
        """
        try:
            # Try to get model info as a lightweight health check
            logger.debug("Performing health check...")
            model_info(self.model, token=self.api_key)
            logger.debug("Health check passed")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    @property
    def name(self) -> str:
        """Get the backend name.

        Returns:
            The string "HuggingFace"
        """
        return "HuggingFace"

    @property
    def supported_models(self) -> list[str]:
        """Get a list of commonly used models.

        Returns:
            List of popular Stable Diffusion model IDs on HuggingFace
        """
        return [
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "prompthero/openjourney",
        ]

    def set_model(self, model_id: str) -> None:
        """Change the model used for generation.

        Args:
            model_id: HuggingFace model ID
        """
        self.model = model_id
        logger.info(f"Switched to model: {model_id}")
