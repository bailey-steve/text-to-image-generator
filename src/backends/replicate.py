"""Replicate API backend implementation."""

import logging
from datetime import datetime
from typing import Optional
import replicate
from replicate.exceptions import ReplicateError
import io
from PIL import Image

from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest, GeneratedImage

logger = logging.getLogger(__name__)


class ReplicateBackend(BaseBackend):
    """Backend implementation using Replicate API.

    This backend uses Replicate's API to generate images using various
    Stable Diffusion and FLUX models.

    Attributes:
        api_key: Replicate API token
        model: The model version to use for generation
        client: Replicate client instance
    """

    # FLUX.1-schnell is fast and high-quality
    DEFAULT_MODEL = "black-forest-labs/flux-schnell"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize the Replicate backend.

        Args:
            api_key: Replicate API token
            model: Optional model identifier (defaults to FLUX.1-schnell)

        Raises:
            ValueError: If API key is empty
        """
        super().__init__(api_key)

        if not api_key:
            raise ValueError("Replicate API key is required")

        self.model = model or self.DEFAULT_MODEL
        self.client = replicate.Client(api_token=api_key)
        logger.info(f"Initialized Replicate backend with model: {self.model}")

    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        """Generate an image using Replicate API.

        Supports both text-to-image and image-to-image generation.

        Args:
            request: The generation request with prompt and parameters

        Returns:
            GeneratedImage with the generated image data

        Raises:
            RuntimeError: If image generation fails
            ConnectionError: If unable to connect to Replicate API
        """
        try:
            # Determine if this is image-to-image or text-to-image
            is_img2img = request.init_image is not None

            if is_img2img:
                logger.info(f"Generating image-to-image with prompt: {request.prompt[:50]}...")
            else:
                logger.info(f"Generating text-to-image with prompt: {request.prompt[:50]}...")

            # Prepare input for Replicate
            input_params = {
                "prompt": request.prompt,
            }

            # Adjust inference steps based on model
            # FLUX models: max 16 steps (optimized for 4)
            # SDXL/SD models: typically 20-50 steps
            if "flux" in self.model.lower():
                input_params["num_inference_steps"] = min(request.num_inference_steps, 16)
            else:
                input_params["num_inference_steps"] = request.num_inference_steps

            # Add image-to-image specific parameters
            if is_img2img:
                # Convert init_image bytes to base64 data URI
                import base64
                image_base64 = base64.b64encode(request.init_image).decode('utf-8')
                # Detect image format
                if request.init_image.startswith(b'\x89PNG'):
                    mime_type = 'image/png'
                elif request.init_image.startswith(b'\xff\xd8'):
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'image/png'  # default

                input_params["image"] = f"data:{mime_type};base64,{image_base64}"

                # Different models use different parameter names for strength
                # SDXL uses "prompt_strength", some models use "strength"
                if "sdxl" in self.model.lower() or "stability-ai" in self.model.lower():
                    input_params["prompt_strength"] = request.strength
                else:
                    input_params["strength"] = request.strength
            else:
                # Text-to-image specific parameters
                input_params["width"] = request.width
                input_params["height"] = request.height

            # Add optional parameters if supported
            if request.negative_prompt:
                input_params["negative_prompt"] = request.negative_prompt

            if request.seed is not None:
                input_params["seed"] = request.seed

            # Run the model
            logger.debug(f"Calling Replicate API with params: {list(input_params.keys())}")
            output = self.client.run(
                self.model,
                input=input_params
            )

            # Replicate returns either a URL or list of URLs
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output

            # Download the image
            import requests
            response = requests.get(str(image_url), timeout=30)
            response.raise_for_status()
            image_data = response.content

            # Create response
            metadata = {
                "model": self.model,
                "num_inference_steps": request.num_inference_steps,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "image_url": str(image_url),
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

        except ReplicateError as e:
            logger.error(f"Replicate API error: {e}")
            error_msg = str(e)

            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ConnectionError(
                    "Invalid Replicate API token. Please check your REPLICATE_TOKEN."
                ) from e
            elif "rate limit" in error_msg.lower():
                raise RuntimeError(
                    "Rate limit exceeded. Please try again later."
                ) from e
            else:
                raise RuntimeError(f"Replicate API error: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image: {e}")
            raise RuntimeError(f"Failed to download generated image: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during image generation: {e}")
            raise RuntimeError(f"Failed to generate image: {e}") from e

    def health_check(self) -> bool:
        """Check if the Replicate API is accessible.

        Returns:
            True if the backend is healthy, False otherwise
        """
        try:
            logger.debug("Performing health check...")
            # Try to list models as a lightweight health check
            # This verifies API key and connectivity
            models_iter = self.client.models.list()
            next(iter(models_iter))  # Just get the first model
            logger.debug("Health check passed")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    @property
    def name(self) -> str:
        """Get the backend name.

        Returns:
            The string "Replicate"
        """
        return "Replicate"

    @property
    def supported_models(self) -> list[str]:
        """Get a list of commonly used models.

        Returns:
            List of popular model identifiers on Replicate
        """
        return [
            "black-forest-labs/flux-schnell",
            "black-forest-labs/flux-dev",
            "stability-ai/sdxl",
            "stability-ai/stable-diffusion",
        ]

    def set_model(self, model_id: str) -> None:
        """Change the model used for generation.

        Args:
            model_id: Replicate model identifier
        """
        self.model = model_id
        logger.info(f"Switched to model: {model_id}")
