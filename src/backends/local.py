"""Local backend for offline CPU-optimized image generation using Diffusers."""

import logging
from typing import Optional
from pathlib import Path
import io

from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest

logger = logging.getLogger(__name__)


class LocalBackend(BaseBackend):
    """Local backend using Diffusers library for CPU-optimized inference.

    This backend supports offline image generation using models optimized for CPU inference.
    Models are downloaded once and cached locally for future use.

    Supported models:
    - stabilityai/sd-turbo: Fast, 1-4 step inference, 512x512
    - stabilityai/sdxl-turbo: Higher quality, 1-4 steps, 1024x1024

    Attributes:
        model: Current model identifier
        cache_dir: Directory for model cache
        pipeline: Loaded Diffusers pipeline
    """

    DEFAULT_MODEL = "stabilityai/sd-turbo"
    SUPPORTED_MODELS = [
        "stabilityai/sd-turbo",
        "stabilityai/sdxl-turbo"
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize the local backend.

        Args:
            model: Model identifier (defaults to sd-turbo)
            cache_dir: Directory for caching models (defaults to ~/.cache/huggingface)
        """
        super().__init__(api_key=None)  # No API key needed for local
        self.model = model or self.DEFAULT_MODEL
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pipeline = None

        logger.info(f"Initializing LocalBackend with model: {self.model}")

    @property
    def name(self) -> str:
        """Get backend name."""
        return "Local"

    @property
    def supported_models(self) -> list[str]:
        """Get list of supported models."""
        return self.SUPPORTED_MODELS.copy()

    def _load_pipeline(self):
        """Load the Diffusers pipeline.

        This downloads the model if not cached and loads it for inference.
        Uses CPU-optimized settings for best performance.
        """
        if self.pipeline is not None:
            return

        try:
            from diffusers import AutoPipelineForText2Image
            import torch

            logger.info(f"Loading model {self.model}...")

            # Load pipeline with CPU optimization
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model,
                torch_dtype=torch.float32,  # Use float32 for CPU
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                safety_checker=None,  # Disable for faster inference
                requires_safety_checker=False
            )

            # Move to CPU and optimize
            self.pipeline = self.pipeline.to("cpu")

            # Enable memory efficient attention if available
            try:
                self.pipeline.enable_attention_slicing()
            except Exception:
                pass  # Not all models support this

            logger.info(f"Model {self.model} loaded successfully")

        except ImportError as e:
            raise ImportError(
                "Missing required dependencies for local backend. "
                "Install with: pip install torch diffusers transformers accelerate"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model {self.model}: {e}")
            raise

    def generate_image(self, request: GenerationRequest) -> bytes:
        """Generate an image locally using the Diffusers pipeline.

        Args:
            request: Generation parameters

        Returns:
            Generated image as PNG bytes

        Raises:
            RuntimeError: If generation fails
        """
        # Load pipeline if not already loaded
        self._load_pipeline()

        try:
            logger.info(f"Generating image locally with prompt: {request.prompt[:50]}...")

            # Prepare generation kwargs
            generation_kwargs = {
                "prompt": request.prompt,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "width": request.width,
                "height": request.height,
            }

            # Add negative prompt if provided
            if request.negative_prompt:
                generation_kwargs["negative_prompt"] = request.negative_prompt

            # Add seed if provided
            if request.seed is not None:
                import torch
                generator = torch.Generator(device="cpu").manual_seed(request.seed)
                generation_kwargs["generator"] = generator

            # Generate image
            result = self.pipeline(**generation_kwargs)
            image = result.images[0]

            # Convert to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            logger.info("Image generated successfully")
            return img_bytes.getvalue()

        except Exception as e:
            error_msg = f"Local generation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def health_check(self) -> bool:
        """Check if the local backend is functional.

        Returns:
            True if backend can load the model, False otherwise
        """
        try:
            self._load_pipeline()
            return self.pipeline is not None
        except Exception as e:
            logger.error(f"Local backend health check failed: {e}")
            return False

    def set_model(self, model: str) -> None:
        """Set a different model.

        Args:
            model: Model identifier

        Raises:
            ValueError: If model is not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model} not supported. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        self.model = model
        self.pipeline = None  # Reset pipeline to force reload
        logger.info(f"Model set to: {model}")

    def __repr__(self) -> str:
        """String representation."""
        return f"LocalBackend(model='{self.model}')"
