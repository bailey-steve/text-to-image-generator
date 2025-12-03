"""Dummy backend that generates simple colored images for testing."""

import io
import random
import logging
from typing import Optional
from PIL import Image
from datetime import datetime

from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest, GeneratedImage

logger = logging.getLogger(__name__)


class DummyBackend(BaseBackend):
    """A dummy backend that generates simple colored rectangles.

    This backend is useful for:
    - Testing the plugin system
    - Demonstrating how to create custom backend plugins
    - Fast offline testing without API keys or heavy models

    The backend generates solid-color images with random colors based on
    the prompt hash, so the same prompt always generates the same color.

    Example:
        backend = DummyBackend()
        request = GenerationRequest(prompt="test", width=512, height=512)
        image = backend.generate_image(request)
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the dummy backend.

        Args:
            api_key: Not used, but accepted for compatibility
        """
        super().__init__(api_key=api_key)
        logger.info("DummyBackend initialized")

    @property
    def name(self) -> str:
        """Get backend name."""
        return "Dummy"

    @property
    def supported_models(self) -> list[str]:
        """Get list of supported models."""
        return ["dummy-v1", "dummy-v2"]

    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        """Generate a solid-color image.

        The color is determined by hashing the prompt, so the same prompt
        always generates the same color.

        Args:
            request: Generation parameters

        Returns:
            GeneratedImage with a solid-color image
        """
        logger.info(f"Generating dummy image for prompt: {request.prompt[:50]}...")

        # Generate a color based on the prompt hash
        prompt_hash = hash(request.prompt)
        random.seed(prompt_hash)

        # Generate RGB values
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = (r, g, b)

        # Create image
        image = Image.new('RGB', (request.width, request.height), color=color)

        # Add some text to the image
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)

            # Try to use a default font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            # Draw prompt on image
            text = f"Dummy: {request.prompt[:30]}"
            draw.text((10, 10), text, fill=(255, 255, 255))
            draw.text((10, 40), f"Color: RGB{color}", fill=(255, 255, 255))

        except Exception as e:
            logger.warning(f"Could not add text to image: {e}")

        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        image_data = img_bytes.getvalue()

        # Create response
        generated_image = GeneratedImage(
            image_data=image_data,
            prompt=request.prompt,
            backend=self.name,
            timestamp=datetime.now(),
            metadata={
                "model": "dummy-v1",
                "color": f"RGB{color}",
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
            }
        )

        logger.info(f"Generated dummy image ({len(image_data)} bytes)")
        return generated_image

    def health_check(self) -> bool:
        """Check if the backend is functional.

        Returns:
            Always True for dummy backend
        """
        return True

    def __repr__(self) -> str:
        """String representation."""
        return "DummyBackend(model='dummy-v1')"
