"""Core data models for text-to-image generation."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request model for image generation.

    Attributes:
        prompt: The text prompt describing the desired image
        negative_prompt: Optional text describing what to avoid in the image
        guidance_scale: How closely to follow the prompt (1.0-20.0)
        num_inference_steps: Number of denoising steps (more = better quality but slower)
        seed: Random seed for reproducibility (None = random)
        width: Output image width in pixels
        height: Output image height in pixels
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text prompt describing the desired image"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Text describing what to avoid in the image"
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="How closely to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=150,
        description="Number of denoising steps"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    width: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Output image width in pixels"
    )
    height: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Output image height in pixels"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A serene landscape with mountains and a lake at sunset",
                "negative_prompt": "blurry, low quality, distorted",
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "seed": 42,
                "width": 512,
                "height": 512
            }
        }


class GeneratedImage(BaseModel):
    """Response model for generated images.

    Attributes:
        image_data: Raw image bytes
        prompt: The prompt used to generate the image
        backend: Name of the backend that generated the image
        timestamp: When the image was generated
        metadata: Additional information about the generation
    """

    image_data: bytes = Field(
        ...,
        description="Raw image bytes"
    )
    prompt: str = Field(
        ...,
        description="The prompt used to generate the image"
    )
    backend: str = Field(
        ...,
        description="Name of the backend that generated the image"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the image was generated"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information about the generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image_data": b"<binary image data>",
                "prompt": "A serene landscape with mountains and a lake at sunset",
                "backend": "huggingface",
                "timestamp": "2025-11-30T12:00:00",
                "metadata": {
                    "model": "stable-diffusion-v1-5",
                    "guidance_scale": 7.5,
                    "steps": 50
                }
            }
        }
