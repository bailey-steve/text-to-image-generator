"""Unit tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.core.models import GenerationRequest, GeneratedImage


class TestGenerationRequest:
    """Tests for GenerationRequest model."""

    def test_valid_request_minimal(self):
        """Test creation with minimal required fields."""
        request = GenerationRequest(prompt="A cat")

        assert request.prompt == "A cat"
        assert request.negative_prompt is None
        assert request.guidance_scale == 7.5  # default
        assert request.num_inference_steps == 50  # default
        assert request.seed is None
        assert request.width == 512  # default
        assert request.height == 512  # default

    def test_valid_request_full(self):
        """Test creation with all fields."""
        request = GenerationRequest(
            prompt="A beautiful sunset",
            negative_prompt="blurry",
            guidance_scale=10.0,
            num_inference_steps=75,
            seed=42,
            width=768,
            height=768
        )

        assert request.prompt == "A beautiful sunset"
        assert request.negative_prompt == "blurry"
        assert request.guidance_scale == 10.0
        assert request.num_inference_steps == 75
        assert request.seed == 42
        assert request.width == 768
        assert request.height == 768

    def test_prompt_too_short(self):
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="")

        assert "prompt" in str(exc_info.value)

    def test_prompt_too_long(self):
        """Test that overly long prompt is rejected."""
        long_prompt = "A" * 1001  # exceeds max_length=1000

        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt=long_prompt)

        assert "prompt" in str(exc_info.value)

    def test_guidance_scale_too_low(self):
        """Test that guidance_scale below minimum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="test", guidance_scale=0.5)

        assert "guidance_scale" in str(exc_info.value)

    def test_guidance_scale_too_high(self):
        """Test that guidance_scale above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="test", guidance_scale=25.0)

        assert "guidance_scale" in str(exc_info.value)

    def test_num_inference_steps_too_low(self):
        """Test that num_inference_steps below minimum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="test", num_inference_steps=0)

        assert "num_inference_steps" in str(exc_info.value)

    def test_num_inference_steps_too_high(self):
        """Test that num_inference_steps above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="test", num_inference_steps=200)

        assert "num_inference_steps" in str(exc_info.value)

    def test_width_too_small(self):
        """Test that width below minimum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="test", width=128)

        assert "width" in str(exc_info.value)

    def test_width_too_large(self):
        """Test that width above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="test", width=2048)

        assert "width" in str(exc_info.value)

    def test_height_validation(self):
        """Test height validation."""
        # Valid
        GenerationRequest(prompt="test", height=512)

        # Too small
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", height=128)

        # Too large
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", height=2048)


class TestGeneratedImage:
    """Tests for GeneratedImage model."""

    def test_valid_generated_image(self):
        """Test creation of GeneratedImage."""
        image_data = b"fake_image_bytes"
        result = GeneratedImage(
            image_data=image_data,
            prompt="A cat",
            backend="test_backend"
        )

        assert result.image_data == image_data
        assert result.prompt == "A cat"
        assert result.backend == "test_backend"
        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}

    def test_generated_image_with_metadata(self):
        """Test GeneratedImage with metadata."""
        metadata = {
            "model": "stable-diffusion-v1-5",
            "guidance_scale": 7.5,
            "steps": 50
        }

        result = GeneratedImage(
            image_data=b"test",
            prompt="test prompt",
            backend="huggingface",
            metadata=metadata
        )

        assert result.metadata == metadata
        assert result.metadata["model"] == "stable-diffusion-v1-5"

    def test_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated."""
        before = datetime.now()

        result = GeneratedImage(
            image_data=b"test",
            prompt="test",
            backend="test"
        )

        after = datetime.now()

        assert before <= result.timestamp <= after

    def test_custom_timestamp(self):
        """Test that custom timestamp can be provided."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0)

        result = GeneratedImage(
            image_data=b"test",
            prompt="test",
            backend="test",
            timestamp=custom_time
        )

        assert result.timestamp == custom_time

    def test_required_fields(self):
        """Test that all required fields must be provided."""
        # Missing image_data
        with pytest.raises(ValidationError):
            GeneratedImage(
                prompt="test",
                backend="test"
            )

        # Missing prompt
        with pytest.raises(ValidationError):
            GeneratedImage(
                image_data=b"test",
                backend="test"
            )

        # Missing backend
        with pytest.raises(ValidationError):
            GeneratedImage(
                image_data=b"test",
                prompt="test"
            )
