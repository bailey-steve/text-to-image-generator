"""Shared test fixtures and configuration."""

import pytest
import os
from unittest.mock import Mock
from PIL import Image

from src.core.models import GenerationRequest, GeneratedImage


@pytest.fixture
def sample_prompt():
    """Return a sample prompt for testing."""
    return "A beautiful sunset over mountains"


@pytest.fixture
def sample_generation_request():
    """Return a sample GenerationRequest for testing."""
    return GenerationRequest(
        prompt="A beautiful sunset over mountains",
        negative_prompt="blurry, low quality",
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=42,
        width=512,
        height=512
    )


@pytest.fixture
def sample_fake_image():
    """Return a fake PIL Image for testing."""
    return Image.new('RGB', (512, 512), color='red')


@pytest.fixture
def sample_image_bytes(sample_fake_image):
    """Return sample image as bytes."""
    import io
    img_byte_arr = io.BytesIO()
    sample_fake_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


@pytest.fixture
def sample_generated_image(sample_image_bytes, sample_prompt):
    """Return a sample GeneratedImage for testing."""
    return GeneratedImage(
        image_data=sample_image_bytes,
        prompt=sample_prompt,
        backend="test_backend",
        metadata={
            "model": "test-model",
            "guidance_scale": 7.5,
            "steps": 50
        }
    )


@pytest.fixture
def mock_huggingface_client(sample_fake_image):
    """Return a mocked HuggingFace InferenceClient."""
    mock_client = Mock()
    mock_client.text_to_image.return_value = sample_fake_image
    mock_client.get_model_status.return_value = {"status": "ok"}
    return mock_client


@pytest.fixture
def test_api_key():
    """Return a test API key."""
    return "hf_test_token_12345"


# Skip integration tests unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Automatically skip integration tests unless RUN_INTEGRATION_TESTS is set."""
    skip_integration = pytest.mark.skip(reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=true to enable)")

    for item in items:
        if "integration" in item.keywords:
            if not os.getenv("RUN_INTEGRATION_TESTS", "").lower() == "true":
                item.add_marker(skip_integration)
