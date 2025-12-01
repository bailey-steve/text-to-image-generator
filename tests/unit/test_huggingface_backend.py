"""Unit tests for HuggingFace backend."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from src.backends.huggingface import HuggingFaceBackend
from src.core.models import GenerationRequest, GeneratedImage
from huggingface_hub.utils import HfHubHTTPError


class TestHuggingFaceBackend:
    """Tests for HuggingFaceBackend."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = HuggingFaceBackend(api_key="test_token")

        assert backend.api_key == "test_token"
        assert backend.name == "HuggingFace"
        assert backend.model == HuggingFaceBackend.DEFAULT_MODEL

    def test_initialization_with_custom_model(self):
        """Test backend initialization with custom model."""
        custom_model = "runwayml/stable-diffusion-v1-5"
        backend = HuggingFaceBackend(api_key="test_token", model=custom_model)

        assert backend.model == custom_model

    def test_initialization_empty_api_key(self):
        """Test that initialization fails with empty API key."""
        with pytest.raises(ValueError, match="API key is required"):
            HuggingFaceBackend(api_key="")

    def test_name_property(self):
        """Test the name property."""
        backend = HuggingFaceBackend(api_key="test_token")
        assert backend.name == "HuggingFace"

    def test_supported_models(self):
        """Test that supported_models returns a list."""
        backend = HuggingFaceBackend(api_key="test_token")
        models = backend.supported_models

        assert isinstance(models, list)
        assert len(models) > 0
        assert "stabilityai/stable-diffusion-2-1" in models

    @patch('src.backends.huggingface.InferenceClient')
    def test_generate_image_success(self, mock_client_class):
        """Test successful image generation."""
        # Create a fake PIL image
        fake_image = Image.new('RGB', (512, 512), color='red')

        # Mock the InferenceClient
        mock_client = Mock()
        mock_client.text_to_image.return_value = fake_image
        mock_client_class.return_value = mock_client

        # Create backend and generate image
        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(
            prompt="A beautiful sunset",
            negative_prompt="blurry",
            guidance_scale=7.5,
            num_inference_steps=50,
            width=512,
            height=512
        )

        result = backend.generate_image(request)

        # Assertions
        assert isinstance(result, GeneratedImage)
        assert result.prompt == "A beautiful sunset"
        assert result.backend == "HuggingFace"
        assert len(result.image_data) > 0
        assert result.metadata["model"] == backend.model
        assert result.metadata["guidance_scale"] == 7.5
        assert result.metadata["num_inference_steps"] == 50

        # Verify the mock was called correctly
        mock_client.text_to_image.assert_called_once()
        call_kwargs = mock_client.text_to_image.call_args.kwargs
        assert call_kwargs["prompt"] == "A beautiful sunset"
        assert call_kwargs["negative_prompt"] == "blurry"
        assert call_kwargs["guidance_scale"] == 7.5

    @patch('src.backends.huggingface.InferenceClient')
    def test_generate_image_401_error(self, mock_client_class):
        """Test handling of 401 authentication error."""
        # Create mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 401

        mock_client = Mock()
        mock_client.text_to_image.side_effect = HfHubHTTPError(
            "Unauthorized",
            response=mock_response
        )
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(ConnectionError, match="Invalid HuggingFace API token"):
            backend.generate_image(request)

    @patch('src.backends.huggingface.InferenceClient')
    def test_generate_image_429_rate_limit(self, mock_client_class):
        """Test handling of 429 rate limit error."""
        # Create mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 429

        mock_client = Mock()
        mock_client.text_to_image.side_effect = HfHubHTTPError(
            "Rate limit exceeded",
            response=mock_response
        )
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            backend.generate_image(request)

    @patch('src.backends.huggingface.InferenceClient')
    def test_generate_image_other_http_error(self, mock_client_class):
        """Test handling of other HTTP errors (not 401/429)."""
        # Create mock HTTP error with status code 500
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = Mock()
        mock_client.text_to_image.side_effect = HfHubHTTPError(
            "Internal server error",
            response=mock_response
        )
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="HuggingFace API error"):
            backend.generate_image(request)

    @patch('src.backends.huggingface.InferenceClient')
    def test_generate_image_generic_error(self, mock_client_class):
        """Test handling of generic errors."""
        mock_client = Mock()
        mock_client.text_to_image.side_effect = Exception("Something went wrong")
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Failed to generate image"):
            backend.generate_image(request)

    @patch('src.backends.huggingface.InferenceClient')
    def test_health_check_success(self, mock_client_class):
        """Test successful health check."""
        mock_client = Mock()
        mock_client.get_model_status.return_value = {"status": "ok"}
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        result = backend.health_check()

        assert result is True
        mock_client.get_model_status.assert_called_once_with(backend.model)

    @patch('src.backends.huggingface.InferenceClient')
    def test_health_check_failure(self, mock_client_class):
        """Test failed health check."""
        mock_client = Mock()
        mock_client.get_model_status.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        result = backend.health_check()

        assert result is False

    @patch('src.backends.huggingface.InferenceClient')
    def test_set_model(self, mock_client_class):
        """Test changing the model."""
        mock_client_class.return_value = Mock()

        backend = HuggingFaceBackend(api_key="test_token")
        original_model = backend.model

        new_model = "runwayml/stable-diffusion-v1-5"
        backend.set_model(new_model)

        assert backend.model == new_model
        assert backend.model != original_model

    @patch('src.backends.huggingface.InferenceClient')
    def test_repr(self, mock_client_class):
        """Test string representation."""
        mock_client_class.return_value = Mock()

        backend = HuggingFaceBackend(api_key="test_token")
        repr_str = repr(backend)

        assert "HuggingFaceBackend" in repr_str
        assert "HuggingFace" in repr_str
