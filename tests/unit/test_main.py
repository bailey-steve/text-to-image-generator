"""Unit tests for main application logic."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io


class TestMainLogic:
    """Tests for main application logic without importing gradio."""

    @patch('app.config.Settings')
    def test_backend_creation_logic(self, mock_settings_class):
        """Test backend creation logic."""
        # This test validates the configuration logic
        mock_settings = Mock()
        mock_settings.huggingface_token = "test_token"
        mock_settings.default_backend = "huggingface"
        mock_settings_class.return_value = mock_settings

        # Verify settings can be accessed
        from app.config import settings
        assert settings is not None

    def test_generation_request_validation(self):
        """Test that GenerationRequest validates inputs."""
        from src.core.models import GenerationRequest
        from pydantic import ValidationError

        # Valid request
        request = GenerationRequest(prompt="test")
        assert request.prompt == "test"

        # Invalid - empty prompt
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="")

        # Invalid - guidance scale too high
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", guidance_scale=100)

    def test_backend_interface_contract(self):
        """Test that backends follow the expected interface."""
        from src.backends.huggingface import HuggingFaceBackend
        from src.core.base_backend import BaseBackend

        # HuggingFaceBackend should be a subclass of BaseBackend
        assert issubclass(HuggingFaceBackend, BaseBackend)

    @patch('src.backends.huggingface.InferenceClient')
    def test_image_generation_flow(self, mock_client_class):
        """Test the full image generation flow."""
        from src.backends.huggingface import HuggingFaceBackend
        from src.core.models import GenerationRequest

        # Create a fake PIL image
        fake_image = Image.new('RGB', (512, 512), color='red')

        # Mock the InferenceClient
        mock_client = Mock()
        mock_client.text_to_image.return_value = fake_image
        mock_client_class.return_value = mock_client

        # Create backend and generate
        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(prompt="A beautiful sunset")

        result = backend.generate_image(request)

        # Verify the flow worked
        assert result.prompt == "A beautiful sunset"
        assert result.backend == "HuggingFace"
        assert len(result.image_data) > 0

    def test_error_handling_flow(self):
        """Test error handling in the application."""
        from src.backends.huggingface import HuggingFaceBackend

        # Test that empty API key raises error
        with pytest.raises(ValueError, match="API key is required"):
            HuggingFaceBackend(api_key="")

    @patch('src.backends.huggingface.InferenceClient')
    def test_connection_error_handling(self, mock_client_class):
        """Test handling of connection errors."""
        from src.backends.huggingface import HuggingFaceBackend
        from src.core.models import GenerationRequest
        from huggingface_hub.utils import HfHubHTTPError

        # Mock a 401 error
        mock_response = Mock()
        mock_response.status_code = 401

        mock_client = Mock()
        mock_client.text_to_image.side_effect = HfHubHTTPError(
            "Unauthorized",
            response=mock_response
        )
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="invalid_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(ConnectionError, match="Invalid HuggingFace API token"):
            backend.generate_image(request)

    @patch('src.backends.huggingface.InferenceClient')
    def test_rate_limit_handling(self, mock_client_class):
        """Test handling of rate limit errors."""
        from src.backends.huggingface import HuggingFaceBackend
        from src.core.models import GenerationRequest
        from huggingface_hub.utils import HfHubHTTPError

        # Mock a 429 error
        mock_response = Mock()
        mock_response.status_code = 429

        mock_client = Mock()
        mock_client.text_to_image.side_effect = HfHubHTTPError(
            "Rate limit",
            response=mock_response
        )
        mock_client_class.return_value = mock_client

        backend = HuggingFaceBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            backend.generate_image(request)

    def test_metadata_preservation(self):
        """Test that metadata is preserved in results."""
        from src.core.models import GeneratedImage
        from datetime import datetime

        metadata = {
            "model": "stable-diffusion-v1-5",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "custom_field": "custom_value"
        }

        result = GeneratedImage(
            image_data=b"test",
            prompt="test prompt",
            backend="test",
            metadata=metadata
        )

        assert result.metadata["model"] == "stable-diffusion-v1-5"
        assert result.metadata["guidance_scale"] == 7.5
        assert result.metadata["custom_field"] == "custom_value"
        assert isinstance(result.timestamp, datetime)
