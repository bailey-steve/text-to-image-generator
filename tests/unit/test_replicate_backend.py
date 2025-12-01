"""Unit tests for Replicate backend."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from src.backends.replicate import ReplicateBackend
from src.core.models import GenerationRequest, GeneratedImage
from replicate.exceptions import ReplicateError


class TestReplicateBackend:
    """Tests for ReplicateBackend."""

    @patch('src.backends.replicate.replicate.Client')
    def test_initialization(self, mock_client_class):
        """Test backend initialization."""
        backend = ReplicateBackend(api_key="test_token")

        assert backend.api_key == "test_token"
        assert backend.name == "Replicate"
        assert backend.model == ReplicateBackend.DEFAULT_MODEL
        mock_client_class.assert_called_once_with(api_token="test_token")

    @patch('src.backends.replicate.replicate.Client')
    def test_initialization_with_custom_model(self, mock_client_class):
        """Test backend initialization with custom model."""
        custom_model = "stability-ai/sdxl"
        backend = ReplicateBackend(api_key="test_token", model=custom_model)

        assert backend.model == custom_model

    def test_initialization_empty_api_key(self):
        """Test that initialization fails with empty API key."""
        with pytest.raises(ValueError, match="API key is required"):
            ReplicateBackend(api_key="")

    def test_name_property(self):
        """Test the name property."""
        with patch('src.backends.replicate.replicate.Client'):
            backend = ReplicateBackend(api_key="test_token")
            assert backend.name == "Replicate"

    def test_supported_models(self):
        """Test that supported_models returns a list."""
        with patch('src.backends.replicate.replicate.Client'):
            backend = ReplicateBackend(api_key="test_token")
            models = backend.supported_models

            assert isinstance(models, list)
            assert len(models) > 0
            assert "black-forest-labs/flux-schnell" in models

    @patch('requests.get')
    @patch('src.backends.replicate.replicate.Client')
    def test_generate_image_success(self, mock_client_class, mock_requests_get):
        """Test successful image generation."""
        # Create fake image
        fake_image_pil = Image.new('RGB', (512, 512), color='blue')
        img_byte_arr = io.BytesIO()
        fake_image_pil.save(img_byte_arr, format='PNG')
        fake_image_bytes = img_byte_arr.getvalue()

        # Mock Replicate client
        mock_client = Mock()
        mock_client.run.return_value = "https://example.com/image.png"
        mock_client_class.return_value = mock_client

        # Mock requests.get
        mock_response = Mock()
        mock_response.content = fake_image_bytes
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        # Create backend and generate
        backend = ReplicateBackend(api_key="test_token")
        request = GenerationRequest(prompt="A beautiful sunset")

        result = backend.generate_image(request)

        # Assertions
        assert isinstance(result, GeneratedImage)
        assert result.prompt == "A beautiful sunset"
        assert result.backend == "Replicate"
        assert len(result.image_data) > 0
        assert result.metadata["model"] == backend.model

        mock_client.run.assert_called_once()
        mock_requests_get.assert_called_once()

    @patch('src.backends.replicate.replicate.Client')
    def test_generate_image_authentication_error(self, mock_client_class):
        """Test handling of authentication errors."""
        mock_client = Mock()
        mock_client.run.side_effect = ReplicateError("authentication failed")
        mock_client_class.return_value = mock_client

        backend = ReplicateBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(ConnectionError, match="Invalid Replicate API token"):
            backend.generate_image(request)

    @patch('src.backends.replicate.replicate.Client')
    def test_generate_image_rate_limit_error(self, mock_client_class):
        """Test handling of rate limit errors."""
        mock_client = Mock()
        mock_client.run.side_effect = ReplicateError("rate limit exceeded")
        mock_client_class.return_value = mock_client

        backend = ReplicateBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            backend.generate_image(request)

    @patch('src.backends.replicate.replicate.Client')
    def test_generate_image_generic_replicate_error(self, mock_client_class):
        """Test handling of generic Replicate errors."""
        mock_client = Mock()
        mock_client.run.side_effect = ReplicateError("some error")
        mock_client_class.return_value = mock_client

        backend = ReplicateBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Replicate API error"):
            backend.generate_image(request)

    @patch('requests.get')
    @patch('src.backends.replicate.replicate.Client')
    def test_generate_image_download_failure(self, mock_client_class, mock_requests_get):
        """Test handling of image download failures."""
        import requests

        mock_client = Mock()
        mock_client.run.return_value = "https://example.com/image.png"
        mock_client_class.return_value = mock_client

        mock_requests_get.side_effect = requests.exceptions.RequestException("Download failed")

        backend = ReplicateBackend(api_key="test_token")
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Failed to download generated image"):
            backend.generate_image(request)

    @patch('src.backends.replicate.replicate.Client')
    def test_generate_image_list_output(self, mock_client_class):
        """Test handling of list output from Replicate."""
        with patch('requests.get') as mock_get:
            # Mock Replicate returning a list
            mock_client = Mock()
            mock_client.run.return_value = ["https://example.com/image1.png"]
            mock_client_class.return_value = mock_client

            # Mock image download
            fake_image = Image.new('RGB', (512, 512))
            img_bytes = io.BytesIO()
            fake_image.save(img_bytes, format='PNG')

            mock_response = Mock()
            mock_response.content = img_bytes.getvalue()
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            backend = ReplicateBackend(api_key="test_token")
            request = GenerationRequest(prompt="test")

            result = backend.generate_image(request)

            assert result.backend == "Replicate"
            mock_get.assert_called_with("https://example.com/image1.png", timeout=30)

    @patch('src.backends.replicate.replicate.Client')
    def test_health_check_success(self, mock_client_class):
        """Test successful health check."""
        mock_client = Mock()
        mock_models = Mock()
        # Create a mock model object
        mock_model = Mock()
        mock_model.name = "test-model"
        # Return an iterator that will yield the mock model
        mock_models.list.return_value = iter([mock_model])
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client

        backend = ReplicateBackend(api_key="test_token")
        result = backend.health_check()

        assert result is True

    @patch('src.backends.replicate.replicate.Client')
    def test_health_check_failure(self, mock_client_class):
        """Test failed health check."""
        mock_client = Mock()
        mock_models = Mock()
        mock_models.list.side_effect = Exception("Connection error")
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client

        backend = ReplicateBackend(api_key="test_token")
        result = backend.health_check()

        assert result is False

    @patch('src.backends.replicate.replicate.Client')
    def test_set_model(self, mock_client_class):
        """Test changing the model."""
        mock_client_class.return_value = Mock()

        backend = ReplicateBackend(api_key="test_token")
        original_model = backend.model

        new_model = "stability-ai/sdxl"
        backend.set_model(new_model)

        assert backend.model == new_model
        assert backend.model != original_model
