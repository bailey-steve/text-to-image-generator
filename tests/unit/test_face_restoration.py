"""Tests for face restoration functionality."""

import pytest
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from replicate.exceptions import ReplicateError

from src.utils.face_restoration import FaceRestoration, get_face_restoration, reset_face_restoration


@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes for testing."""
    img = Image.new('RGB', (512, 512), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def mock_replicate_client():
    """Create a mock Replicate client."""
    with patch('src.utils.face_restoration.replicate.Client') as mock_client:
        yield mock_client


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for downloading images."""
    with patch('src.utils.face_restoration.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.content = b'enhanced_image_data'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        yield mock_get


class TestFaceRestoration:
    """Test FaceRestoration class."""

    def test_initialization(self, mock_replicate_client):
        """Test initialization with API key."""
        restorer = FaceRestoration("test_api_key")
        assert restorer.api_key == "test_api_key"
        assert restorer.client is not None
        mock_replicate_client.assert_called_once_with(api_token="test_api_key")

    def test_initialization_empty_api_key(self):
        """Test initialization fails with empty API key."""
        with pytest.raises(ValueError, match="API key is required"):
            FaceRestoration("")

    def test_enhance_faces_success(
        self,
        mock_replicate_client,
        mock_requests_get,
        sample_image_bytes
    ):
        """Test successful face enhancement."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/enhanced.png"
        mock_replicate_client.return_value = mock_client_instance

        restorer = FaceRestoration("test_api_key")
        result = restorer.enhance_faces(sample_image_bytes, scale=2, version="v1.4")

        assert result == b'enhanced_image_data'
        mock_client_instance.run.assert_called_once()
        mock_requests_get.assert_called_once()

    def test_enhance_faces_empty_image_data(self, mock_replicate_client):
        """Test enhance_faces with empty image data."""
        restorer = FaceRestoration("test_api_key")

        with pytest.raises(ValueError, match="Image data cannot be empty"):
            restorer.enhance_faces(b"")

    def test_enhance_faces_invalid_scale(self, mock_replicate_client, sample_image_bytes):
        """Test enhance_faces with invalid scale."""
        restorer = FaceRestoration("test_api_key")

        with pytest.raises(ValueError, match="Scale must be between 1 and 4"):
            restorer.enhance_faces(sample_image_bytes, scale=5)

        with pytest.raises(ValueError, match="Scale must be between 1 and 4"):
            restorer.enhance_faces(sample_image_bytes, scale=0)

    def test_enhance_faces_invalid_version(self, mock_replicate_client, sample_image_bytes):
        """Test enhance_faces with invalid version."""
        restorer = FaceRestoration("test_api_key")

        with pytest.raises(ValueError, match="Version must be"):
            restorer.enhance_faces(sample_image_bytes, version="v2.0")

    def test_enhance_faces_authentication_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test enhance_faces with authentication error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Authentication failed")
        mock_replicate_client.return_value = mock_client_instance

        restorer = FaceRestoration("test_api_key")

        with pytest.raises(ConnectionError, match="Invalid Replicate API token"):
            restorer.enhance_faces(sample_image_bytes)

    def test_enhance_faces_rate_limit_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test enhance_faces with rate limit error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Rate limit exceeded")
        mock_replicate_client.return_value = mock_client_instance

        restorer = FaceRestoration("test_api_key")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            restorer.enhance_faces(sample_image_bytes)

    def test_enhance_faces_payment_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test enhance_faces with payment/credit error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Insufficient credit")
        mock_replicate_client.return_value = mock_client_instance

        restorer = FaceRestoration("test_api_key")

        with pytest.raises(RuntimeError, match="Insufficient Replicate credits"):
            restorer.enhance_faces(sample_image_bytes)

    def test_enhance_faces_generic_replicate_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test enhance_faces with generic Replicate error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Something went wrong")
        mock_replicate_client.return_value = mock_client_instance

        restorer = FaceRestoration("test_api_key")

        with pytest.raises(RuntimeError, match="Face enhancement failed"):
            restorer.enhance_faces(sample_image_bytes)

    def test_enhance_faces_download_failure(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test enhance_faces with download failure."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/enhanced.png"
        mock_replicate_client.return_value = mock_client_instance

        with patch('src.utils.face_restoration.requests.get') as mock_get:
            mock_get.side_effect = Exception("Download failed")

            restorer = FaceRestoration("test_api_key")

            with pytest.raises(RuntimeError, match="Face enhancement failed"):
                restorer.enhance_faces(sample_image_bytes)

    def test_enhance_faces_with_jpeg_image(
        self,
        mock_replicate_client,
        mock_requests_get
    ):
        """Test enhance_faces with JPEG image."""
        # Create JPEG image
        img = Image.new('RGB', (256, 256), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        jpeg_bytes = img_bytes.getvalue()

        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/enhanced.jpg"
        mock_replicate_client.return_value = mock_client_instance

        restorer = FaceRestoration("test_api_key")
        result = restorer.enhance_faces(jpeg_bytes, scale=2)

        assert result == b'enhanced_image_data'
        # Verify JPEG format was detected
        call_args = mock_client_instance.run.call_args
        assert "image/jpeg" in call_args[1]["input"]["img"]


class TestGlobalFaceRestoration:
    """Test global face restoration singleton."""

    def test_get_face_restoration(self, mock_replicate_client):
        """Test getting global face restoration instance."""
        reset_face_restoration()

        restorer1 = get_face_restoration("api_key_1")
        restorer2 = get_face_restoration("api_key_2")

        # Should return same instance
        assert restorer1 is restorer2

    def test_reset_face_restoration(self, mock_replicate_client):
        """Test resetting global face restoration instance."""
        restorer1 = get_face_restoration("api_key_1")
        reset_face_restoration()
        restorer2 = get_face_restoration("api_key_2")

        # Should be different instances after reset
        assert restorer1 is not restorer2
