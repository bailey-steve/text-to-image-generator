"""Tests for image-to-video generation functionality."""

import pytest
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from replicate.exceptions import ReplicateError

from src.utils.video_generator import VideoGenerator, get_video_generator, reset_video_generator


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
    with patch('src.utils.video_generator.replicate.Client') as mock_client:
        yield mock_client


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for downloading videos."""
    with patch('src.utils.video_generator.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.content = b'video_data'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        yield mock_get


class TestVideoGenerator:
    """Test VideoGenerator class."""

    def test_initialization(self, mock_replicate_client):
        """Test initialization with API key."""
        generator = VideoGenerator("test_api_key")
        assert generator.api_key == "test_api_key"
        assert generator.client is not None
        mock_replicate_client.assert_called_once_with(api_token="test_api_key")

    def test_initialization_empty_api_key(self):
        """Test initialization fails with empty API key."""
        with pytest.raises(ValueError, match="API key is required"):
            VideoGenerator("")

    def test_generate_video_success(
        self,
        mock_replicate_client,
        mock_requests_get,
        sample_image_bytes
    ):
        """Test successful video generation."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/video.mp4"
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")
        result = generator.generate_video(sample_image_bytes, fps=6, num_frames=14)

        assert result == b'video_data'
        mock_client_instance.run.assert_called_once()
        mock_requests_get.assert_called_once()

    def test_generate_video_empty_image_data(self, mock_replicate_client):
        """Test generate_video with empty image data."""
        generator = VideoGenerator("test_api_key")

        with pytest.raises(ValueError, match="Image data cannot be empty"):
            generator.generate_video(b"")

    def test_generate_video_invalid_fps(self, mock_replicate_client, sample_image_bytes):
        """Test generate_video with invalid FPS."""
        generator = VideoGenerator("test_api_key")

        with pytest.raises(ValueError, match="FPS must be between 1 and 30"):
            generator.generate_video(sample_image_bytes, fps=0)

        with pytest.raises(ValueError, match="FPS must be between 1 and 30"):
            generator.generate_video(sample_image_bytes, fps=31)

    def test_generate_video_invalid_motion_bucket(self, mock_replicate_client, sample_image_bytes):
        """Test generate_video with invalid motion bucket ID."""
        generator = VideoGenerator("test_api_key")

        with pytest.raises(ValueError, match="Motion bucket ID must be between 1 and 255"):
            generator.generate_video(sample_image_bytes, motion_bucket_id=0)

        with pytest.raises(ValueError, match="Motion bucket ID must be between 1 and 255"):
            generator.generate_video(sample_image_bytes, motion_bucket_id=256)

    def test_generate_video_invalid_cond_aug(self, mock_replicate_client, sample_image_bytes):
        """Test generate_video with invalid conditioning augmentation."""
        generator = VideoGenerator("test_api_key")

        with pytest.raises(ValueError, match="Conditioning augmentation must be between 0 and 1"):
            generator.generate_video(sample_image_bytes, cond_aug=-0.1)

        with pytest.raises(ValueError, match="Conditioning augmentation must be between 0 and 1"):
            generator.generate_video(sample_image_bytes, cond_aug=1.1)

    def test_generate_video_invalid_decoding_t(self, mock_replicate_client, sample_image_bytes):
        """Test generate_video with invalid decoding timesteps."""
        generator = VideoGenerator("test_api_key")

        with pytest.raises(ValueError, match="Decoding timesteps must be between 1 and 14"):
            generator.generate_video(sample_image_bytes, decoding_t=0)

        with pytest.raises(ValueError, match="Decoding timesteps must be between 1 and 14"):
            generator.generate_video(sample_image_bytes, decoding_t=15)

    def test_generate_video_invalid_num_frames(self, mock_replicate_client, sample_image_bytes):
        """Test generate_video with invalid number of frames."""
        generator = VideoGenerator("test_api_key")

        with pytest.raises(ValueError, match="Number of frames must be 14 or 25"):
            generator.generate_video(sample_image_bytes, num_frames=10)

    def test_generate_video_authentication_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test generate_video with authentication error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Authentication failed")
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")

        with pytest.raises(ConnectionError, match="Invalid Replicate API token"):
            generator.generate_video(sample_image_bytes)

    def test_generate_video_rate_limit_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test generate_video with rate limit error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Rate limit exceeded")
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            generator.generate_video(sample_image_bytes)

    def test_generate_video_payment_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test generate_video with payment/credit error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Insufficient credit")
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")

        with pytest.raises(RuntimeError, match="Insufficient Replicate credits"):
            generator.generate_video(sample_image_bytes)

    def test_generate_video_generic_error(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test generate_video with generic error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Something went wrong")
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")

        with pytest.raises(RuntimeError, match="Video generation failed"):
            generator.generate_video(sample_image_bytes)

    def test_generate_video_download_failure(
        self,
        mock_replicate_client,
        sample_image_bytes
    ):
        """Test generate_video with download failure."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/video.mp4"
        mock_replicate_client.return_value = mock_client_instance

        with patch('src.utils.video_generator.requests.get') as mock_get:
            mock_get.side_effect = Exception("Download failed")

            generator = VideoGenerator("test_api_key")

            with pytest.raises(RuntimeError, match="Video generation failed"):
                generator.generate_video(sample_image_bytes)

    def test_generate_video_with_jpeg_image(
        self,
        mock_replicate_client,
        mock_requests_get
    ):
        """Test generate_video with JPEG image."""
        # Create JPEG image
        img = Image.new('RGB', (256, 256), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        jpeg_bytes = img_bytes.getvalue()

        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/video.mp4"
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")
        result = generator.generate_video(jpeg_bytes, fps=6, num_frames=14)

        assert result == b'video_data'
        # Verify JPEG format was detected
        call_args = mock_client_instance.run.call_args
        assert "image/jpeg" in call_args[1]["input"]["input_image"]

    def test_generate_video_25_frames(
        self,
        mock_replicate_client,
        mock_requests_get,
        sample_image_bytes
    ):
        """Test generate_video with 25 frames."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/video.mp4"
        mock_replicate_client.return_value = mock_client_instance

        generator = VideoGenerator("test_api_key")
        result = generator.generate_video(sample_image_bytes, num_frames=25)

        assert result == b'video_data'
        # Verify 25 frames mode was used
        call_args = mock_client_instance.run.call_args
        assert call_args[1]["input"]["video_length"] == "25_frames_with_svd_xt"


class TestGlobalVideoGenerator:
    """Test global video generator singleton."""

    def test_get_video_generator(self, mock_replicate_client):
        """Test getting global video generator instance."""
        reset_video_generator()

        generator1 = get_video_generator("api_key_1")
        generator2 = get_video_generator("api_key_2")

        # Should return same instance
        assert generator1 is generator2

    def test_reset_video_generator(self, mock_replicate_client):
        """Test resetting global video generator instance."""
        generator1 = get_video_generator("api_key_1")
        reset_video_generator()
        generator2 = get_video_generator("api_key_2")

        # Should be different instances after reset
        assert generator1 is not generator2
