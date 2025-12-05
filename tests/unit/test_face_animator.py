"""Tests for face animation functionality."""

import pytest
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from replicate.exceptions import ReplicateError

from src.utils.face_animator import FaceAnimator, get_face_animator, reset_face_animator


@pytest.fixture
def sample_portrait_bytes():
    """Create a sample portrait image as bytes for testing."""
    img = Image.new('RGB', (512, 512), color='beige')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def mock_replicate_client():
    """Create a mock Replicate client."""
    with patch('src.utils.face_animator.replicate.Client') as mock_client:
        yield mock_client


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for downloading videos."""
    with patch('src.utils.face_animator.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.content = b'animated_video_data'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        yield mock_get


class TestFaceAnimator:
    """Test FaceAnimator class."""

    def test_initialization(self, mock_replicate_client):
        """Test initialization with API key."""
        animator = FaceAnimator("test_api_key")
        assert animator.api_key == "test_api_key"
        assert animator.client is not None
        mock_replicate_client.assert_called_once_with(api_token="test_api_key")

    def test_initialization_empty_api_key(self):
        """Test initialization fails with empty API key."""
        with pytest.raises(ValueError, match="API key is required"):
            FaceAnimator("")

    def test_animate_face_success(
        self,
        mock_replicate_client,
        mock_requests_get,
        sample_portrait_bytes
    ):
        """Test successful face animation."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/animated.mp4"
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")
        result = animator.animate_face(sample_portrait_bytes, video_length=3)

        assert result == b'animated_video_data'
        mock_client_instance.run.assert_called_once()
        mock_requests_get.assert_called_once()

    def test_animate_face_empty_image_data(self, mock_replicate_client):
        """Test animate_face with empty image data."""
        animator = FaceAnimator("test_api_key")

        with pytest.raises(ValueError, match="Image data cannot be empty"):
            animator.animate_face(b"")

    def test_animate_face_invalid_expression_scale(self, mock_replicate_client, sample_portrait_bytes):
        """Test animate_face with invalid expression scale."""
        animator = FaceAnimator("test_api_key")

        with pytest.raises(ValueError, match="Expression scale must be between 0 and 2"):
            animator.animate_face(sample_portrait_bytes, expression_scale=-0.1)

        with pytest.raises(ValueError, match="Expression scale must be between 0 and 2"):
            animator.animate_face(sample_portrait_bytes, expression_scale=2.1)

    def test_animate_face_invalid_head_rotation_scale(self, mock_replicate_client, sample_portrait_bytes):
        """Test animate_face with invalid head rotation scale."""
        animator = FaceAnimator("test_api_key")

        with pytest.raises(ValueError, match="Head rotation scale must be between 0 and 2"):
            animator.animate_face(sample_portrait_bytes, head_rotation_scale=-0.1)

        with pytest.raises(ValueError, match="Head rotation scale must be between 0 and 2"):
            animator.animate_face(sample_portrait_bytes, head_rotation_scale=2.1)

    def test_animate_face_invalid_video_length(self, mock_replicate_client, sample_portrait_bytes):
        """Test animate_face with invalid video length."""
        animator = FaceAnimator("test_api_key")

        with pytest.raises(ValueError, match="Video length must be between 1 and 10 seconds"):
            animator.animate_face(sample_portrait_bytes, video_length=0)

        with pytest.raises(ValueError, match="Video length must be between 1 and 10 seconds"):
            animator.animate_face(sample_portrait_bytes, video_length=11)

    def test_animate_face_authentication_error(
        self,
        mock_replicate_client,
        sample_portrait_bytes
    ):
        """Test animate_face with authentication error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Authentication failed")
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")

        with pytest.raises(ConnectionError, match="Invalid Replicate API token"):
            animator.animate_face(sample_portrait_bytes)

    def test_animate_face_rate_limit_error(
        self,
        mock_replicate_client,
        sample_portrait_bytes
    ):
        """Test animate_face with rate limit error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Rate limit exceeded")
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            animator.animate_face(sample_portrait_bytes)

    def test_animate_face_payment_error(
        self,
        mock_replicate_client,
        sample_portrait_bytes
    ):
        """Test animate_face with payment/credit error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Insufficient credit")
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")

        with pytest.raises(RuntimeError, match="Insufficient Replicate credits"):
            animator.animate_face(sample_portrait_bytes)

    def test_animate_face_no_face_detected(
        self,
        mock_replicate_client,
        sample_portrait_bytes
    ):
        """Test animate_face with no face detected error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("No face detected in image")
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")

        with pytest.raises(RuntimeError, match="No face detected in image"):
            animator.animate_face(sample_portrait_bytes)

    def test_animate_face_generic_error(
        self,
        mock_replicate_client,
        sample_portrait_bytes
    ):
        """Test animate_face with generic error."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.side_effect = ReplicateError("Something went wrong")
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")

        with pytest.raises(RuntimeError, match="Face animation failed"):
            animator.animate_face(sample_portrait_bytes)

    def test_animate_face_download_failure(
        self,
        mock_replicate_client,
        sample_portrait_bytes
    ):
        """Test animate_face with download failure."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/animated.mp4"
        mock_replicate_client.return_value = mock_client_instance

        with patch('src.utils.face_animator.requests.get') as mock_get:
            mock_get.side_effect = Exception("Download failed")

            animator = FaceAnimator("test_api_key")

            with pytest.raises(RuntimeError, match="Face animation failed"):
                animator.animate_face(sample_portrait_bytes)

    def test_animate_face_with_jpeg_image(
        self,
        mock_replicate_client,
        mock_requests_get
    ):
        """Test animate_face with JPEG image."""
        # Create JPEG image
        img = Image.new('RGB', (256, 256), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        jpeg_bytes = img_bytes.getvalue()

        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/animated.mp4"
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")
        result = animator.animate_face(jpeg_bytes, video_length=3)

        assert result == b'animated_video_data'
        # Verify JPEG format was detected
        call_args = mock_client_instance.run.call_args
        assert "image/jpeg" in call_args[1]["input"]["source_image"]

    def test_animate_face_with_custom_parameters(
        self,
        mock_replicate_client,
        mock_requests_get,
        sample_portrait_bytes
    ):
        """Test animate_face with custom parameters."""
        mock_client_instance = MagicMock()
        mock_client_instance.run.return_value = "https://example.com/animated.mp4"
        mock_replicate_client.return_value = mock_client_instance

        animator = FaceAnimator("test_api_key")
        result = animator.animate_face(
            sample_portrait_bytes,
            expression_scale=1.5,
            head_rotation_scale=0.5,
            blink=False,
            video_length=5
        )

        assert result == b'animated_video_data'
        # Verify parameters were passed correctly (mapped to SadTalker parameters)
        call_args = mock_client_instance.run.call_args
        assert call_args[1]["input"]["expression_scale"] == 1.5
        # head_rotation_scale 0.5 maps to pose_style = int(0.5 * 22) = 11
        assert call_args[1]["input"]["pose_style"] == 11
        # head_rotation_scale 0.5 > 0.3, so still should be False
        assert call_args[1]["input"]["still"] is False
        assert call_args[1]["input"]["preprocess"] == "crop"


class TestGlobalFaceAnimator:
    """Test global face animator singleton."""

    def test_get_face_animator(self, mock_replicate_client):
        """Test getting global face animator instance."""
        reset_face_animator()

        animator1 = get_face_animator("api_key_1")
        animator2 = get_face_animator("api_key_2")

        # Should return same instance
        assert animator1 is animator2

    def test_reset_face_animator(self, mock_replicate_client):
        """Test resetting global face animator instance."""
        animator1 = get_face_animator("api_key_1")
        reset_face_animator()
        animator2 = get_face_animator("api_key_2")

        # Should be different instances after reset
        assert animator1 is not animator2
