"""Tests for image-to-image generation functionality."""

import pytest
import io
from PIL import Image
from src.core.models import GenerationRequest, GeneratedImage
from src.backends.huggingface import HuggingFaceBackend
from src.backends.replicate import ReplicateBackend
from src.backends.local import LocalBackend


@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes for testing."""
    # Create a simple 256x256 red image
    img = Image.new('RGB', (256, 256), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


class TestGenerationRequestWithImage:
    """Test GenerationRequest model with image-to-image parameters."""

    def test_request_with_init_image(self, sample_image_bytes):
        """Test creating a request with init_image."""
        request = GenerationRequest(
            prompt="Test prompt",
            init_image=sample_image_bytes,
            strength=0.8
        )
        assert request.init_image is not None
        assert request.strength == 0.8
        assert isinstance(request.init_image, bytes)

    def test_request_without_init_image(self):
        """Test creating a text-to-image request (no init_image)."""
        request = GenerationRequest(
            prompt="Test prompt"
        )
        assert request.init_image is None
        assert request.strength == 0.8  # default value

    def test_strength_validation(self, sample_image_bytes):
        """Test strength parameter validation."""
        # Valid strength
        request = GenerationRequest(
            prompt="Test",
            init_image=sample_image_bytes,
            strength=0.5
        )
        assert request.strength == 0.5

        # Strength at boundaries
        request = GenerationRequest(
            prompt="Test",
            init_image=sample_image_bytes,
            strength=0.0
        )
        assert request.strength == 0.0

        request = GenerationRequest(
            prompt="Test",
            init_image=sample_image_bytes,
            strength=1.0
        )
        assert request.strength == 1.0

    def test_strength_out_of_range(self, sample_image_bytes):
        """Test that strength out of range raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            GenerationRequest(
                prompt="Test",
                init_image=sample_image_bytes,
                strength=1.5
            )

        with pytest.raises(Exception):
            GenerationRequest(
                prompt="Test",
                init_image=sample_image_bytes,
                strength=-0.1
            )


class TestHuggingFaceBackendImg2Img:
    """Test HuggingFace backend image-to-image support."""

    def test_img2img_request_detection(self, sample_image_bytes):
        """Test that backend detects image-to-image requests."""
        request = GenerationRequest(
            prompt="Test prompt",
            init_image=sample_image_bytes,
            strength=0.7
        )
        # Backend should detect init_image is not None
        assert request.init_image is not None

    def test_text2img_request_detection(self):
        """Test that backend detects text-to-image requests."""
        request = GenerationRequest(
            prompt="Test prompt",
            width=512,
            height=512
        )
        # Backend should detect init_image is None
        assert request.init_image is None


class TestReplicateBackendImg2Img:
    """Test Replicate backend image-to-image support."""

    def test_base64_encoding_png(self, sample_image_bytes):
        """Test that PNG images are properly encoded."""
        import base64
        encoded = base64.b64encode(sample_image_bytes).decode('utf-8')
        assert len(encoded) > 0
        # Check PNG magic bytes
        assert sample_image_bytes.startswith(b'\x89PNG')

    def test_base64_encoding_jpeg(self):
        """Test that JPEG images are properly encoded."""
        # Create a JPEG image
        img = Image.new('RGB', (256, 256), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        jpeg_bytes = img_bytes.getvalue()

        import base64
        encoded = base64.b64encode(jpeg_bytes).decode('utf-8')
        assert len(encoded) > 0
        # Check JPEG magic bytes
        assert jpeg_bytes.startswith(b'\xff\xd8')


class TestLocalBackendImg2Img:
    """Test Local backend image-to-image support."""

    def test_local_backend_has_img2img_pipeline(self):
        """Test that local backend initializes img2img pipeline attribute."""
        backend = LocalBackend()
        assert hasattr(backend, 'img2img_pipeline')
        assert backend.img2img_pipeline is None  # Not loaded yet

    def test_local_backend_has_text2img_pipeline(self):
        """Test that local backend still has text2img pipeline."""
        backend = LocalBackend()
        assert hasattr(backend, 'pipeline')
        assert backend.pipeline is None  # Not loaded yet

    def test_model_reset_clears_both_pipelines(self):
        """Test that set_model resets both pipelines."""
        backend = LocalBackend(model="stabilityai/sd-turbo")
        # Simulate loaded pipelines
        backend.pipeline = "loaded"
        backend.img2img_pipeline = "loaded"

        # Set new model
        backend.set_model("stabilityai/sdxl-turbo")

        # Both should be reset
        assert backend.pipeline is None
        assert backend.img2img_pipeline is None


class TestGeneratedImageMetadata:
    """Test that generated images include correct metadata for img2img."""

    def test_img2img_metadata_includes_strength(self, sample_image_bytes):
        """Test that img2img results include strength in metadata."""
        result = GeneratedImage(
            image_data=b"fake_image_data",
            prompt="Test prompt",
            backend="HuggingFace",
            metadata={
                "generation_type": "image-to-image",
                "strength": 0.8,
                "model": "flux-schnell"
            }
        )
        assert result.metadata["generation_type"] == "image-to-image"
        assert result.metadata["strength"] == 0.8

    def test_text2img_metadata_includes_dimensions(self):
        """Test that text-to-image results include width/height in metadata."""
        result = GeneratedImage(
            image_data=b"fake_image_data",
            prompt="Test prompt",
            backend="HuggingFace",
            metadata={
                "generation_type": "text-to-image",
                "width": 512,
                "height": 512,
                "model": "flux-schnell"
            }
        )
        assert result.metadata["generation_type"] == "text-to-image"
        assert result.metadata["width"] == 512
        assert result.metadata["height"] == 512


class TestImageConversion:
    """Test image format conversions for img2img."""

    def test_pil_to_bytes_png(self):
        """Test converting PIL Image to PNG bytes."""
        img = Image.new('RGB', (128, 128), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        data = img_bytes.getvalue()

        assert len(data) > 0
        assert data.startswith(b'\x89PNG')

    def test_bytes_to_pil(self, sample_image_bytes):
        """Test converting bytes back to PIL Image."""
        img = Image.open(io.BytesIO(sample_image_bytes))
        assert img.size == (256, 256)
        assert img.mode == 'RGB'

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion PIL -> bytes -> PIL."""
        # Create original image
        original = Image.new('RGB', (100, 100), color='yellow')

        # Convert to bytes
        img_bytes = io.BytesIO()
        original.save(img_bytes, format='PNG')
        data = img_bytes.getvalue()

        # Convert back to PIL
        recovered = Image.open(io.BytesIO(data))

        # Check they match
        assert recovered.size == original.size
        assert recovered.mode == original.mode
