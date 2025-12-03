"""Unit tests for image utilities."""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import io
from datetime import datetime

from src.utils.image_utils import (
    ImageFormat,
    add_metadata_to_image,
    create_downloadable_image,
    extract_metadata_from_image,
    get_image_info
)
from src.core.models import GeneratedImage


class TestImageFormat:
    """Tests for ImageFormat constants."""

    def test_format_constants(self):
        """Test that format constants are defined."""
        assert ImageFormat.PNG == "PNG"
        assert ImageFormat.JPEG == "JPEG"
        assert ImageFormat.WEBP == "WEBP"


class TestAddMetadataToImage:
    """Tests for add_metadata_to_image function."""

    def test_add_metadata_png(self):
        """Test adding metadata to PNG image."""
        # Create test image
        image = Image.new('RGB', (100, 100), color='red')
        metadata = {
            "prompt": "test prompt",
            "backend": "HuggingFace",
            "model": "test-model"
        }

        result = add_metadata_to_image(image, metadata, ImageFormat.PNG)

        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's a valid PNG
        result_image = Image.open(io.BytesIO(result))
        assert result_image.format == "PNG"

    def test_add_metadata_jpeg(self):
        """Test adding metadata to JPEG image."""
        image = Image.new('RGB', (100, 100), color='blue')
        metadata = {"prompt": "test", "backend": "Replicate"}

        result = add_metadata_to_image(image, metadata, ImageFormat.JPEG)

        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's a valid JPEG
        result_image = Image.open(io.BytesIO(result))
        assert result_image.format == "JPEG"

    def test_add_metadata_webp(self):
        """Test adding metadata to WebP image."""
        image = Image.new('RGB', (100, 100), color='green')
        metadata = {"prompt": "test", "backend": "HuggingFace"}

        result = add_metadata_to_image(image, metadata, ImageFormat.WEBP)

        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's a valid WebP
        result_image = Image.open(io.BytesIO(result))
        assert result_image.format == "WEBP"

    def test_rgba_to_jpeg_conversion(self):
        """Test that RGBA images are converted to RGB for JPEG."""
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        metadata = {"prompt": "test"}

        result = add_metadata_to_image(image, metadata, ImageFormat.JPEG)

        result_image = Image.open(io.BytesIO(result))
        assert result_image.mode == "RGB"


class TestCreateDownloadableImage:
    """Tests for create_downloadable_image function."""

    def test_create_png_download(self):
        """Test creating downloadable PNG."""
        # Create test GeneratedImage
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="A test image",
            backend="HuggingFace",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"model": "test-model"}
        )

        image_bytes, filename = create_downloadable_image(generated, ImageFormat.PNG)

        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
        assert filename.endswith('.png')
        assert "20240101_120000" in filename
        assert "A_test_image" in filename

    def test_create_jpeg_download(self):
        """Test creating downloadable JPEG."""
        test_image = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="Test prompt",
            backend="Replicate",
            metadata={}
        )

        image_bytes, filename = create_downloadable_image(generated, ImageFormat.JPEG)

        assert isinstance(image_bytes, bytes)
        assert filename.endswith('.jpeg')
        assert "Test_prompt" in filename

    def test_create_webp_download(self):
        """Test creating downloadable WebP."""
        test_image = Image.new('RGB', (100, 100), color='green')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="WebP test",
            backend="HuggingFace",
            metadata={}
        )

        image_bytes, filename = create_downloadable_image(generated, ImageFormat.WEBP)

        assert isinstance(image_bytes, bytes)
        assert filename.endswith('.webp')

    def test_filename_sanitization(self):
        """Test that special characters are sanitized in filename."""
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="Test/with\\special:chars*?",
            backend="HuggingFace",
            metadata={}
        )

        _, filename = create_downloadable_image(generated, ImageFormat.PNG)

        # Special chars should be replaced with underscores
        assert "/" not in filename
        assert "\\" not in filename
        assert ":" not in filename
        assert "*" not in filename
        assert "?" not in filename

    def test_prompt_truncation(self):
        """Test that long prompts are truncated in filename."""
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        long_prompt = "A" * 100  # Very long prompt

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt=long_prompt,
            backend="HuggingFace",
            metadata={}
        )

        _, filename = create_downloadable_image(generated, ImageFormat.PNG)

        # Filename shouldn't be too long (30 char limit on prompt + timestamp + extension)
        assert len(filename) < 100


class TestExtractMetadataFromImage:
    """Tests for extract_metadata_from_image function."""

    def test_extract_png_metadata(self):
        """Test extracting metadata from PNG."""
        # Create image with metadata
        image = Image.new('RGB', (100, 100), color='red')
        metadata = {
            "prompt": "test prompt",
            "backend": "HuggingFace"
        }

        img_bytes = add_metadata_to_image(image, metadata, ImageFormat.PNG)

        # Extract metadata
        extracted = extract_metadata_from_image(img_bytes)

        assert extracted is not None
        assert "metadata_json" in extracted or "prompt" in extracted

    def test_extract_no_metadata(self):
        """Test extracting from image with no metadata."""
        # Create image without metadata
        image = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')

        extracted = extract_metadata_from_image(img_bytes.getvalue())

        # Should return None or empty dict
        assert extracted is None or len(extracted) == 0

    def test_extract_invalid_image(self):
        """Test extracting from invalid image data."""
        invalid_data = b"not an image"

        extracted = extract_metadata_from_image(invalid_data)

        assert extracted is None


class TestGetImageInfo:
    """Tests for get_image_info function."""

    def test_get_image_info_png(self):
        """Test getting info from PNG image."""
        image = Image.new('RGB', (200, 150), color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')

        info = get_image_info(img_bytes.getvalue())

        assert info["width"] == 200
        assert info["height"] == 150
        assert info["format"] == "PNG"
        assert info["mode"] == "RGB"
        assert info["size_bytes"] > 0

    def test_get_image_info_jpeg(self):
        """Test getting info from JPEG image."""
        image = Image.new('RGB', (300, 200), color='blue')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')

        info = get_image_info(img_bytes.getvalue())

        assert info["width"] == 300
        assert info["height"] == 200
        assert info["format"] == "JPEG"
