"""Unit tests for image generator orchestrator."""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import io

from src.core.image_generator import ImageGenerator
from src.core.models import GenerationRequest, GeneratedImage


class TestImageGenerator:
    """Tests for ImageGenerator."""

    def test_initialization_no_fallback(self):
        """Test initialization without fallback backends."""
        primary = Mock()
        primary.name = "Primary"

        generator = ImageGenerator(primary)

        assert generator.primary_backend == primary
        assert generator.fallback_backends == []

    def test_initialization_with_fallback(self):
        """Test initialization with fallback backends."""
        primary = Mock()
        primary.name = "Primary"

        fallback = Mock()
        fallback.name = "Fallback"

        generator = ImageGenerator(primary, [fallback])

        assert generator.primary_backend == primary
        assert len(generator.fallback_backends) == 1
        assert generator.fallback_backends[0] == fallback

    def test_generate_image_success(self):
        """Test successful image generation with primary backend."""
        primary = Mock()
        primary.name = "Primary"

        fake_result = GeneratedImage(
            image_data=b"fake_image",
            prompt="test",
            backend="Primary"
        )
        primary.generate_image.return_value = fake_result

        generator = ImageGenerator(primary)
        request = GenerationRequest(prompt="test")

        result = generator.generate_image(request)

        assert result == fake_result
        primary.generate_image.assert_called_once()

    def test_generate_image_fallback_success(self):
        """Test fallback to secondary backend when primary fails."""
        primary = Mock()
        primary.name = "Primary"
        primary.generate_image.side_effect = RuntimeError("Primary failed")

        fallback = Mock()
        fallback.name = "Fallback"

        fake_result = GeneratedImage(
            image_data=b"fake_image",
            prompt="test",
            backend="Fallback"
        )
        fallback.generate_image.return_value = fake_result

        generator = ImageGenerator(primary, [fallback])
        request = GenerationRequest(prompt="test")

        result = generator.generate_image(request, use_fallback=True)

        assert result == fake_result
        assert result.backend == "Fallback"
        primary.generate_image.assert_called()
        fallback.generate_image.assert_called()

    def test_generate_image_no_fallback_on_auth_error(self):
        """Test that authentication errors don't trigger fallback."""
        primary = Mock()
        primary.name = "Primary"
        primary.generate_image.side_effect = ConnectionError("Auth failed")

        fallback = Mock()
        fallback.name = "Fallback"

        generator = ImageGenerator(primary, [fallback])
        request = GenerationRequest(prompt="test")

        with pytest.raises(ConnectionError, match="Auth failed"):
            generator.generate_image(request, use_fallback=True)

        # Fallback should NOT be called for auth errors
        fallback.generate_image.assert_not_called()

    def test_generate_image_all_backends_fail(self):
        """Test when all backends fail."""
        primary = Mock()
        primary.name = "Primary"
        primary.generate_image.side_effect = RuntimeError("Primary failed")

        fallback = Mock()
        fallback.name = "Fallback"
        fallback.generate_image.side_effect = RuntimeError("Fallback failed")

        generator = ImageGenerator(primary, [fallback])
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="All backends failed"):
            generator.generate_image(request, use_fallback=True)

    def test_generate_image_fallback_disabled(self):
        """Test that fallback is not used when disabled."""
        primary = Mock()
        primary.name = "Primary"
        primary.generate_image.side_effect = RuntimeError("Primary failed")

        fallback = Mock()
        fallback.name = "Fallback"

        generator = ImageGenerator(primary, [fallback])
        request = GenerationRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Image generation failed"):
            generator.generate_image(request, use_fallback=False)

        # Fallback should not be called
        fallback.generate_image.assert_not_called()

    def test_health_check_all(self):
        """Test health checking all backends."""
        primary = Mock()
        primary.name = "Primary"
        primary.health_check.return_value = True

        fallback = Mock()
        fallback.name = "Fallback"
        fallback.health_check.return_value = False

        generator = ImageGenerator(primary, [fallback])

        results = generator.health_check_all()

        assert results == {
            "Primary": True,
            "Fallback": False
        }

    def test_get_backend_names(self):
        """Test getting backend names."""
        primary = Mock()
        primary.name = "Primary"

        fallback1 = Mock()
        fallback1.name = "Fallback1"

        fallback2 = Mock()
        fallback2.name = "Fallback2"

        generator = ImageGenerator(primary, [fallback1, fallback2])

        names = generator.get_backend_names()

        assert names == {
            "primary": "Primary",
            "fallbacks": ["Fallback1", "Fallback2"]
        }

    def test_multiple_fallbacks(self):
        """Test with multiple fallback backends."""
        primary = Mock()
        primary.name = "Primary"
        primary.generate_image.side_effect = RuntimeError("Failed")

        fallback1 = Mock()
        fallback1.name = "Fallback1"
        fallback1.generate_image.side_effect = RuntimeError("Failed")

        fallback2 = Mock()
        fallback2.name = "Fallback2"

        fake_result = GeneratedImage(
            image_data=b"fake",
            prompt="test",
            backend="Fallback2"
        )
        fallback2.generate_image.return_value = fake_result

        generator = ImageGenerator(primary, [fallback1, fallback2])
        request = GenerationRequest(prompt="test")

        result = generator.generate_image(request, use_fallback=True)

        assert result.backend == "Fallback2"
        primary.generate_image.assert_called()
        fallback1.generate_image.assert_called()
        fallback2.generate_image.assert_called()
