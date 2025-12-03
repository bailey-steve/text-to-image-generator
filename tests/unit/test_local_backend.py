"""Unit tests for LocalBackend."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import io
from PIL import Image

# Mock torch and diffusers before importing LocalBackend
sys.modules['torch'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()

from src.backends.local import LocalBackend
from src.core.models import GenerationRequest, GeneratedImage


class TestLocalBackend:
    """Tests for LocalBackend class."""

    def test_initialization(self):
        """Test backend initialization with defaults."""
        backend = LocalBackend()

        assert backend.name == "Local"
        assert backend.model == "stabilityai/sd-turbo"
        assert backend.cache_dir is None
        assert backend.pipeline is None

    def test_initialization_with_custom_model(self):
        """Test backend initialization with custom model."""
        backend = LocalBackend(model="stabilityai/sdxl-turbo")

        assert backend.model == "stabilityai/sdxl-turbo"

    def test_initialization_with_cache_dir(self):
        """Test backend initialization with custom cache directory."""
        backend = LocalBackend(cache_dir="/tmp/models")

        assert str(backend.cache_dir) == "/tmp/models"

    def test_name_property(self):
        """Test that name property returns correct value."""
        backend = LocalBackend()

        assert backend.name == "Local"

    def test_supported_models(self):
        """Test that supported_models returns correct list."""
        backend = LocalBackend()

        models = backend.supported_models
        assert "stabilityai/sd-turbo" in models
        assert "stabilityai/sdxl-turbo" in models
        assert len(models) == 2

    @patch('diffusers.AutoPipelineForText2Image')
    def test_load_pipeline(self, mock_pipeline_class):
        """Test pipeline loading."""
        backend = LocalBackend()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Load pipeline
        backend._load_pipeline()

        # Verify pipeline was loaded
        assert backend.pipeline is not None
        mock_pipeline_class.from_pretrained.assert_called_once()
        mock_pipeline.to.assert_called_with("cpu")

    @patch('diffusers.AutoPipelineForText2Image')
    def test_load_pipeline_only_once(self, mock_pipeline_class):
        """Test that pipeline is only loaded once."""
        backend = LocalBackend()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Load pipeline twice
        backend._load_pipeline()
        backend._load_pipeline()

        # Verify from_pretrained was only called once
        assert mock_pipeline_class.from_pretrained.call_count == 1

    @patch('diffusers.AutoPipelineForText2Image')
    def test_load_pipeline_missing_dependencies(self, mock_pipeline_class):
        """Test error when dependencies are missing."""
        backend = LocalBackend()

        # Simulate ImportError
        mock_pipeline_class.from_pretrained.side_effect = ImportError("torch not found")

        # Should raise ImportError with helpful message
        with pytest.raises(ImportError) as exc_info:
            backend._load_pipeline()

        assert "Missing required dependencies" in str(exc_info.value)

    @patch('torch.Generator')
    @patch('diffusers.AutoPipelineForText2Image')
    def test_generate_image_success(self, mock_pipeline_class, mock_generator_class):
        """Test successful image generation."""
        backend = LocalBackend()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Mock generated image
        test_image = Image.new('RGB', (512, 512), color='red')
        mock_result = MagicMock()
        mock_result.images = [test_image]
        mock_pipeline.return_value = mock_result

        # Create request
        request = GenerationRequest(
            prompt="A test image",
            num_inference_steps=1,
            guidance_scale=1.0,
            width=512,
            height=512
        )

        # Generate image
        result = backend.generate_image(request)

        # Verify result
        assert isinstance(result, GeneratedImage)
        assert len(result.image_data) > 0
        assert result.prompt == "A test image"
        assert result.backend == "Local"

        # Verify pipeline was called with correct parameters
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["prompt"] == "A test image"
        assert call_kwargs["num_inference_steps"] == 1
        assert call_kwargs["width"] == 512
        assert call_kwargs["height"] == 512

    @patch('torch.Generator')
    @patch('diffusers.AutoPipelineForText2Image')
    def test_generate_image_with_negative_prompt(self, mock_pipeline_class, mock_generator_class):
        """Test image generation with negative prompt."""
        backend = LocalBackend()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Mock generated image
        test_image = Image.new('RGB', (512, 512), color='blue')
        mock_result = MagicMock()
        mock_result.images = [test_image]
        mock_pipeline.return_value = mock_result

        # Create request with negative prompt
        request = GenerationRequest(
            prompt="A beautiful landscape",
            negative_prompt="ugly, blurry",
            num_inference_steps=1,
            guidance_scale=1.0,
            width=512,
            height=512
        )

        # Generate image
        backend.generate_image(request)

        # Verify negative prompt was passed
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] == "ugly, blurry"

    @patch('torch.Generator')
    @patch('diffusers.AutoPipelineForText2Image')
    def test_generate_image_with_seed(self, mock_pipeline_class, mock_generator_class):
        """Test image generation with seed."""
        backend = LocalBackend()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Mock generated image
        test_image = Image.new('RGB', (512, 512), color='green')
        mock_result = MagicMock()
        mock_result.images = [test_image]
        mock_pipeline.return_value = mock_result

        # Mock generator
        mock_gen = MagicMock()
        mock_generator_class.return_value.manual_seed.return_value = mock_gen

        # Create request with seed
        request = GenerationRequest(
            prompt="A test image",
            num_inference_steps=1,
            guidance_scale=1.0,
            width=512,
            height=512,
            seed=42
        )

        # Generate image
        backend.generate_image(request)

        # Verify generator was created with seed
        mock_generator_class.return_value.manual_seed.assert_called_with(42)

    @patch('diffusers.AutoPipelineForText2Image')
    def test_generate_image_failure(self, mock_pipeline_class):
        """Test error handling when generation fails."""
        backend = LocalBackend()

        # Mock pipeline to raise error
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.side_effect = RuntimeError("Generation failed")

        # Create request
        request = GenerationRequest(
            prompt="A test image",
            num_inference_steps=1,
            guidance_scale=1.0,
            width=512,
            height=512
        )

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            backend.generate_image(request)

        assert "Local generation failed" in str(exc_info.value)

    @patch('diffusers.AutoPipelineForText2Image')
    def test_health_check_success(self, mock_pipeline_class):
        """Test health check when backend is healthy."""
        backend = LocalBackend()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        # Health check should succeed
        assert backend.health_check() is True

    @patch('diffusers.AutoPipelineForText2Image')
    def test_health_check_failure(self, mock_pipeline_class):
        """Test health check when backend fails to load."""
        backend = LocalBackend()

        # Mock pipeline to fail
        mock_pipeline_class.from_pretrained.side_effect = RuntimeError("Failed to load")

        # Health check should fail
        assert backend.health_check() is False

    def test_set_model_valid(self):
        """Test setting a valid model."""
        backend = LocalBackend()

        backend.set_model("stabilityai/sdxl-turbo")

        assert backend.model == "stabilityai/sdxl-turbo"
        assert backend.pipeline is None  # Pipeline should be reset

    def test_set_model_invalid(self):
        """Test setting an invalid model."""
        backend = LocalBackend()

        with pytest.raises(ValueError) as exc_info:
            backend.set_model("invalid/model")

        assert "not supported" in str(exc_info.value)

    def test_repr(self):
        """Test string representation."""
        backend = LocalBackend(model="stabilityai/sd-turbo")

        assert repr(backend) == "LocalBackend(model='stabilityai/sd-turbo')"
