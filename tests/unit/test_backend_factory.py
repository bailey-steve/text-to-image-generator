"""Unit tests for backend factory."""

import pytest
from unittest.mock import Mock, patch

from src.core.backend_factory import BackendFactory
from src.core.plugin_manager import PluginManager
from src.backends.huggingface import HuggingFaceBackend
from src.backends.replicate import ReplicateBackend
from src.backends.local import LocalBackend


class TestBackendFactory:
    """Tests for BackendFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset plugin manager and factory before each test
        PluginManager.reset_instance()
        BackendFactory._plugin_manager = None

    def teardown_method(self):
        """Clean up test fixtures."""
        # Reset plugin manager and factory after each test
        PluginManager.reset_instance()
        BackendFactory._plugin_manager = None

    def test_get_supported_backends(self):
        """Test getting list of supported backends."""
        backends = BackendFactory.get_supported_backends()

        assert isinstance(backends, list)
        assert "huggingface" in backends
        assert "replicate" in backends
        assert "local" in backends

    def test_is_supported_huggingface(self):
        """Test checking if huggingface is supported."""
        assert BackendFactory.is_supported("huggingface")
        assert BackendFactory.is_supported("HuggingFace")
        assert BackendFactory.is_supported("HUGGINGFACE")

    def test_is_supported_replicate(self):
        """Test checking if replicate is supported."""
        assert BackendFactory.is_supported("replicate")
        assert BackendFactory.is_supported("Replicate")

    def test_is_not_supported(self):
        """Test checking unsupported backend."""
        assert not BackendFactory.is_supported("unknown")
        assert not BackendFactory.is_supported("openai")

    def test_create_huggingface_backend(self):
        """Test creating HuggingFace backend."""
        result = BackendFactory.create_backend("huggingface", "test_token")

        assert isinstance(result, HuggingFaceBackend)
        assert result.api_key == "test_token"
        assert result.name == "HuggingFace"

    def test_create_replicate_backend(self):
        """Test creating Replicate backend."""
        with patch('src.backends.replicate.replicate.Client'):
            result = BackendFactory.create_backend("replicate", "test_token")

            assert isinstance(result, ReplicateBackend)
            assert result.api_key == "test_token"
            assert result.name == "Replicate"

    def test_create_backend_with_model(self):
        """Test creating backend with custom model."""
        custom_model = "custom-model"
        result = BackendFactory.create_backend(
            "huggingface",
            "test_token",
            model=custom_model
        )

        assert isinstance(result, HuggingFaceBackend)
        assert result.model == custom_model

    def test_create_unsupported_backend(self):
        """Test that creating unsupported backend raises error."""
        with pytest.raises(ValueError, match="Unsupported backend type"):
            BackendFactory.create_backend("unknown", "test_token")

    def test_create_backend_empty_api_key(self):
        """Test that empty API key raises error."""
        with pytest.raises(ValueError, match="API key is required"):
            BackendFactory.create_backend("huggingface", "")

    def test_case_insensitive_backend_type(self):
        """Test that backend type is case insensitive."""
        result1 = BackendFactory.create_backend("HUGGINGFACE", "token")
        assert isinstance(result1, HuggingFaceBackend)

        result2 = BackendFactory.create_backend("HuggingFace", "token")
        assert isinstance(result2, HuggingFaceBackend)

        with patch('src.backends.replicate.replicate.Client'):
            result3 = BackendFactory.create_backend("REPLICATE", "token")
            assert isinstance(result3, ReplicateBackend)

            result4 = BackendFactory.create_backend("Replicate", "token")
            assert isinstance(result4, ReplicateBackend)

    def test_is_supported_local(self):
        """Test checking if local is supported."""
        assert BackendFactory.is_supported("local")
        assert BackendFactory.is_supported("Local")
        assert BackendFactory.is_supported("LOCAL")

    def test_create_local_backend(self):
        """Test creating Local backend."""
        result = BackendFactory.create_backend("local")

        assert isinstance(result, LocalBackend)
        assert result.name == "Local"
        assert result.model == "stabilityai/sd-turbo"

    def test_create_local_backend_with_model(self):
        """Test creating Local backend with custom model."""
        result = BackendFactory.create_backend("local", model="stabilityai/sdxl-turbo")

        assert isinstance(result, LocalBackend)
        assert result.model == "stabilityai/sdxl-turbo"

    def test_create_local_backend_no_api_key_required(self):
        """Test that local backend doesn't require API key."""
        # Should not raise error
        result = BackendFactory.create_backend("local", api_key=None)
        assert isinstance(result, LocalBackend)

    def test_create_cloud_backend_requires_api_key(self):
        """Test that cloud backends still require API keys."""
        with pytest.raises(ValueError, match="API key is required"):
            BackendFactory.create_backend("huggingface", api_key=None)

        with pytest.raises(ValueError, match="API key is required"):
            BackendFactory.create_backend("replicate", api_key=None)
