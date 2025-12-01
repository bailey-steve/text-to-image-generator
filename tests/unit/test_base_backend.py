"""Unit tests for base backend abstract class."""

import pytest
from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest, GeneratedImage


class ConcreteBackend(BaseBackend):
    """Concrete implementation of BaseBackend for testing."""

    def __init__(self, api_key=None):
        super().__init__(api_key)
        self._health_status = True

    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        """Mock implementation."""
        return GeneratedImage(
            image_data=b"fake_image",
            prompt=request.prompt,
            backend=self.name
        )

    def health_check(self) -> bool:
        """Mock implementation."""
        return self._health_status

    @property
    def name(self) -> str:
        """Mock implementation."""
        return "ConcreteBackend"

    @property
    def supported_models(self) -> list[str]:
        """Mock implementation."""
        return ["model1", "model2"]


class TestBaseBackend:
    """Tests for BaseBackend abstract class."""

    def test_initialization_with_api_key(self):
        """Test backend initialization with API key."""
        backend = ConcreteBackend(api_key="test_key")
        assert backend.api_key == "test_key"

    def test_initialization_without_api_key(self):
        """Test backend initialization without API key."""
        backend = ConcreteBackend()
        assert backend.api_key is None

    def test_generate_image(self):
        """Test that generate_image can be called."""
        backend = ConcreteBackend(api_key="test_key")
        request = GenerationRequest(prompt="test prompt")

        result = backend.generate_image(request)

        assert isinstance(result, GeneratedImage)
        assert result.prompt == "test prompt"
        assert result.backend == "ConcreteBackend"

    def test_health_check(self):
        """Test that health_check can be called."""
        backend = ConcreteBackend(api_key="test_key")

        assert backend.health_check() is True

        backend._health_status = False
        assert backend.health_check() is False

    def test_name_property(self):
        """Test that name property returns correct value."""
        backend = ConcreteBackend(api_key="test_key")
        assert backend.name == "ConcreteBackend"

    def test_supported_models_property(self):
        """Test that supported_models property returns list."""
        backend = ConcreteBackend(api_key="test_key")
        models = backend.supported_models

        assert isinstance(models, list)
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_repr(self):
        """Test string representation."""
        backend = ConcreteBackend(api_key="test_key")
        repr_str = repr(backend)

        assert "ConcreteBackend" in repr_str
        assert "name='ConcreteBackend'" in repr_str

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBackend(api_key="test")


class IncompleteBackend(BaseBackend):
    """Incomplete backend missing some abstract methods."""

    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        pass


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_missing_abstract_methods(self):
        """Test that missing abstract methods prevent instantiation."""
        with pytest.raises(TypeError):
            IncompleteBackend(api_key="test")
