"""Abstract base class for image generation backends."""

from abc import ABC, abstractmethod
from typing import Optional
from .models import GenerationRequest, GeneratedImage


class BaseBackend(ABC):
    """Abstract interface that all image generation backends must implement.

    This defines the contract for backend implementations, allowing the application
    to swap between different backends (HuggingFace, Replicate, local models) without
    changing the core application logic.

    Attributes:
        api_key: Optional API key for cloud-based backends
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the backend.

        Args:
            api_key: Optional API key for authentication with cloud services
        """
        self.api_key = api_key

    @abstractmethod
    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        """Generate an image from a text prompt.

        Args:
            request: The generation request containing prompt and parameters

        Returns:
            GeneratedImage containing the image data and metadata

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the generation fails
            ConnectionError: If unable to connect to the backend service
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is available and working.

        Returns:
            True if the backend is healthy and can generate images, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the human-readable name of this backend.

        Returns:
            The backend name (e.g., "HuggingFace", "Replicate", "Local")
        """
        pass

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Get a list of models supported by this backend.

        Returns:
            List of model identifiers available for this backend
        """
        pass

    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"{self.__class__.__name__}(name='{self.name}')"
