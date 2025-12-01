"""Factory for creating backend instances."""

import logging
from typing import Optional

from src.core.base_backend import BaseBackend
from src.backends.huggingface import HuggingFaceBackend
from src.backends.replicate import ReplicateBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory class for creating backend instances.

    This class implements the Factory pattern to create appropriate
    backend instances based on the backend type.
    """

    SUPPORTED_BACKENDS = {
        "huggingface": HuggingFaceBackend,
        "replicate": ReplicateBackend,
    }

    @classmethod
    def create_backend(
        cls,
        backend_type: str,
        api_key: str,
        model: Optional[str] = None
    ) -> BaseBackend:
        """Create a backend instance.

        Args:
            backend_type: The type of backend ("huggingface" or "replicate")
            api_key: API key for the backend service
            model: Optional model identifier

        Returns:
            An instance of the requested backend

        Raises:
            ValueError: If backend_type is not supported
            ValueError: If API key is missing
        """
        backend_type_lower = backend_type.lower()

        if backend_type_lower not in cls.SUPPORTED_BACKENDS:
            supported = ", ".join(cls.SUPPORTED_BACKENDS.keys())
            raise ValueError(
                f"Unsupported backend type: '{backend_type}'. "
                f"Supported backends: {supported}"
            )

        if not api_key:
            raise ValueError(f"API key is required for {backend_type} backend")

        backend_class = cls.SUPPORTED_BACKENDS[backend_type_lower]

        logger.info(f"Creating {backend_type} backend")

        if model:
            return backend_class(api_key=api_key, model=model)
        else:
            return backend_class(api_key=api_key)

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backend types.

        Returns:
            List of supported backend type names
        """
        return list(cls.SUPPORTED_BACKENDS.keys())

    @classmethod
    def is_supported(cls, backend_type: str) -> bool:
        """Check if a backend type is supported.

        Args:
            backend_type: The backend type to check

        Returns:
            True if supported, False otherwise
        """
        return backend_type.lower() in cls.SUPPORTED_BACKENDS
