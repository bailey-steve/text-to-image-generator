"""Image generation orchestrator with fallback support."""

import logging
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest, GeneratedImage

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Orchestrator for image generation with fallback support.

    This class manages multiple backends and provides automatic fallback
    if the primary backend fails. It also implements retry logic for
    transient failures.

    Attributes:
        primary_backend: The primary backend to use
        fallback_backends: List of fallback backends to try if primary fails
    """

    def __init__(
        self,
        primary_backend: BaseBackend,
        fallback_backends: Optional[List[BaseBackend]] = None
    ):
        """Initialize the image generator.

        Args:
            primary_backend: The primary backend to use for generation
            fallback_backends: Optional list of fallback backends
        """
        self.primary_backend = primary_backend
        self.fallback_backends = fallback_backends or []

        logger.info(
            f"Initialized ImageGenerator with primary: {primary_backend.name}, "
            f"fallbacks: {[b.name for b in self.fallback_backends]}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True
    )
    def _generate_with_retry(
        self,
        backend: BaseBackend,
        request: GenerationRequest
    ) -> GeneratedImage:
        """Generate image with retry logic for transient failures.

        Args:
            backend: The backend to use
            request: The generation request

        Returns:
            Generated image

        Raises:
            RuntimeError: If generation fails after retries
            ConnectionError: If authentication fails (no retry)
        """
        return backend.generate_image(request)

    def generate_image(
        self,
        request: GenerationRequest,
        use_fallback: bool = True
    ) -> GeneratedImage:
        """Generate an image with automatic fallback.

        Attempts to generate an image using the primary backend.
        If it fails and use_fallback is True, tries each fallback
        backend in order until one succeeds.

        Args:
            request: The generation request
            use_fallback: Whether to use fallback backends on failure

        Returns:
            Generated image

        Raises:
            RuntimeError: If all backends fail
            ConnectionError: If authentication fails on primary backend
        """
        # Try primary backend
        try:
            logger.info(f"Attempting generation with primary backend: {self.primary_backend.name}")
            result = self._generate_with_retry(self.primary_backend, request)
            logger.info(f"Successfully generated image with {self.primary_backend.name}")
            return result

        except ConnectionError as e:
            # Authentication errors shouldn't trigger fallback
            logger.error(f"Authentication error with {self.primary_backend.name}: {e}")
            raise

        except Exception as e:
            logger.warning(
                f"Primary backend {self.primary_backend.name} failed: {e}"
            )

            if not use_fallback or not self.fallback_backends:
                logger.error("No fallback backends available")
                raise RuntimeError(
                    f"Image generation failed with {self.primary_backend.name}: {e}"
                ) from e

            # Try fallback backends
            for i, fallback in enumerate(self.fallback_backends, 1):
                try:
                    logger.info(
                        f"Attempting fallback {i}/{len(self.fallback_backends)}: "
                        f"{fallback.name}"
                    )
                    result = self._generate_with_retry(fallback, request)
                    logger.info(f"Successfully generated image with fallback: {fallback.name}")
                    return result

                except Exception as fallback_error:
                    logger.warning(f"Fallback backend {fallback.name} failed: {fallback_error}")
                    continue

            # All backends failed
            error_msg = (
                f"All backends failed. Primary: {self.primary_backend.name}, "
                f"Fallbacks: {[b.name for b in self.fallback_backends]}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def health_check_all(self) -> dict[str, bool]:
        """Check health of all configured backends.

        Returns:
            Dictionary mapping backend names to health status
        """
        results = {}

        # Check primary
        logger.debug(f"Health checking primary backend: {self.primary_backend.name}")
        results[self.primary_backend.name] = self.primary_backend.health_check()

        # Check fallbacks
        for fallback in self.fallback_backends:
            logger.debug(f"Health checking fallback backend: {fallback.name}")
            results[fallback.name] = fallback.health_check()

        logger.info(f"Health check results: {results}")
        return results

    def get_backend_names(self) -> dict[str, List[str]]:
        """Get names of all configured backends.

        Returns:
            Dictionary with 'primary' and 'fallbacks' lists
        """
        return {
            "primary": self.primary_backend.name,
            "fallbacks": [b.name for b in self.fallback_backends]
        }
