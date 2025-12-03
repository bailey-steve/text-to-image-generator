"""Plugin definition for Dummy backend.

This file is required for all plugins. It must:
1. Be named __plugin__.py
2. Define a class named 'Plugin' that inherits from BackendPlugin
3. Implement all required methods
"""

import logging
from typing import Type

from src.core.plugin import BackendPlugin, PluginMetadata, PluginType
from src.core.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class Plugin(BackendPlugin):
    """Dummy backend plugin for testing and demonstration.

    This plugin provides a simple backend that generates colored rectangles
    instead of using AI models. It's useful for:
    - Testing the plugin system
    - Demonstrating how to create custom backends
    - Fast development without API keys or heavy dependencies
    """

    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="dummy_backend",
            display_name="Dummy Backend",
            version="1.0.0",
            author="Example Developer",
            description="Simple backend that generates colored rectangles for testing",
            plugin_type=PluginType.BACKEND,
            dependencies=["PIL"],  # Pillow for image generation
            requires_api_key=False
        )

    def initialize(self) -> bool:
        """Initialize the plugin.

        Returns:
            True if initialization succeeded
        """
        logger.info("Dummy backend plugin initialized")
        return True

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        logger.info("Dummy backend plugin cleaned up")

    def get_backend_class(self) -> Type[BaseBackend]:
        """Get the backend class provided by this plugin.

        Returns:
            DummyBackend class
        """
        from plugins.dummy_backend.backend import DummyBackend
        return DummyBackend
