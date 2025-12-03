"""Built-in backend plugins for HuggingFace, Replicate, and Local backends."""

import logging
from typing import Type

from src.core.plugin import BackendPlugin, PluginMetadata, PluginType
from src.core.base_backend import BaseBackend
from src.backends.huggingface import HuggingFaceBackend
from src.backends.replicate import ReplicateBackend
from src.backends.local import LocalBackend

logger = logging.getLogger(__name__)


class HuggingFacePlugin(BackendPlugin):
    """Built-in plugin for HuggingFace backend."""

    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="huggingface",
            display_name="HuggingFace",
            version="1.0.0",
            author="Bailey Steve",
            description="Generate images using HuggingFace Inference API",
            plugin_type=PluginType.BACKEND,
            dependencies=["requests"],
            requires_api_key=True
        )

    def initialize(self) -> bool:
        """Initialize the plugin."""
        logger.info("HuggingFace plugin initialized")
        return True

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        logger.info("HuggingFace plugin cleaned up")

    def get_backend_class(self) -> Type[BaseBackend]:
        """Get the backend class."""
        return HuggingFaceBackend


class ReplicatePlugin(BackendPlugin):
    """Built-in plugin for Replicate backend."""

    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="replicate",
            display_name="Replicate",
            version="1.0.0",
            author="Bailey Steve",
            description="Generate images using Replicate API",
            plugin_type=PluginType.BACKEND,
            dependencies=["replicate"],
            requires_api_key=True
        )

    def initialize(self) -> bool:
        """Initialize the plugin."""
        logger.info("Replicate plugin initialized")
        return True

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        logger.info("Replicate plugin cleaned up")

    def get_backend_class(self) -> Type[BaseBackend]:
        """Get the backend class."""
        return ReplicateBackend


class LocalPlugin(BackendPlugin):
    """Built-in plugin for Local backend."""

    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="local",
            display_name="Local",
            version="1.0.0",
            author="Bailey Steve",
            description="Generate images locally using Diffusers (CPU/GPU)",
            plugin_type=PluginType.BACKEND,
            dependencies=["torch", "diffusers", "transformers", "accelerate"],
            requires_api_key=False
        )

    def initialize(self) -> bool:
        """Initialize the plugin."""
        logger.info("Local plugin initialized")
        return True

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        logger.info("Local plugin cleaned up")

    def get_backend_class(self) -> Type[BaseBackend]:
        """Get the backend class."""
        return LocalBackend


def register_builtin_plugins(plugin_manager) -> None:
    """Register all built-in plugins with the plugin manager.

    Args:
        plugin_manager: PluginManager instance to register plugins with
    """
    plugin_manager.register_builtin_plugin("huggingface", HuggingFacePlugin)
    plugin_manager.register_builtin_plugin("replicate", ReplicatePlugin)
    plugin_manager.register_builtin_plugin("local", LocalPlugin)

    logger.info("Registered 3 built-in backend plugins")
