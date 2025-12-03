"""Factory for creating backend instances using plugin system."""

import logging
from typing import Optional

from src.core.base_backend import BaseBackend
from src.core.plugin_manager import PluginManager
from src.core.plugin import BackendPlugin
from src.core.builtin_plugins import register_builtin_plugins

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory class for creating backend instances.

    This class implements the Factory pattern to create appropriate
    backend instances based on the backend type. It uses the plugin
    system to discover and load backends, allowing for extensibility
    through custom backend plugins.
    """

    _plugin_manager: Optional[PluginManager] = None

    @classmethod
    def _get_plugin_manager(cls) -> PluginManager:
        """Get or initialize the plugin manager.

        Returns:
            Initialized PluginManager instance
        """
        if cls._plugin_manager is None:
            cls._plugin_manager = PluginManager.get_instance()
            # Register built-in backend plugins
            register_builtin_plugins(cls._plugin_manager)
            # Discover external plugins
            cls._plugin_manager.discover_plugins()
            logger.info("Backend plugin system initialized")

        return cls._plugin_manager

    @classmethod
    def create_backend(
        cls,
        backend_type: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> BaseBackend:
        """Create a backend instance using the plugin system.

        Args:
            backend_type: The type of backend (e.g., "huggingface", "replicate", "local", or custom plugin name)
            api_key: API key for cloud backend services (not required for backends that don't need it)
            model: Optional model identifier

        Returns:
            An instance of the requested backend

        Raises:
            ValueError: If backend_type is not supported
            ValueError: If API key is missing for backends that require it
            RuntimeError: If plugin fails to load
        """
        backend_type_lower = backend_type.lower()
        plugin_manager = cls._get_plugin_manager()

        # Check if backend plugin is available
        if not backend_type_lower in plugin_manager.list_available_plugins():
            supported = ", ".join(plugin_manager.list_available_plugins())
            raise ValueError(
                f"Unsupported backend type: '{backend_type}'. "
                f"Supported backends: {supported}"
            )

        # Load the plugin if not already loaded
        if not plugin_manager.is_plugin_loaded(backend_type_lower):
            success = plugin_manager.load_plugin(backend_type_lower, auto_enable=True)
            if not success:
                raise RuntimeError(f"Failed to load plugin: {backend_type_lower}")

        # Get the plugin
        plugin = plugin_manager.get_plugin(backend_type_lower)
        if not isinstance(plugin, BackendPlugin):
            raise RuntimeError(f"Plugin '{backend_type_lower}' is not a backend plugin")

        # Check if API key is required
        if plugin.metadata.requires_api_key and not api_key:
            raise ValueError(f"API key is required for {backend_type} backend")

        # Get the backend class from the plugin
        backend_class = plugin.get_backend_class()

        logger.info(f"Creating {backend_type} backend from plugin")

        # Create backend instance based on requirements
        if plugin.metadata.requires_api_key:
            # Cloud backends require API key
            if model:
                return backend_class(api_key=api_key, model=model)
            else:
                return backend_class(api_key=api_key)
        else:
            # Local or other backends that don't need API key
            if model:
                return backend_class(model=model)
            else:
                return backend_class()

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backend types.

        Returns:
            List of supported backend type names (including custom plugins)
        """
        plugin_manager = cls._get_plugin_manager()
        return plugin_manager.list_available_plugins()

    @classmethod
    def is_supported(cls, backend_type: str) -> bool:
        """Check if a backend type is supported.

        Args:
            backend_type: The backend type to check

        Returns:
            True if supported, False otherwise
        """
        plugin_manager = cls._get_plugin_manager()
        return backend_type.lower() in plugin_manager.list_available_plugins()

    @classmethod
    def get_plugin_manager(cls) -> PluginManager:
        """Get the plugin manager instance.

        Returns:
            PluginManager instance used by the factory

        Note:
            Exposed for advanced use cases (e.g., manual plugin management)
        """
        return cls._get_plugin_manager()
