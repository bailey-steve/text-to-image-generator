"""Plugin manager for discovering and managing plugins."""

import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Dict, List, Type
import sys

from src.core.plugin import BasePlugin, BackendPlugin, PluginType, PluginMetadata

logger = logging.getLogger(__name__)


class PluginManager:
    """Singleton manager for discovering, loading, and managing plugins.

    The PluginManager is responsible for:
    - Discovering plugins from the plugins directory
    - Loading and initializing plugins
    - Managing plugin lifecycle (enable/disable)
    - Providing access to loaded plugins

    Usage:
        manager = PluginManager.get_instance()
        manager.discover_plugins()
        manager.load_plugin("myplugin")
        plugin = manager.get_plugin("myplugin")
    """

    _instance: Optional['PluginManager'] = None

    def __init__(self, plugins_dir: Optional[Path] = None):
        """Initialize the plugin manager.

        Args:
            plugins_dir: Directory containing plugins (defaults to ./plugins)
        """
        if plugins_dir is None:
            # Default to plugins/ in project root
            project_root = Path(__file__).parent.parent.parent
            plugins_dir = project_root / "plugins"

        self.plugins_dir = Path(plugins_dir)
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        self._builtin_plugins: Dict[str, Type[BasePlugin]] = {}

        logger.info(f"PluginManager initialized with plugins dir: {self.plugins_dir}")

    @classmethod
    def get_instance(cls, plugins_dir: Optional[Path] = None) -> 'PluginManager':
        """Get the singleton instance of PluginManager.

        Args:
            plugins_dir: Directory containing plugins (only used on first call)

        Returns:
            The singleton PluginManager instance
        """
        if cls._instance is None:
            cls._instance = cls(plugins_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def register_builtin_plugin(
        self,
        plugin_name: str,
        plugin_class: Type[BasePlugin]
    ) -> None:
        """Register a built-in plugin class.

        Built-in plugins are plugins that come with the application
        (e.g., HuggingFace, Replicate, Local backends). They don't need
        to be discovered from the plugins directory.

        Args:
            plugin_name: Unique name for the plugin (lowercase)
            plugin_class: Plugin class to register
        """
        if plugin_name in self._builtin_plugins:
            logger.warning(f"Built-in plugin '{plugin_name}' already registered, overwriting")

        self._builtin_plugins[plugin_name] = plugin_class
        logger.info(f"Registered built-in plugin: {plugin_name}")

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugins directory.

        Searches for Python packages in the plugins directory that contain
        a __plugin__.py file defining a Plugin class.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        if not self.plugins_dir.exists():
            logger.info(f"Plugins directory does not exist: {self.plugins_dir}")
            logger.info("Creating plugins directory...")
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            return discovered

        # Search for plugin packages
        for item in self.plugins_dir.iterdir():
            if not item.is_dir():
                continue

            # Look for __plugin__.py
            plugin_file = item / "__plugin__.py"
            if not plugin_file.exists():
                logger.debug(f"Skipping {item.name}: no __plugin__.py found")
                continue

            try:
                plugin_name = item.name
                plugin_class = self._load_plugin_class(plugin_name, plugin_file)

                if plugin_class is not None:
                    self._plugin_classes[plugin_name] = plugin_class
                    discovered.append(plugin_name)
                    logger.info(f"Discovered plugin: {plugin_name}")

            except Exception as e:
                logger.error(f"Failed to discover plugin {item.name}: {e}")

        logger.info(f"Discovered {len(discovered)} plugin(s): {', '.join(discovered)}")
        return discovered

    def _load_plugin_class(
        self,
        plugin_name: str,
        plugin_file: Path
    ) -> Optional[Type[BasePlugin]]:
        """Load a plugin class from a file.

        Args:
            plugin_name: Name of the plugin
            plugin_file: Path to __plugin__.py file

        Returns:
            Plugin class if successful, None otherwise
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}",
                plugin_file
            )
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create module spec for {plugin_name}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{plugin_name}"] = module
            spec.loader.exec_module(module)

            # Look for a class that inherits from BasePlugin
            plugin_class = getattr(module, "Plugin", None)
            if plugin_class is None:
                logger.error(f"Plugin {plugin_name} does not define 'Plugin' class")
                return None

            if not issubclass(plugin_class, BasePlugin):
                logger.error(f"Plugin {plugin_name} 'Plugin' class does not inherit from BasePlugin")
                return None

            return plugin_class

        except Exception as e:
            logger.error(f"Failed to load plugin class for {plugin_name}: {e}")
            return None

    def load_plugin(
        self,
        plugin_name: str,
        config: Optional[Dict] = None,
        auto_enable: bool = True
    ) -> bool:
        """Load and optionally enable a plugin.

        Args:
            plugin_name: Name of the plugin to load
            config: Optional configuration for the plugin
            auto_enable: Whether to automatically enable the plugin after loading

        Returns:
            True if plugin was loaded successfully, False otherwise
        """
        if plugin_name in self._plugins:
            logger.info(f"Plugin '{plugin_name}' already loaded")
            return True

        # Check built-in plugins first
        if plugin_name in self._builtin_plugins:
            plugin_class = self._builtin_plugins[plugin_name]
        elif plugin_name in self._plugin_classes:
            plugin_class = self._plugin_classes[plugin_name]
        else:
            logger.error(f"Plugin '{plugin_name}' not found. Run discover_plugins() first.")
            return False

        try:
            # Instantiate the plugin
            plugin = plugin_class(config=config)

            # Validate dependencies
            all_installed, missing = plugin.validate_dependencies()
            if not all_installed:
                logger.error(
                    f"Plugin '{plugin_name}' has missing dependencies: {', '.join(missing)}"
                )
                return False

            # Store the plugin instance
            self._plugins[plugin_name] = plugin
            logger.info(f"Loaded plugin: {plugin}")

            # Enable if requested
            if auto_enable:
                success = plugin.enable()
                if not success:
                    logger.error(f"Failed to enable plugin '{plugin_name}'")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to load plugin '{plugin_name}': {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if plugin was unloaded successfully
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' not loaded")
            return False

        plugin = self._plugins[plugin_name]
        plugin.disable()
        del self._plugins[plugin_name]
        logger.info(f"Unloaded plugin: {plugin_name}")
        return True

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance if found and loaded, None otherwise
        """
        return self._plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all loaded plugins of a specific type.

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            List of plugin instances matching the type
        """
        return [
            plugin for plugin in self._plugins.values()
            if plugin.metadata.plugin_type == plugin_type
        ]

    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """Get all loaded plugins.

        Returns:
            Dictionary mapping plugin names to plugin instances
        """
        return self._plugins.copy()

    def get_backend_plugins(self) -> List[BackendPlugin]:
        """Get all loaded backend plugins.

        Returns:
            List of BackendPlugin instances
        """
        plugins = self.get_plugins_by_type(PluginType.BACKEND)
        return [p for p in plugins if isinstance(p, BackendPlugin)]

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is currently loaded.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if plugin is loaded, False otherwise
        """
        return plugin_name in self._plugins

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is currently enabled.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if plugin is loaded and enabled, False otherwise
        """
        plugin = self.get_plugin(plugin_name)
        return plugin is not None and plugin.enabled

    def list_available_plugins(self) -> List[str]:
        """List all available plugins (discovered + built-in).

        Returns:
            List of plugin names
        """
        return list(set(self._plugin_classes.keys()) | set(self._builtin_plugins.keys()))

    def __repr__(self) -> str:
        """String representation."""
        return f"PluginManager(loaded={len(self._plugins)}, available={len(self.list_available_plugins())})"
