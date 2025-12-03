"""Base classes for plugin system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict
from enum import Enum


class PluginType(Enum):
    """Types of plugins supported by the application."""
    BACKEND = "backend"
    FILTER = "filter"
    EXTENSION = "extension"


@dataclass
class PluginMetadata:
    """Metadata describing a plugin.

    Attributes:
        name: Unique identifier for the plugin (lowercase, no spaces)
        display_name: Human-readable name for UI display
        version: Plugin version (semantic versioning)
        author: Plugin author name
        description: Brief description of plugin functionality
        plugin_type: Type of plugin (backend, filter, extension)
        dependencies: List of required Python packages
        requires_api_key: Whether plugin requires an API key
    """
    name: str
    display_name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: list[str] = None
    requires_api_key: bool = False

    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.dependencies is None:
            self.dependencies = []

        # Validate name is lowercase with no spaces
        if not self.name.islower() or ' ' in self.name:
            raise ValueError(
                f"Plugin name must be lowercase with no spaces: '{self.name}'"
            )


class BasePlugin(ABC):
    """Abstract base class for all plugins.

    All plugins must inherit from this class and implement the required methods.
    Plugins are discovered and loaded by the PluginManager.

    Attributes:
        metadata: Plugin metadata (name, version, author, etc.)
        enabled: Whether the plugin is currently enabled
        config: Plugin-specific configuration dictionary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the plugin.

        Args:
            config: Optional configuration dictionary for the plugin
        """
        self.enabled = False
        self.config = config or {}

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata.

        Returns:
            PluginMetadata describing this plugin
        """
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin.

        Called when the plugin is first loaded. Should perform any
        necessary setup (loading models, connecting to services, etc.).

        Returns:
            True if initialization succeeded, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources.

        Called when the plugin is disabled or the application shuts down.
        Should release any resources held by the plugin.
        """
        pass

    def enable(self) -> bool:
        """Enable the plugin.

        Returns:
            True if plugin was enabled successfully
        """
        if self.enabled:
            return True

        success = self.initialize()
        if success:
            self.enabled = True
        return success

    def disable(self) -> None:
        """Disable the plugin."""
        if not self.enabled:
            return

        self.cleanup()
        self.enabled = False

    def validate_dependencies(self) -> tuple[bool, list[str]]:
        """Check if all required dependencies are installed.

        Returns:
            Tuple of (all_installed, missing_packages)
        """
        missing = []

        for package in self.metadata.dependencies:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        return (len(missing) == 0, missing)

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return f"{self.metadata.display_name} v{self.metadata.version} ({status})"


class BackendPlugin(BasePlugin):
    """Base class for backend plugins.

    Backend plugins provide image generation capabilities from different
    sources (APIs, local models, etc.). They must implement the backend
    interface defined in BaseBackend.

    Example:
        class MyBackendPlugin(BackendPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mybackend",
                    display_name="My Backend",
                    version="1.0.0",
                    author="Your Name",
                    description="Custom backend plugin",
                    plugin_type=PluginType.BACKEND,
                    requires_api_key=True
                )

            def get_backend_class(self):
                from my_backend import MyBackend
                return MyBackend
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata (must specify BACKEND type)."""
        meta = self._get_metadata()
        if meta.plugin_type != PluginType.BACKEND:
            raise ValueError("BackendPlugin must have plugin_type=PluginType.BACKEND")
        return meta

    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata implementation."""
        pass

    @abstractmethod
    def get_backend_class(self) -> type:
        """Get the backend class provided by this plugin.

        Returns:
            A class that inherits from BaseBackend
        """
        pass
