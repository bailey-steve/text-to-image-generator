"""Unit tests for PluginManager."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.core.plugin_manager import PluginManager
from src.core.plugin import BasePlugin, BackendPlugin, PluginMetadata, PluginType
from src.core.base_backend import BaseBackend


class TestPluginManager:
    """Tests for PluginManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset singleton before each test
        PluginManager.reset_instance()

        # Create temporary plugins directory
        self.temp_dir = tempfile.mkdtemp()
        self.plugins_dir = Path(self.temp_dir) / "plugins"
        self.plugins_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

        # Reset singleton
        PluginManager.reset_instance()

    def test_singleton_instance(self):
        """Test that PluginManager is a singleton."""
        manager1 = PluginManager.get_instance(self.plugins_dir)
        manager2 = PluginManager.get_instance()

        assert manager1 is manager2

    def test_reset_instance(self):
        """Test resetting singleton instance."""
        manager1 = PluginManager.get_instance(self.plugins_dir)
        PluginManager.reset_instance()
        manager2 = PluginManager.get_instance(self.plugins_dir)

        assert manager1 is not manager2

    def test_initialization(self):
        """Test PluginManager initialization."""
        manager = PluginManager(self.plugins_dir)

        assert manager.plugins_dir == self.plugins_dir
        assert len(manager.get_all_plugins()) == 0

    def test_register_builtin_plugin(self):
        """Test registering a built-in plugin."""
        manager = PluginManager(self.plugins_dir)

        # Create a mock plugin class
        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)

        available = manager.list_available_plugins()
        assert "mock" in available

    def test_discover_plugins_empty_directory(self):
        """Test discovering plugins in empty directory."""
        manager = PluginManager(self.plugins_dir)

        discovered = manager.discover_plugins()

        assert discovered == []

    def test_discover_plugins_nonexistent_directory(self):
        """Test discovering plugins when directory doesn't exist."""
        non_existent = self.plugins_dir / "nonexistent"
        manager = PluginManager(non_existent)

        discovered = manager.discover_plugins()

        assert discovered == []
        assert non_existent.exists()  # Should create directory

    def test_load_builtin_plugin(self):
        """Test loading a built-in plugin."""
        manager = PluginManager(self.plugins_dir)

        # Create mock plugin class
        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION,
                    dependencies=[]  # No dependencies
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)

        # Load the plugin
        success = manager.load_plugin("mock")

        assert success is True
        assert manager.is_plugin_loaded("mock")
        assert manager.is_plugin_enabled("mock")

    def test_load_plugin_not_found(self):
        """Test loading a plugin that doesn't exist."""
        manager = PluginManager(self.plugins_dir)

        success = manager.load_plugin("nonexistent")

        assert success is False

    def test_load_plugin_already_loaded(self):
        """Test loading a plugin that's already loaded."""
        manager = PluginManager(self.plugins_dir)

        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)
        manager.load_plugin("mock")

        # Load again
        success = manager.load_plugin("mock")

        assert success is True

    def test_load_plugin_with_missing_dependencies(self):
        """Test loading plugin with missing dependencies."""
        manager = PluginManager(self.plugins_dir)

        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION,
                    dependencies=["nonexistent_package_12345"]
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)

        # Should fail due to missing dependency
        success = manager.load_plugin("mock")

        assert success is False
        assert not manager.is_plugin_loaded("mock")

    def test_load_plugin_without_auto_enable(self):
        """Test loading plugin without auto-enabling."""
        manager = PluginManager(self.plugins_dir)

        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)

        # Load without enabling
        success = manager.load_plugin("mock", auto_enable=False)

        assert success is True
        assert manager.is_plugin_loaded("mock")
        assert not manager.is_plugin_enabled("mock")

    def test_unload_plugin(self):
        """Test unloading a plugin."""
        manager = PluginManager(self.plugins_dir)

        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)
        manager.load_plugin("mock")

        # Unload
        success = manager.unload_plugin("mock")

        assert success is True
        assert not manager.is_plugin_loaded("mock")

    def test_unload_plugin_not_loaded(self):
        """Test unloading a plugin that's not loaded."""
        manager = PluginManager(self.plugins_dir)

        success = manager.unload_plugin("nonexistent")

        assert success is False

    def test_get_plugin(self):
        """Test getting a loaded plugin."""
        manager = PluginManager(self.plugins_dir)

        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)
        manager.load_plugin("mock")

        plugin = manager.get_plugin("mock")

        assert plugin is not None
        assert isinstance(plugin, MockPlugin)

    def test_get_plugin_not_loaded(self):
        """Test getting a plugin that's not loaded."""
        manager = PluginManager(self.plugins_dir)

        plugin = manager.get_plugin("nonexistent")

        assert plugin is None

    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        manager = PluginManager(self.plugins_dir)

        # Create backend plugin
        class MockBackendPlugin(BackendPlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="backend1",
                    display_name="Backend 1",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.BACKEND
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

            def get_backend_class(self):
                return Mock

        # Create extension plugin
        class MockExtensionPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="extension1",
                    display_name="Extension 1",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("backend1", MockBackendPlugin)
        manager.register_builtin_plugin("extension1", MockExtensionPlugin)
        manager.load_plugin("backend1")
        manager.load_plugin("extension1")

        # Get backend plugins
        backend_plugins = manager.get_plugins_by_type(PluginType.BACKEND)

        assert len(backend_plugins) == 1
        assert backend_plugins[0].metadata.name == "backend1"

    def test_get_backend_plugins(self):
        """Test getting backend plugins specifically."""
        manager = PluginManager(self.plugins_dir)

        class MockBackendPlugin(BackendPlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="backend1",
                    display_name="Backend 1",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.BACKEND
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

            def get_backend_class(self):
                return Mock

        manager.register_builtin_plugin("backend1", MockBackendPlugin)
        manager.load_plugin("backend1")

        backend_plugins = manager.get_backend_plugins()

        assert len(backend_plugins) == 1
        assert isinstance(backend_plugins[0], BackendPlugin)

    def test_list_available_plugins(self):
        """Test listing available plugins."""
        manager = PluginManager(self.plugins_dir)

        class MockPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="mock",
                    display_name="Mock",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        manager.register_builtin_plugin("mock", MockPlugin)

        available = manager.list_available_plugins()

        assert "mock" in available

    def test_plugin_manager_repr(self):
        """Test PluginManager string representation."""
        manager = PluginManager(self.plugins_dir)

        repr_str = repr(manager)

        assert "PluginManager" in repr_str
        assert "loaded=" in repr_str
        assert "available=" in repr_str
