"""Unit tests for plugin base classes."""

import pytest
from typing import Type
from src.core.plugin import (
    BasePlugin,
    BackendPlugin,
    PluginMetadata,
    PluginType
)
from src.core.base_backend import BaseBackend


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="testplugin",
            display_name="Test Plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            plugin_type=PluginType.BACKEND
        )

        assert metadata.name == "testplugin"
        assert metadata.display_name == "Test Plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "A test plugin"
        assert metadata.plugin_type == PluginType.BACKEND
        assert metadata.dependencies == []
        assert metadata.requires_api_key is False

    def test_metadata_with_dependencies(self):
        """Test metadata with dependencies."""
        metadata = PluginMetadata(
            name="testplugin",
            display_name="Test Plugin",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type=PluginType.BACKEND,
            dependencies=["requests", "pillow"],
            requires_api_key=True
        )

        assert metadata.dependencies == ["requests", "pillow"]
        assert metadata.requires_api_key is True

    def test_invalid_name_uppercase(self):
        """Test that uppercase names are rejected."""
        with pytest.raises(ValueError, match="lowercase"):
            PluginMetadata(
                name="TestPlugin",
                display_name="Test Plugin",
                version="1.0.0",
                author="Test",
                description="Test",
                plugin_type=PluginType.BACKEND
            )

    def test_invalid_name_with_spaces(self):
        """Test that names with spaces are rejected."""
        with pytest.raises(ValueError, match="lowercase"):
            PluginMetadata(
                name="test plugin",
                display_name="Test Plugin",
                version="1.0.0",
                author="Test",
                description="Test",
                plugin_type=PluginType.BACKEND
            )


class MockPlugin(BasePlugin):
    """Mock plugin for testing."""

    def __init__(self, config=None, should_initialize=True):
        super().__init__(config)
        self.should_initialize = should_initialize
        self.initialized_count = 0
        self.cleanup_count = 0

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="mockplugin",
            display_name="Mock Plugin",
            version="1.0.0",
            author="Test",
            description="Mock plugin for testing",
            plugin_type=PluginType.EXTENSION,
            dependencies=["requests"]
        )

    def initialize(self) -> bool:
        self.initialized_count += 1
        return self.should_initialize

    def cleanup(self) -> None:
        self.cleanup_count += 1


class TestBasePlugin:
    """Tests for BasePlugin class."""

    def test_plugin_creation(self):
        """Test creating a plugin instance."""
        plugin = MockPlugin()

        assert not plugin.enabled
        assert plugin.config == {}
        assert plugin.initialized_count == 0
        assert plugin.cleanup_count == 0

    def test_plugin_with_config(self):
        """Test creating plugin with config."""
        config = {"setting": "value"}
        plugin = MockPlugin(config=config)

        assert plugin.config == config

    def test_enable_plugin(self):
        """Test enabling a plugin."""
        plugin = MockPlugin(should_initialize=True)

        success = plugin.enable()

        assert success is True
        assert plugin.enabled is True
        assert plugin.initialized_count == 1

    def test_enable_already_enabled(self):
        """Test enabling an already enabled plugin."""
        plugin = MockPlugin(should_initialize=True)
        plugin.enable()

        # Enable again
        plugin.enable()

        # Should only initialize once
        assert plugin.initialized_count == 1

    def test_enable_fails(self):
        """Test enabling a plugin that fails to initialize."""
        plugin = MockPlugin(should_initialize=False)

        success = plugin.enable()

        assert success is False
        assert plugin.enabled is False
        assert plugin.initialized_count == 1

    def test_disable_plugin(self):
        """Test disabling a plugin."""
        plugin = MockPlugin(should_initialize=True)
        plugin.enable()

        plugin.disable()

        assert plugin.enabled is False
        assert plugin.cleanup_count == 1

    def test_disable_not_enabled(self):
        """Test disabling a plugin that's not enabled."""
        plugin = MockPlugin()

        plugin.disable()

        # Should not call cleanup
        assert plugin.cleanup_count == 0

    def test_validate_dependencies_all_installed(self):
        """Test dependency validation when all are installed."""
        # Create plugin with common built-in dependency
        class TestPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test",
                    display_name="Test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type=PluginType.EXTENSION,
                    dependencies=["sys"]  # Built-in, always available
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

        plugin = TestPlugin()
        all_installed, missing = plugin.validate_dependencies()

        assert all_installed is True
        assert missing == []

    def test_validate_dependencies_missing(self):
        """Test dependency validation with missing packages."""
        # Create plugin with non-existent dependency
        class TestPlugin(BasePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="test",
                    display_name="Test",
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

        plugin = TestPlugin()
        all_installed, missing = plugin.validate_dependencies()

        assert all_installed is False
        assert "nonexistent_package_12345" in missing

    def test_plugin_repr(self):
        """Test plugin string representation."""
        plugin = MockPlugin()
        plugin.enable()

        repr_str = repr(plugin)

        assert "Mock Plugin" in repr_str
        assert "1.0.0" in repr_str
        assert "enabled" in repr_str


class MockBackend(BaseBackend):
    """Mock backend for testing."""

    @property
    def name(self):
        return "Mock"

    @property
    def supported_models(self):
        return ["mock-v1"]

    def generate_image(self, request):
        pass

    def health_check(self):
        return True


class MockBackendPlugin(BackendPlugin):
    """Mock backend plugin for testing."""

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="mockbackend",
            display_name="Mock Backend",
            version="1.0.0",
            author="Test",
            description="Mock backend plugin",
            plugin_type=PluginType.BACKEND,
            requires_api_key=True
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def get_backend_class(self) -> Type[BaseBackend]:
        return MockBackend


class TestBackendPlugin:
    """Tests for BackendPlugin class."""

    def test_backend_plugin_creation(self):
        """Test creating a backend plugin."""
        plugin = MockBackendPlugin()

        assert plugin.metadata.plugin_type == PluginType.BACKEND
        assert plugin.metadata.name == "mockbackend"
        assert plugin.metadata.requires_api_key is True

    def test_get_backend_class(self):
        """Test getting backend class from plugin."""
        plugin = MockBackendPlugin()

        backend_class = plugin.get_backend_class()

        assert backend_class == MockBackend
        assert issubclass(backend_class, BaseBackend)

    def test_wrong_plugin_type_raises_error(self):
        """Test that wrong plugin type raises error."""

        class InvalidBackendPlugin(BackendPlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="invalid",
                    display_name="Invalid",
                    version="1.0.0",
                    author="Test",
                    description="Invalid plugin",
                    plugin_type=PluginType.EXTENSION,  # Wrong type!
                    requires_api_key=False
                )

            def initialize(self):
                return True

            def cleanup(self):
                pass

            def get_backend_class(self):
                return MockBackend

        plugin = InvalidBackendPlugin()

        with pytest.raises(ValueError, match="BACKEND"):
            _ = plugin.metadata
