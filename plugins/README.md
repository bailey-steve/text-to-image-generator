# Plugins Directory

This directory contains external plugins for the text-to-image generator.

## Plugin Structure

Each plugin should be a Python package (directory) with the following structure:

```
plugins/
└── my_plugin/
    ├── __init__.py          # Package initialization
    ├── __plugin__.py        # Plugin definition (required)
    └── backend.py           # Your backend implementation
```

## Creating a Backend Plugin

1. **Create Plugin Directory**
   ```bash
   mkdir plugins/my_backend
   ```

2. **Create `__init__.py`**
   ```python
   """My backend plugin package."""
   __version__ = "1.0.0"
   ```

3. **Create Backend Class** (`backend.py`)
   ```python
   from src.core.base_backend import BaseBackend
   from src.core.models import GenerationRequest, GeneratedImage

   class MyBackend(BaseBackend):
       @property
       def name(self) -> str:
           return "MyBackend"

       @property
       def supported_models(self) -> list[str]:
           return ["my-model-v1"]

       def generate_image(self, request: GenerationRequest) -> GeneratedImage:
           # Your image generation logic here
           pass

       def health_check(self) -> bool:
           return True
   ```

4. **Create Plugin Definition** (`__plugin__.py`)
   ```python
   from src.core.plugin import BackendPlugin, PluginMetadata, PluginType
   from src.core.base_backend import BaseBackend
   from typing import Type

   class Plugin(BackendPlugin):
       def _get_metadata(self) -> PluginMetadata:
           return PluginMetadata(
               name="my_backend",  # Lowercase, no spaces
               display_name="My Backend",
               version="1.0.0",
               author="Your Name",
               description="My custom backend",
               plugin_type=PluginType.BACKEND,
               dependencies=["requests"],  # Python packages required
               requires_api_key=True  # Set to False if no API key needed
           )

       def initialize(self) -> bool:
           # Initialize your plugin here
           return True

       def cleanup(self) -> None:
           # Clean up resources
           pass

       def get_backend_class(self) -> Type[BaseBackend]:
           from plugins.my_backend.backend import MyBackend
           return MyBackend
   ```

5. **Use Your Plugin**
   ```python
   from src.core.backend_factory import BackendFactory

   # The plugin is automatically discovered
   backend = BackendFactory.create_backend("my_backend", api_key="...")
   ```

## Example Plugin

See `dummy_backend/` for a complete example plugin that generates colored rectangles.

## Plugin Requirements

- Plugin name must be lowercase with no spaces
- Must define a `Plugin` class in `__plugin__.py`
- Plugin class must inherit from `BackendPlugin`
- Backend class must inherit from `BaseBackend`
- All dependencies must be installed for the plugin to load

## Plugin Discovery

Plugins are automatically discovered when the application starts. The plugin system:
1. Scans the `plugins/` directory
2. Looks for directories containing `__plugin__.py`
3. Loads the `Plugin` class
4. Validates dependencies
5. Registers the plugin with the PluginManager

## Built-in vs External Plugins

**Built-in Plugins** (Cannot be removed):
- `huggingface` - HuggingFace Inference API
- `replicate` - Replicate API
- `local` - Local Diffusers models

**External Plugins** (In `plugins/` directory):
- Can be added/removed without modifying code
- Automatically discovered on startup
- Can be enabled/disabled dynamically

## Troubleshooting

### Plugin Not Discovered
- Ensure `__plugin__.py` exists in plugin directory
- Check that `Plugin` class is defined in `__plugin__.py`
- Verify plugin name is lowercase with no spaces

### Plugin Fails to Load
- Check that all dependencies are installed
- Review logs for error messages
- Ensure backend class inherits from `BaseBackend`

### Import Errors
- Use absolute imports from your plugin package
- Make sure `__init__.py` exists
- Check that all required modules are present
