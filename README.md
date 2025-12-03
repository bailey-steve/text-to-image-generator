# Text-to-Image Generator

A Python-based text-to-image generation application with a web interface, built using Gradio and HuggingFace's Stable Diffusion models.

## Features

- 🎨 Generate images from text prompts using AI
- 🌐 Web-based user interface (Gradio)
- ☁️ Multiple backends: HuggingFace, Replicate, Local (offline)
- 💾 Image download with metadata embedding (PNG/JPEG/WebP)
- 📸 Image gallery and history management (up to 50 images)
- 🎲 Batch generation (2-9 variations with different seeds)
- ⚙️ Adjustable generation parameters (guidance scale, steps, image size)
- 📐 Aspect ratio presets (Square, Portrait, Landscape, Widescreen)
- 🔄 Support for negative prompts
- 🔁 Automatic fallback between backends
- 🏠 Local CPU-optimized models for offline use (SD-Turbo, SDXL-Turbo)
- 📊 Extensible architecture for multiple backends
- 🔌 Plugin system for custom backend development
- 🛠️ Example dummy backend plugin included
- 🐳 Docker and Docker Compose support
- 💪 Production-ready with health checks and monitoring
- 🛡️ Rate limiting for API protection
- 📈 System metrics and performance monitoring
- ✨ Prompt enhancement with templates and style presets
- 💬 Intelligent prompt suggestions and improvements
- 🎯 Negative prompt generation

## Prerequisites

- Python 3.8 or higher
- HuggingFace account and API token (free)
- Internet connection for cloud-based generation

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /home/sxbailey/CLionProjects/images
```

### 2. Install Python venv package (if not already installed)

```bash
# On Ubuntu/Debian
sudo apt install python3-venv

# On other systems, venv usually comes with Python
```

### 3. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your HuggingFace token
# Get your token from: https://huggingface.co/settings/tokens
nano .env  # or use your preferred editor
```

Your `.env` file should look like:
```
HUGGINGFACE_TOKEN=hf_your_actual_token_here
DEFAULT_BACKEND=huggingface
LOG_LEVEL=INFO
```

## Usage

### Running the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the application
python -m app.main
```

The application will start on http://localhost:7860

### Using the Web Interface

1. Open your browser to http://localhost:7860
2. Enter a text prompt describing the image you want
3. (Optional) Add a negative prompt to avoid unwanted elements
4. (Optional) Adjust advanced settings:
   - **Guidance Scale**: How closely to follow your prompt (1.0-20.0)
   - **Inference Steps**: More steps = better quality but slower (1-150)
   - **Width/Height**: Output image dimensions (256-1024)
5. Click "Generate Image"
6. Wait for the image to appear (typically 5-15 seconds)

### Example Prompts

- "A serene landscape with mountains and a lake at sunset, digital art"
- "A futuristic city with flying cars at night, cyberpunk style"
- "A cute cat wearing a wizard hat, digital art"
- "An astronaut riding a horse on Mars, photorealistic"

### Using Local Backend (Offline Mode)

The local backend allows you to generate images offline using CPU-optimized models:

**Setup:**

```bash
# Install Stage 4 dependencies (torch, diffusers, etc.)
pip install -r requirements.txt

# Configure for local mode (optional - can also select in UI)
# Edit .env file:
DEFAULT_BACKEND=local
LOCAL_MODEL=stabilityai/sd-turbo  # or stabilityai/sdxl-turbo
```

**Features:**
- **No API keys required** - Works completely offline
- **First-time setup**: Model downloads automatically (~1-2GB) and caches locally
- **Faster subsequent runs**: Models load from cache (~5-10 seconds)
- **CPU-optimized**: Uses SD-Turbo (512x512) or SDXL-Turbo (1024x1024)
- **1-4 inference steps**: Very fast generation compared to standard models

**Usage in UI:**
1. Select "local" from the Backend Selection dropdown
2. Generate images as normal - no internet required!
3. First generation will download the model (one-time only)

**Performance Notes:**
- First generation: 1-2 minutes (includes model download)
- Subsequent generations: 10-30 seconds on CPU
- GPU support: Automatic if CUDA is available

## Plugin System (Stage 5)

The application features an extensible plugin system that allows you to add custom image generation backends without modifying the core code.

### What are Plugins?

Plugins are Python packages that extend the application's functionality. Currently, the plugin system supports **backend plugins** that provide new image generation sources (APIs, models, services).

### Built-in Backends as Plugins

All existing backends are now plugin-based:
- **huggingface**: HuggingFace Inference API (requires API key)
- **replicate**: Replicate API (requires API key)
- **local**: Local Diffusers models (no API key needed)

### Creating a Custom Backend Plugin

**1. Create Plugin Directory**

```bash
mkdir -p plugins/my_backend
```

**2. Create Package Files**

```bash
# plugins/my_backend/__init__.py
"""My custom backend plugin."""
__version__ = "1.0.0"
```

**3. Implement Your Backend** (`plugins/my_backend/backend.py`)

```python
from src.core.base_backend import BaseBackend
from src.core.models import GenerationRequest, GeneratedImage
from datetime import datetime
import io

class MyBackend(BaseBackend):
    """Your custom backend implementation."""

    @property
    def name(self) -> str:
        return "MyBackend"

    @property
    def supported_models(self) -> list[str]:
        return ["my-model-v1"]

    def generate_image(self, request: GenerationRequest) -> GeneratedImage:
        # Your image generation logic here
        # ...

        return GeneratedImage(
            image_data=image_bytes,
            prompt=request.prompt,
            backend=self.name,
            timestamp=datetime.now(),
            metadata={"model": "my-model-v1"}
        )

    def health_check(self) -> bool:
        return True
```

**4. Create Plugin Definition** (`plugins/my_backend/__plugin__.py`)

```python
from src.core.plugin import BackendPlugin, PluginMetadata, PluginType
from src.core.base_backend import BaseBackend
from typing import Type

class Plugin(BackendPlugin):
    """Plugin definition for My Backend."""

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_backend",  # Lowercase, no spaces
            display_name="My Backend",
            version="1.0.0",
            author="Your Name",
            description="Custom backend for special image generation",
            plugin_type=PluginType.BACKEND,
            dependencies=["requests"],  # Required Python packages
            requires_api_key=True  # Set to False if no API key needed
        )

    def initialize(self) -> bool:
        # Initialize your plugin (optional)
        return True

    def cleanup(self) -> None:
        # Clean up resources (optional)
        pass

    def get_backend_class(self) -> Type[BaseBackend]:
        from plugins.my_backend.backend import MyBackend
        return MyBackend
```

**5. Use Your Plugin**

Your plugin is automatically discovered on startup:

```python
from src.core.backend_factory import BackendFactory

# Create backend from your plugin
backend = BackendFactory.create_backend("my_backend", api_key="...")

# Or use it in the UI - it will appear in the backend dropdown
```

### Example Plugin

The repository includes a **dummy_backend** plugin in `plugins/dummy_backend/` that generates simple colored rectangles. This is useful for:
- Understanding the plugin structure
- Testing without API keys or heavy models
- Fast development and debugging

To use the dummy backend:
```python
backend = BackendFactory.create_backend("dummy_backend")
```

### Plugin Requirements

- Plugin name must be **lowercase with no spaces** (e.g., `my_backend`)
- Must have `__plugin__.py` with a `Plugin` class
- Plugin class must inherit from `BackendPlugin`
- Backend class must inherit from `BaseBackend`
- All dependencies must be installed before the plugin loads

### Plugin Discovery

The plugin system automatically:
1. Scans the `plugins/` directory on startup
2. Looks for directories containing `__plugin__.py`
3. Validates the plugin structure and dependencies
4. Registers discovered plugins with the PluginManager
5. Makes plugins available through BackendFactory

### Plugin Development Tips

- **Start Simple**: Copy the dummy_backend plugin and modify it
- **Test Locally**: Ensure all dependencies are installed
- **Error Handling**: Add proper error handling in your backend
- **Metadata**: Provide clear descriptions and accurate dependency lists
- **Documentation**: Add a README in your plugin directory

For more details, see `plugins/README.md` and the example plugin in `plugins/dummy_backend/`.

## Prompt Enhancement

The application includes a powerful prompt enhancement system to help you create better images.

### Features

**🎨 Style Presets:**
- Photorealistic
- Digital Art
- Anime/Manga
- Oil Painting
- Watercolor
- Cyberpunk
- Fantasy
- And more!

**⭐ Quality Levels:**
- Standard
- High Quality
- Masterpiece
- Professional

**📝 Template Library:**
- Portrait templates
- Landscape templates
- Character design
- Architecture
- Product photography
- Food photography
- Abstract art
- And more!

### Using Prompt Enhancement

**Python API:**
```python
from src.utils.prompt_enhancer import PromptEnhancer, PromptStyle, PromptQuality

enhancer = PromptEnhancer()

# Basic enhancement
enhanced = enhancer.enhance_prompt("a cat sitting")
# Result: "a cat sitting, detailed, sharp focus"

# With style and quality
enhanced = enhancer.enhance_prompt(
    "a mountain landscape",
    style=PromptStyle.PHOTOREALISTIC,
    quality=PromptQuality.MASTERPIECE
)
# Result: "a mountain landscape, photorealistic, highly detailed, 8k uhd,
#          dslr, soft lighting, high quality, masterpiece, best quality,
#          highly detailed, award-winning"

# Generate negative prompts
negative = enhancer.generate_negative_prompt()
# Result: "blurry, low quality, distorted, deformed, bad anatomy, ..."

# Get suggestions for improvement
suggestions = enhancer.suggest_improvements("cat")
# Returns: issues, recommendations, and enhanced examples
```

**Using Templates:**
```python
from src.utils.prompt_enhancer import PromptLibrary

# Get a template
template = PromptLibrary.get_template("portrait")

# Format with your values
prompt = template.format(
    subject="a wise wizard",
    style="oil painting",
    quality="masterpiece",
    lighting="dramatic lighting"
)
# Result: "portrait of a wise wizard, oil painting, masterpiece,
#          detailed face, dramatic lighting"

# Search for templates
templates = PromptLibrary.search_templates("nature")
# Returns all templates related to nature

# Browse by category
landscape_templates = PromptLibrary.get_templates_by_category("nature")
```

### Available Templates

| Template | Category | Description |
|----------|----------|-------------|
| portrait | people | High-quality portrait generation |
| landscape | nature | Beautiful landscape scenes |
| character | fantasy | Character design and concept art |
| architecture | buildings | Architectural visualization |
| product | commercial | Product photography style |
| animal | nature | Animal and wildlife images |
| abstract | artistic | Abstract compositions |
| food | commercial | Food photography |

### Style Examples

**Photorealistic:**
```
"a beautiful sunset over mountains, photorealistic, highly detailed,
8k uhd, dslr, soft lighting, high quality"
```

**Digital Art:**
```
"a futuristic city, digital art, concept art, trending on artstation,
highly detailed"
```

**Anime:**
```
"a magical girl character, anime style, manga illustration, vibrant colors,
clean linework"
```

### Tips for Better Prompts

1. **Be Specific:** Instead of "a cat", try "a fluffy orange tabby cat"
2. **Add Details:** Include lighting, mood, composition details
3. **Use Quality Modifiers:** "high quality", "detailed", "masterpiece"
4. **Specify Style:** Photorealistic, digital art, oil painting, etc.
5. **Use Negative Prompts:** Specify what you don't want in the image

## Production Deployment (Stage 6)

The application is production-ready with Docker support, monitoring, and rate limiting.

### Quick Deploy with Docker

```bash
# 1. Create .env file
cp .env.example .env
# Edit .env and add your API tokens

# 2. Start with Docker Compose
docker-compose up -d

# 3. Access the application
open http://localhost:7860
```

### Features

**Docker Containerization:**
- Multi-stage Docker build for optimized images
- Non-root user for security
- Health checks built-in
- Automatic restart on failure

**Monitoring:**
- `/health` endpoint for load balancers
- System metrics (CPU, memory, disk usage)
- Request/error tracking
- Uptime monitoring

**Rate Limiting:**
- Configurable request limits per client
- Sliding window algorithm
- Automatic client tracking
- Protection against abuse

**Production Configuration:**
```bash
# In .env
PRODUCTION_MODE=true
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100  # requests per window
RATE_LIMIT_WINDOW=60     # seconds
ENABLE_HEALTH_CHECKS=true
ENABLE_METRICS=true
```

### Deployment Options

**Docker Compose (Recommended):**
```bash
docker-compose up -d
```

**Standalone Docker:**
```bash
docker build -t text-to-image-generator .
docker run -d -p 7860:7860 --env-file .env text-to-image-generator
```

**Kubernetes:**
See `DEPLOYMENT.md` for Kubernetes manifests and configuration.

### Monitoring & Health Checks

Check application health:
```bash
curl http://localhost:7860/health
```

Response:
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "details": {
    "uptime_seconds": 3600,
    "cpu_usage_percent": 15.2,
    "memory_usage_percent": 45.3,
    "request_count": 150,
    "error_rate": 0.0133
  }
}
```

### Complete Deployment Guide

For comprehensive deployment instructions including:
- Security best practices
- Scaling strategies
- Load balancer configuration
- Troubleshooting
- Production checklist

See **[DEPLOYMENT.md](DEPLOYMENT.md)**

## Testing

### Run all tests

```bash
pytest
```

### Run with coverage report

```bash
pytest --cov=src --cov=app --cov-report=html
```

### Run only unit tests (fast)

```bash
pytest tests/unit/
```

### Run specific test file

```bash
pytest tests/unit/test_models.py -v
```

## Project Structure

```
/home/sxbailey/CLionProjects/images/
├── app/                          # Application entry point
│   ├── main.py                   # Gradio UI
│   ├── config.py                 # Configuration management
│   └── ui/                       # UI components (future)
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── models.py             # Pydantic data models
│   │   └── base_backend.py       # Abstract backend interface
│   ├── backends/                 # Backend implementations
│   │   └── huggingface.py        # HuggingFace API backend
│   └── utils/                    # Utility functions
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── pytest.ini                    # Pytest configuration
└── README.md                     # This file
```

## Architecture

The application follows a modular architecture:

- **Strategy Pattern**: Backends implement a common interface (`BaseBackend`)
- **Pydantic Models**: Type-safe data validation for requests and responses
- **Configuration Management**: Environment-based settings using `pydantic-settings`
- **Gradio UI**: Modern web interface for easy interaction

## Roadmap

This is a 6-stage development plan:

### ✅ Completed Stages

**Stage 1: Foundation**
- Basic text-to-image with HuggingFace backend
- Gradio web interface
- Core architecture and tests

**Stage 2: Multi-Backend Support**
- Replicate backend integration
- Automatic fallback logic
- Backend health monitoring

**Stage 3: Advanced Features**
- Image download with metadata (PNG/JPEG/WebP)
- Image gallery and history management
- Batch generation (2-9 variations)
- Aspect ratio presets

**Stage 4: Local CPU-Optimized Models**
- Offline image generation using Diffusers
- CPU-optimized models (SD-Turbo, SDXL-Turbo)
- Model caching for faster loading
- No API keys required for local mode

**Stage 5: Plugin System**
- Extensible plugin architecture for custom backends
- Plugin discovery and management system
- Built-in plugin wrappers for existing backends
- Example dummy backend plugin for testing
- Comprehensive plugin development documentation

**Stage 6: Production Features** (Current)
- Docker containerization with multi-stage builds
- Docker Compose for easy deployment
- Health check endpoints and monitoring
- Rate limiting for API protection
- System metrics collection (CPU, memory, disk)
- Production configuration management
- Comprehensive deployment documentation

### 🎉 All Development Stages Complete!

## Troubleshooting

### "HuggingFace API token is required" error
- Make sure you've created a `.env` file from `.env.example`
- Add your HuggingFace token to the `.env` file
- Get a token from: https://huggingface.co/settings/tokens

### "Rate limit exceeded" error
- You've hit the free tier limit
- Wait a few minutes and try again
- Consider upgrading to HuggingFace PRO ($9/month) for higher limits

### Tests failing
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Some tests may be skipped if integration tests are disabled (this is normal)

### Slow generation
- Cloud API generation takes 5-15 seconds per image
- Reduce inference steps for faster (but lower quality) results
- Consider upgrading to HuggingFace PRO for faster processing

## Contributing

This project follows an incremental development approach. Each stage builds on the previous one with comprehensive tests.

## License

This project is for educational and personal use.

## Credits

- Built with [Gradio](https://gradio.app/)
- Powered by [HuggingFace](https://huggingface.co/) Stable Diffusion models
- Architecture inspired by the Strategy and Factory design patterns
