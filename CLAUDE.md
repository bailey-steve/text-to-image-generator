# Text-to-Image Generator Project

## Project Overview

A production-ready text-to-image generation system with multi-backend support (HuggingFace, Replicate, Local), built with Python and Gradio. Features include prompt enhancement, batch generation, history tracking, health monitoring, and a plugin system for extensibility.

**Current Stage**: Stage 6 Complete + Prompt Enhancement (Improvement #2)

## Architecture

### Directory Structure
```
/home/sxbailey/CLionProjects/images/
├── app/                    # Gradio UI application
│   ├── main.py            # Main Gradio interface (1247 lines)
│   └── config.py          # Application configuration
├── src/                   # Core application code
│   ├── backends/          # Image generation backends
│   │   ├── huggingface.py
│   │   ├── replicate.py
│   │   └── local.py
│   ├── core/              # Core functionality
│   │   ├── models.py      # Data models (GenerationRequest, GeneratedImage)
│   │   ├── image_generator.py
│   │   ├── backend_factory.py
│   │   ├── plugin.py      # Plugin system base
│   │   ├── plugin_manager.py
│   │   └── builtin_plugins.py
│   └── utils/             # Utility modules
│       ├── image_utils.py
│       ├── history_manager.py
│       ├── prompt_enhancer.py  # Prompt enhancement system
│       ├── rate_limiter.py     # Production rate limiting
│       └── health.py           # Production health monitoring
├── tests/                 # Test suite (259 tests, 94.44% coverage)
│   └── unit/
├── plugins/               # External plugins directory
├── .env                   # Environment variables (API tokens)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Production Docker image
├── docker-compose.yml    # Deployment orchestration
└── DEPLOYMENT.md         # Production deployment guide
```

### Key Components

1. **Backends**: Pluggable image generation backends (HuggingFace, Replicate, Local CPU)
2. **Image Generator**: Orchestrates backends with automatic fallback
3. **Gradio UI**: Web interface with prompt enhancement, batch generation, history
4. **Plugin System**: Extensible architecture for custom backends
5. **Prompt Enhancer**: 10 styles, 4 quality levels, 8 templates
6. **Production Features**: Rate limiting, health monitoring, Docker deployment

## Code Style & Conventions

### Python Style
- **Python Version**: 3.11+
- **Style Guide**: PEP 8
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Type Hints**: Required for all function signatures
- **Docstrings**: Required for all public functions and classes (Google style)
- **Line Length**: 100 characters maximum
- **Imports**: Group standard library, third-party, local (separated by blank lines)

### Key Patterns
- **Return Types**: Backend methods return `GeneratedImage` objects, not raw bytes
- **Error Handling**: Use specific exceptions (ValueError, ConnectionError, RuntimeError)
- **Logging**: Use module-level loggers with appropriate levels
- **Singletons**: Use global `get_*()` functions for shared instances (prompt_enhancer, rate_limiter, health_checker)
- **Configuration**: Use Pydantic BaseSettings in `app/config.py`

### Example Code Pattern
```python
def generate_image(prompt: str, style: Optional[PromptStyle] = None) -> GeneratedImage:
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of desired image
        style: Optional style preset to apply

    Returns:
        GeneratedImage object with image data and metadata

    Raises:
        ValueError: If prompt is empty
        ConnectionError: If backend is unavailable
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty")

    # Implementation here
    logger.info(f"Generated image with prompt: {prompt[:50]}...")
    return result
```

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API tokens
```

### Running the Application
```bash
# Run Gradio UI (default: http://localhost:7861)
python -m app.main

# Run with specific port
GRADIO_SERVER_PORT=7862 python -m app.main

# Run with environment variables
PYTHONPATH=/home/sxbailey/CLionProjects/images python -m app.main
```

### Testing
```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_prompt_enhancer.py -v

# Run tests with coverage threshold (80%)
python -m pytest tests/ --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80

# Quick test run (no coverage)
python -m pytest tests/unit/test_prompt_enhancer.py -v
```

### Docker Deployment
```bash
# Build Docker image
docker build -t text-to-image-generator .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development Utilities
```bash
# Kill processes on port
lsof -ti:7861 | xargs kill -9

# Check code style
flake8 src/ app/ tests/

# Format code
black src/ app/ tests/
```

## Development Workflow

### Stage-Based Development
The project follows a staged development approach:

- **Stage 1**: Basic single-backend implementation (HuggingFace)
- **Stage 2**: Multi-backend support with automatic fallback
- **Stage 3**: Enhanced UI features (batch generation, history, downloads)
- **Stage 4**: Advanced features (aspect ratios, format selection)
- **Stage 5**: Plugin system for extensibility
- **Stage 6**: Production features (Docker, rate limiting, health monitoring)
- **Improvements**: Prompt enhancement system (#2)

### Git Workflow
```bash
# Commit pattern
git add .
git commit -m "feat: Brief description

Detailed explanation:
- Feature 1
- Feature 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to main
git push origin main
```

### Testing Before Commit
Always run tests before committing:
1. Run full test suite: `python -m pytest tests/ -v`
2. Verify coverage: Must be ≥ 80%
3. Check all tests pass (currently: 259/259)
4. Commit changes with descriptive message

## Key Features & Configuration

### Prompt Enhancement System
- **10 Style Presets**: Photorealistic, Artistic, Anime, Digital Art, Oil Painting, Watercolor, Sketch, Cyberpunk, Fantasy, Minimalist
- **4 Quality Levels**: Standard, High Quality, Masterpiece, Professional
- **8 Templates**: Portrait, Landscape, Character, Architecture, Product, Animal, Abstract, Food
- **Utilities**: Enhancement, negative prompt generation, improvement suggestions
- **Access**: `from src.utils.prompt_enhancer import get_prompt_enhancer, PromptStyle, PromptQuality`

### Backend Configuration
Edit `.env` file:
```bash
# Primary backend (huggingface, replicate, local)
DEFAULT_BACKEND=replicate

# API tokens
HUGGINGFACE_TOKEN=hf_xxxxx
REPLICATE_TOKEN=r8_xxxxx

# Models
HUGGINGFACE_MODEL=black-forest-labs/FLUX.1-schnell
REPLICATE_MODEL=black-forest-labs/flux-schnell

# Fallback settings
ENABLE_FALLBACK=true
FALLBACK_BACKEND=huggingface
```

### Production Settings
```python
# In app/config.py
enable_rate_limiting: bool = True
rate_limit_requests: int = 100
rate_limit_window: int = 60
enable_health_checks: bool = True
production_mode: bool = False
```

## Important Data Models

### GenerationRequest
```python
from src.core.models import GenerationRequest

request = GenerationRequest(
    prompt="A cat sitting on a windowsill",
    negative_prompt="blurry, low quality",
    guidance_scale=7.5,
    num_inference_steps=4,
    width=512,
    height=512,
    seed=None  # Optional, for reproducibility
)
```

### GeneratedImage
```python
from src.core.models import GeneratedImage

# Returned by all backend generate() methods
result = GeneratedImage(
    image_data=bytes,  # PNG image bytes
    prompt="original prompt",
    backend="Replicate",
    metadata={
        "model": "flux-schnell",
        "guidance_scale": 7.5,
        # ... other metadata
    }
)
```

## Plugin Development

### Creating a Custom Backend Plugin
```python
# plugins/my_backend/plugin.py
from src.core.plugin import BackendPlugin, PluginMetadata
from src.core.base_backend import BaseBackend

class MyBackendPlugin(BackendPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MyBackend",
            version="1.0.0",
            description="My custom image generation backend"
        )

    def create_backend(self, **kwargs) -> BaseBackend:
        # Return your backend implementation
        pass
```

Place in `plugins/my_backend/` directory and it will be auto-discovered.

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -ti:7861 | xargs kill -9
   ```

2. **ModuleNotFoundError**
   ```bash
   # Ensure PYTHONPATH is set
   PYTHONPATH=/home/sxbailey/CLionProjects/images python -m app.main
   ```

3. **API Token Issues**
   - Check `.env` file exists and contains valid tokens
   - Verify tokens at https://huggingface.co/settings/tokens or https://replicate.com/account/api-tokens

4. **Tests Failing**
   - Check venv is activated: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check coverage threshold: Must be ≥ 80%

## Project Metrics

- **Total Tests**: 259 (all passing)
- **Code Coverage**: 94.44%
- **Lines of Code**: ~6,000+
- **Python Version**: 3.11+
- **Dependencies**: 15+ packages (see requirements.txt)
- **Supported Backends**: 3 (HuggingFace, Replicate, Local)
- **Available Templates**: 8
- **Style Presets**: 10
- **Quality Levels**: 4

## Next Steps / Potential Improvements

Based on the improvement suggestions provided earlier, potential next features include:
1. ✅ **Prompt Enhancement** (Completed - Improvement #2)
2. Image-to-Image support
3. LoRA model integration
4. Advanced scheduling algorithms
5. API endpoints for programmatic access
6. Image editing capabilities
7. Custom model training interface
8. Multi-language support

## Security & Best Practices

- Never commit `.env` file (contains API tokens)
- Use environment variables for secrets
- Validate all user inputs
- Run security audits with `bandit` or similar tools
- Keep dependencies updated
- Use rate limiting in production
- Monitor health metrics
- Review DEPLOYMENT.md for production deployment guidelines

---

**Last Updated**: 2025-12-03
**Project Status**: Production Ready (Stage 6 Complete)
**Maintainer**: bailey-steve