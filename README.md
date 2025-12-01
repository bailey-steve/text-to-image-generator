# Text-to-Image Generator

A Python-based text-to-image generation application with a web interface, built using Gradio and HuggingFace's Stable Diffusion models.

## Features

- 🎨 Generate images from text prompts using AI
- 🌐 Web-based user interface (Gradio)
- ⚙️ Adjustable generation parameters (guidance scale, steps, image size)
- 🔄 Support for negative prompts
- 📊 Extensible architecture for multiple backends

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

This is **Stage 1** of a 6-stage development plan:

### ✅ Stage 1: Foundation (Current)
- Basic text-to-image with HuggingFace backend
- Gradio web interface
- Core architecture and tests

### 🔄 Future Stages
- **Stage 2**: Multi-backend support (Replicate, fallback logic)
- **Stage 3**: Advanced parameters, image history, presets
- **Stage 4**: Local CPU-optimized models for offline use
- **Stage 5**: Plugin system for extensibility
- **Stage 6**: Production features (Docker, monitoring, rate limiting)

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
