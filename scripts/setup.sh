#!/bin/bash
# Setup script for text-to-image application

set -e  # Exit on error

echo "========================================="
echo "Text-to-Image Generator Setup"
echo "========================================="
echo

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check for venv module
echo
echo "Checking for venv module..."
if ! python3 -m venv --help &> /dev/null; then
    echo "Warning: python3-venv is not installed"
    echo "On Ubuntu/Debian, run: sudo apt install python3-venv"
    echo "Please install it and run this script again."
    exit 1
fi

# Create virtual environment
echo
echo "Creating virtual environment..."
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    echo "Creating new virtual environment..."
    rm -rf venv  # Remove any incomplete venv
    python3 -m venv venv
    if [ ! -f "venv/bin/activate" ]; then
        echo "Error: Failed to create virtual environment"
        echo "Please make sure python3-venv is installed:"
        echo "  sudo apt install python3.13-venv"
        exit 1
    fi
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
echo
echo "Checking for .env file..."
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo
    echo "⚠️  IMPORTANT: Edit .env and add your HuggingFace token!"
    echo "   Get your token from: https://huggingface.co/settings/tokens"
else
    echo ".env file already exists"
fi

# Run tests
echo
echo "Running tests..."
pytest

echo
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo
echo "Next steps:"
echo "1. Edit .env and add your HuggingFace token"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the application: python -m app.main"
echo "4. Open http://localhost:7860 in your browser"
echo
