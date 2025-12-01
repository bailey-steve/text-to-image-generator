# Quick Start Guide

Get your text-to-image generator running in 5 minutes!

## Prerequisites

Before you begin, you'll need:
1. A HuggingFace account (free)
2. A HuggingFace API token

### Get Your HuggingFace API Token

1. Go to https://huggingface.co/ and create a free account (if you don't have one)
2. Navigate to https://huggingface.co/settings/tokens
3. Click "New token"
4. Give it a name (e.g., "text-to-image-app")
5. Select "Read" permissions
6. Click "Generate token"
7. Copy the token (it starts with `hf_`)

## Setup Steps

### 1. Install Python venv (if needed)

```bash
# Check if venv is available
python3 -m venv --help

# If not available, install it (Ubuntu/Debian)
sudo apt install python3-venv
```

### 2. Run the automated setup script

```bash
cd /home/sxbailey/CLionProjects/images
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create a `.env` file
- Run tests to verify everything works

### 3. Add your API token

Edit the `.env` file and add your HuggingFace token:

```bash
nano .env
```

Change this line:
```
HUGGINGFACE_TOKEN=hf_your_token_here
```

To:
```
HUGGINGFACE_TOKEN=hf_ActualTokenYouCopied
```

Save and exit (Ctrl+X, then Y, then Enter)

### 4. Run the application

```bash
source venv/bin/activate
python -m app.main
```

### 5. Open your browser

Navigate to: http://localhost:7860

## Your First Image

1. In the prompt box, type: `A serene landscape with mountains and a lake at sunset, digital art`
2. Click "🎨 Generate Image"
3. Wait 5-15 seconds
4. Your image will appear!

## Tips for Better Results

### Good Prompts
- Be descriptive and specific
- Mention the art style (e.g., "digital art", "photorealistic", "oil painting")
- Include details about lighting, mood, composition

Examples:
- "A futuristic cyberpunk city at night with neon lights, raining, cinematic"
- "A cute corgi puppy playing in a field of flowers, sunny day, digital art"
- "An astronaut floating in space with Earth in background, photorealistic"

### Negative Prompts
Help avoid unwanted elements:
- `blurry, low quality, distorted`
- `text, watermark, signature`
- `cartoon, anime` (if you want photorealistic)

### Parameter Tips
- **Guidance Scale (7.5 default)**:
  - Lower (4-6): More creative, less faithful to prompt
  - Higher (8-12): Closer to prompt, may be less creative
- **Inference Steps (50 default)**:
  - Fewer steps (20-30): Faster, lower quality
  - More steps (50-100): Slower, better quality

## Troubleshooting

### "Invalid HuggingFace API token"
- Double-check you copied the entire token (starts with `hf_`)
- Make sure there are no extra spaces in the `.env` file
- Verify the token is valid at https://huggingface.co/settings/tokens

### "Rate limit exceeded"
- Free tier has limits (typically hundreds of requests per hour)
- Wait a few minutes and try again
- Consider HuggingFace PRO for unlimited usage ($9/month)

### Application won't start
- Make sure virtual environment is activated: `source venv/bin/activate`
- Check all dependencies installed: `pip install -r requirements.txt`
- Look at the error message for specific issues

### Generation is slow
- Cloud API typically takes 5-15 seconds
- This is normal for the free tier
- Future versions will support local generation (slower but offline)

## What's Next?

Now that you have Stage 1 working, you can:
1. Experiment with different prompts and parameters
2. Try the example prompts in the UI
3. Explore the code in `src/` and `app/`
4. Wait for Stage 2 which adds:
   - Multiple backend options (Replicate)
   - Fallback logic when one backend is down
   - Backend health monitoring

## Need Help?

- Check the full README.md for detailed documentation
- Review the code comments in `src/` and `app/`
- Check the plan file at `~/.claude/plans/sleepy-jingling-bubble.md`

## Running Tests

Verify everything is working:

```bash
source venv/bin/activate
pytest
```

You should see all tests pass with >80% coverage.

---

Happy generating! 🎨
