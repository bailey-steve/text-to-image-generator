"""Main Gradio application for text-to-image generation."""

import logging
import io
from PIL import Image
import gradio as gr

from app.config import settings
from src.core.models import GenerationRequest
from src.backends.huggingface import HuggingFaceBackend

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_backend():
    """Create and initialize the backend.

    Returns:
        Initialized backend instance

    Raises:
        ValueError: If required configuration is missing
    """
    try:
        settings.validate_required_keys()
        backend = HuggingFaceBackend(api_key=settings.huggingface_token)
        logger.info(f"Initialized backend: {backend.name}")
        return backend
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_steps: int = 4,
    width: int = 512,
    height: int = 512
):
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of the desired image
        negative_prompt: What to avoid in the image
        guidance_scale: How closely to follow the prompt (1.0-20.0)
        num_steps: Number of denoising steps (1-150)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        PIL Image object or error message
    """
    if not prompt or prompt.strip() == "":
        return None, "Error: Please enter a prompt"

    try:
        logger.info(f"Generating image with prompt: {prompt[:50]}...")

        # Create generation request
        request = GenerationRequest(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip() if negative_prompt else None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            width=width,
            height=height
        )

        # Generate image
        result = backend.generate_image(request)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(result.image_data))

        # Create info message
        info_message = (
            f"✅ Image generated successfully!\n"
            f"Backend: {result.backend}\n"
            f"Model: {result.metadata.get('model', 'N/A')}\n"
            f"Size: {width}x{height}\n"
            f"Steps: {num_steps}\n"
            f"Guidance Scale: {guidance_scale}"
        )

        return image, info_message

    except ValueError as e:
        error_msg = f"❌ Invalid parameters: {e}"
        logger.error(error_msg)
        return None, error_msg

    except ConnectionError as e:
        error_msg = f"❌ Connection error: {e}"
        logger.error(error_msg)
        return None, error_msg

    except RuntimeError as e:
        error_msg = f"❌ Generation failed: {e}"
        logger.error(error_msg)
        return None, error_msg

    except Exception as e:
        error_msg = f"❌ Unexpected error: {e}"
        logger.exception(error_msg)
        return None, error_msg


# Initialize backend
try:
    backend = create_backend()
except Exception as e:
    logger.critical(f"Failed to initialize backend: {e}")
    backend = None


def create_ui():
    """Create the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Text-to-Image Generator") as demo:
        gr.Markdown(
            """
            # 🎨 Text-to-Image Generator

            Generate images from text descriptions using AI. Powered by HuggingFace Stable Diffusion.
            """
        )

        if backend is None:
            gr.Markdown(
                """
                ## ⚠️ Configuration Error

                The application could not initialize. Please check:
                1. Your `.env` file exists and contains a valid `HUGGINGFACE_TOKEN`
                2. Get your token from: https://huggingface.co/settings/tokens
                3. Copy `.env.example` to `.env` and add your token
                """
            )
            return demo

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value="A serene landscape with mountains and a lake at sunset, digital art"
                )

                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="What to avoid in the image...",
                    lines=2,
                    value="blurry, low quality, distorted"
                )

                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt"
                    )

                    num_steps_slider = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Inference Steps",
                        info="FLUX.1-schnell: max 16 steps (4 recommended)"
                    )

                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Width"
                        )

                        height_slider = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Height"
                        )

                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    show_label=True
                )

                output_info = gr.Textbox(
                    label="Generation Info",
                    lines=6,
                    interactive=False
                )

        # Examples
        gr.Examples(
            examples=[
                ["A futuristic city with flying cars at night, cyberpunk style", "blurry, low quality"],
                ["A cute cat wearing a wizard hat, digital art", ""],
                ["A serene Japanese garden with cherry blossoms", "people, animals"],
                ["An astronaut riding a horse on Mars, photorealistic", "cartoon, anime"],
            ],
            inputs=[prompt_input, negative_prompt_input],
        )

        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                negative_prompt_input,
                guidance_scale_slider,
                num_steps_slider,
                width_slider,
                height_slider
            ],
            outputs=[output_image, output_info]
        )

        # Footer
        gr.Markdown(
            """
            ---
            💡 **Tips:**
            - Use descriptive prompts for better results
            - Negative prompts help avoid unwanted elements
            - Higher guidance scale = closer to prompt (but may reduce creativity)
            - More steps = better quality (but slower generation)
            """
        )

    return demo


if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()

    logger.info("Launching Gradio application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
