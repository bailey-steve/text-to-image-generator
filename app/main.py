"""Main Gradio application for text-to-image generation with multi-backend support."""

import logging
import io
from typing import Optional, Tuple
from PIL import Image
import gradio as gr

from app.config import settings
from src.core.models import GenerationRequest
from src.core.backend_factory import BackendFactory
from src.core.image_generator import ImageGenerator

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global generator instance
generator: Optional[ImageGenerator] = None


def create_generator() -> ImageGenerator:
    """Create and initialize the image generator with backends.

    Returns:
        Initialized ImageGenerator instance with primary and fallback backends

    Raises:
        ValueError: If required configuration is missing
    """
    try:
        settings.validate_required_keys()

        # Create primary backend
        if settings.default_backend == "huggingface":
            primary = BackendFactory.create_backend(
                "huggingface",
                settings.huggingface_token
            )
        elif settings.default_backend == "replicate":
            primary = BackendFactory.create_backend(
                "replicate",
                settings.replicate_token or ""
            )
        else:
            primary = BackendFactory.create_backend(
                "huggingface",
                settings.huggingface_token
            )

        # Create fallback backends if enabled
        fallbacks = []
        if settings.enable_fallback and settings.fallback_backend:
            try:
                if settings.fallback_backend == "replicate" and settings.replicate_token:
                    fallbacks.append(
                        BackendFactory.create_backend(
                            "replicate",
                            settings.replicate_token
                        )
                    )
                elif settings.fallback_backend == "huggingface" and settings.huggingface_token:
                    fallbacks.append(
                        BackendFactory.create_backend(
                            "huggingface",
                            settings.huggingface_token
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to create fallback backend: {e}")

        gen = ImageGenerator(primary, fallbacks)
        logger.info(f"Initialized generator: {gen.get_backend_names()}")
        return gen

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise


def get_health_status() -> str:
    """Get health status of all backends.

    Returns:
        Formatted health status string
    """
    if generator is None:
        return "❌ Generator not initialized"

    try:
        health = generator.health_check_all()
        status_lines = []

        for backend_name, is_healthy in health.items():
            icon = "✅" if is_healthy else "❌"
            status_lines.append(f"{icon} {backend_name}")

        return "\n".join(status_lines)
    except Exception as e:
        return f"❌ Error checking health: {e}"


def generate_image(
    prompt: str,
    backend_choice: str = "auto",
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_steps: int = 4,
    width: int = 512,
    height: int = 512
) -> Tuple[Optional[Image.Image], str]:
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of the desired image
        backend_choice: Which backend to use ("auto", "huggingface", "replicate")
        negative_prompt: What to avoid in the image
        guidance_scale: How closely to follow the prompt (1.0-20.0)
        num_steps: Number of denoising steps (1-16)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple of (PIL Image object or None, status message)
    """
    if not prompt or prompt.strip() == "":
        return None, "Error: Please enter a prompt"

    if generator is None:
        return None, "❌ Error: Generator not initialized. Check your API tokens."

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

        # Determine whether to use fallback
        use_fallback = (backend_choice == "auto")

        # If user selected specific backend, temporarily switch primary
        if backend_choice != "auto":
            # Generate with selected backend only
            original_primary = generator.primary_backend

            try:
                # Try to use selected backend
                if backend_choice == "huggingface" and settings.huggingface_token:
                    temp_backend = BackendFactory.create_backend(
                        "huggingface",
                        settings.huggingface_token
                    )
                    generator.primary_backend = temp_backend
                    use_fallback = False
                elif backend_choice == "replicate" and settings.replicate_token:
                    temp_backend = BackendFactory.create_backend(
                        "replicate",
                        settings.replicate_token
                    )
                    generator.primary_backend = temp_backend
                    use_fallback = False

                result = generator.generate_image(request, use_fallback=use_fallback)

            finally:
                # Restore original primary
                generator.primary_backend = original_primary
        else:
            # Use auto mode with fallback
            result = generator.generate_image(request, use_fallback=True)

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


# Initialize generator
try:
    generator = create_generator()
except Exception as e:
    logger.critical(f"Failed to initialize generator: {e}")
    generator = None


def create_ui():
    """Create the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Text-to-Image Generator") as demo:
        gr.Markdown(
            """
            # 🎨 Text-to-Image Generator (Multi-Backend)

            Generate images from text descriptions using AI.
            **Stage 2**: Multi-backend support with automatic fallback!
            """
        )

        if generator is None:
            gr.Markdown(
                """
                ## ⚠️ Configuration Error

                The application could not initialize. Please check:
                1. Your `.env` file exists and contains valid API tokens
                2. For HuggingFace: Get your token from https://huggingface.co/settings/tokens
                3. For Replicate: Get your token from https://replicate.com/account/api-tokens
                4. Copy `.env.example` to `.env` and add your tokens
                """
            )
            return demo

        # Health status display
        with gr.Row():
            health_display = gr.Textbox(
                label="Backend Health Status",
                value=get_health_status(),
                interactive=False,
                lines=3
            )
            refresh_health = gr.Button("🔄 Refresh Health")

        with gr.Row():
            with gr.Column(scale=1):
                # Backend selection
                backend_selector = gr.Radio(
                    choices=["auto", "huggingface", "replicate"],
                    value="auto",
                    label="Backend Selection",
                    info="Auto uses primary with fallback, or choose specific backend"
                )

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
                        info="FLUX models: max 16 steps (4 recommended)"
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
                    lines=8,
                    interactive=False
                )

        # Examples
        gr.Examples(
            examples=[
                ["A futuristic city with flying cars at night, cyberpunk style", "auto", "blurry, low quality"],
                ["A cute cat wearing a wizard hat, digital art", "auto", ""],
                ["A serene Japanese garden with cherry blossoms", "auto", "people, animals"],
                ["An astronaut riding a horse on Mars, photorealistic", "auto", "cartoon, anime"],
            ],
            inputs=[prompt_input, backend_selector, negative_prompt_input],
        )

        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                backend_selector,
                negative_prompt_input,
                guidance_scale_slider,
                num_steps_slider,
                width_slider,
                height_slider
            ],
            outputs=[output_image, output_info]
        )

        refresh_health.click(
            fn=get_health_status,
            outputs=[health_display]
        )

        # Footer
        gr.Markdown(
            """
            ---
            💡 **Stage 2 Features:**
            - 🔄 **Multi-backend support**: HuggingFace + Replicate
            - 🛡️ **Automatic fallback**: If primary fails, tries fallback
            - 💚 **Health monitoring**: Check backend status in real-time
            - ⚡ **Retry logic**: Automatic retries for transient failures

            **Tips:**
            - **Auto mode**: Uses configured primary backend with automatic fallback
            - **Specific backend**: Choose HuggingFace or Replicate to use only that service
            - Use descriptive prompts for better results
            - Negative prompts help avoid unwanted elements
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
