"""Main Gradio application for text-to-image generation with multi-backend support."""

import logging
import io
from typing import Optional, Tuple
from PIL import Image
import gradio as gr

from app.config import settings
from src.core.models import GenerationRequest, GeneratedImage
from src.core.backend_factory import BackendFactory
from src.core.image_generator import ImageGenerator
from src.utils.image_utils import create_downloadable_image, ImageFormat
from src.utils.history_manager import ImageHistoryManager

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global generator instance
generator: Optional[ImageGenerator] = None

# Global storage for last generated image
last_generated_image: Optional[GeneratedImage] = None

# Global history manager
history_manager = ImageHistoryManager(max_history=50)


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
        elif settings.default_backend == "local":
            primary = BackendFactory.create_backend(
                "local",
                model=settings.local_model
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
                elif settings.fallback_backend == "local":
                    fallbacks.append(
                        BackendFactory.create_backend(
                            "local",
                            model=settings.local_model
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


def generate_batch_images(
    prompt: str,
    batch_size: int,
    backend_choice: str = "auto",
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_steps: int = 4,
    width: int = 512,
    height: int = 512,
    progress=gr.Progress()
) -> Tuple[list, str]:
    """Generate multiple images with the same prompt but different seeds.

    Args:
        prompt: Text description
        batch_size: Number of images to generate
        backend_choice: Backend to use
        negative_prompt: What to avoid
        guidance_scale: How closely to follow prompt
        num_steps: Number of inference steps
        width: Image width
        height: Image height
        progress: Gradio progress tracker

    Returns:
        Tuple of (list of PIL Images, status message)
    """
    if not prompt or prompt.strip() == "":
        return [], "Error: Please enter a prompt"

    if generator is None:
        return [], "❌ Error: Generator not initialized"

    images = []
    import random

    for i in range(batch_size):
        try:
            progress((i, batch_size), desc=f"Generating image {i+1}/{batch_size}...")

            # Generate random seed for variation
            seed = random.randint(0, 2**32 - 1)

            # Create generation request
            request = GenerationRequest(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() if negative_prompt else None,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                width=width,
                height=height,
                seed=seed
            )

            # Determine whether to use fallback
            use_fallback = (backend_choice == "auto")

            # Generate image
            if backend_choice != "auto":
                original_primary = generator.primary_backend
                try:
                    if backend_choice == "huggingface" and settings.huggingface_token:
                        temp_backend = BackendFactory.create_backend("huggingface", settings.huggingface_token)
                        generator.primary_backend = temp_backend
                        use_fallback = False
                    elif backend_choice == "replicate" and settings.replicate_token:
                        temp_backend = BackendFactory.create_backend("replicate", settings.replicate_token)
                        generator.primary_backend = temp_backend
                        use_fallback = False
                    elif backend_choice == "local":
                        temp_backend = BackendFactory.create_backend("local", model=settings.local_model)
                        generator.primary_backend = temp_backend
                        use_fallback = False

                    result = generator.generate_image(request, use_fallback=use_fallback)
                finally:
                    generator.primary_backend = original_primary
            else:
                result = generator.generate_image(request, use_fallback=True)

            # Add to history
            history_manager.add(result)

            # Convert to PIL Image
            image = Image.open(io.BytesIO(result.image_data))
            images.append(image)

            logger.info(f"Generated batch image {i+1}/{batch_size}")

        except Exception as e:
            logger.error(f"Failed to generate batch image {i+1}: {e}")
            # Continue with remaining images
            continue

    if not images:
        return [], "❌ All batch generations failed"

    info_message = (
        f"✅ Generated {len(images)}/{batch_size} images successfully!\n"
        f"Backend: {result.backend if 'result' in locals() else 'N/A'}\n"
        f"Check the History tab to view all images"
    )

    return images, info_message


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
                elif backend_choice == "local":
                    temp_backend = BackendFactory.create_backend(
                        "local",
                        model=settings.local_model
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

        # Store globally for download
        global last_generated_image
        last_generated_image = result

        # Add to history
        history_manager.add(result)

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


def get_history_gallery():
    """Get history formatted for gallery display.

    Returns:
        List of images for gallery, count string
    """
    gallery_items = history_manager.get_images_for_gallery()
    count = history_manager.get_count()
    count_str = f"📸 History: {count} image{'s' if count != 1 else ''}"
    return gallery_items, count_str


def clear_history():
    """Clear all history.

    Returns:
        Empty gallery, updated count string
    """
    history_manager.clear()
    logger.info("History cleared")
    return [], "📸 History: 0 images"


def get_history_info(evt: gr.SelectData):
    """Get information about a selected history item.

    Args:
        evt: Gradio SelectData event with selection index

    Returns:
        Formatted info string about the selected image
    """
    # evt.index is the position in the gallery (reversed)
    # We need to map it back to the actual history index
    total = history_manager.get_count()
    if total == 0:
        return "No history available"

    # Gallery shows newest first, so reverse the index
    actual_index = total - 1 - evt.index

    entry = history_manager.get_by_index(actual_index)
    if entry is None:
        return "Image not found in history"

    return entry.get_display_info()


def use_prompt_from_history(evt: gr.SelectData):
    """Extract prompt from selected history item.

    Args:
        evt: Gradio SelectData event with selection index

    Returns:
        The prompt from the selected image
    """
    total = history_manager.get_count()
    if total == 0:
        return ""

    # Gallery shows newest first, so reverse the index
    actual_index = total - 1 - evt.index

    entry = history_manager.get_by_index(actual_index)
    if entry is None:
        return ""

    return entry.generated_image.prompt


def download_image(format_choice: str) -> Optional[str]:
    """Download the last generated image in the specified format.

    Args:
        format_choice: Image format (PNG, JPEG, or WEBP)

    Returns:
        Path to temporary file for download, or None if no image available
    """
    if last_generated_image is None:
        logger.warning("No image available for download")
        return None

    try:
        import tempfile
        import os

        # Map format choice to ImageFormat
        format_map = {
            "PNG": ImageFormat.PNG,
            "JPEG": ImageFormat.JPEG,
            "WebP": ImageFormat.WEBP
        }

        format_enum = format_map.get(format_choice, ImageFormat.PNG)

        # Create downloadable image with metadata
        image_bytes, filename = create_downloadable_image(
            last_generated_image,
            format=format_enum
        )

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        logger.info(f"Created download: {filename} ({len(image_bytes)} bytes) at {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"Failed to create download: {e}")
        return None


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

        # Main tabs
        with gr.Tabs():
            # Generation Tab
            with gr.Tab("🎨 Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Backend selection
                        backend_selector = gr.Radio(
                            choices=["auto", "huggingface", "replicate", "local"],
                            value="auto",
                            label="Backend Selection",
                            info="Auto uses primary with fallback. Local = offline CPU inference"
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
                            # Aspect ratio presets
                            aspect_ratio_preset = gr.Radio(
                                choices=["Square (1:1)", "Portrait (3:4)", "Landscape (4:3)", "Widescreen (16:9)", "Custom"],
                                value="Square (1:1)",
                                label="Aspect Ratio",
                                info="Preset ratios or custom dimensions"
                            )

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

                        # Download controls
                        with gr.Row():
                            format_selector = gr.Radio(
                                choices=["PNG", "JPEG", "WebP"],
                                value="PNG",
                                label="Download Format",
                                info="Format for downloaded image"
                            )

                        download_btn = gr.DownloadButton(
                            label="💾 Download Image",
                            variant="secondary",
                            size="lg"
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

            # History Tab
            with gr.Tab("📸 History"):
                # Hidden state to track selected history index
                selected_history_index = gr.State(value=-1)

                with gr.Row():
                    history_count = gr.Textbox(
                        label="",
                        value="📸 History: 0 images",
                        interactive=False,
                        show_label=False
                    )
                    clear_history_btn = gr.Button("🗑️ Clear History", variant="stop")
                    use_prompt_btn = gr.Button("🔄 Use Selected Prompt", variant="primary")

                with gr.Row():
                    with gr.Column(scale=2):
                        history_gallery = gr.Gallery(
                            label="Generated Images",
                            show_label=False,
                            columns=4,
                            rows=3,
                            object_fit="contain",
                            height="auto"
                        )

                    with gr.Column(scale=1):
                        history_info = gr.Textbox(
                            label="Image Details",
                            lines=15,
                            interactive=False,
                            placeholder="Select an image to view details..."
                        )

            # Batch Generation Tab
            with gr.Tab("🎲 Batch Generate"):
                gr.Markdown("Generate multiple variations of the same prompt with different random seeds.")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Batch controls
                        batch_prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the images you want to generate...",
                            lines=3,
                            value="A serene landscape with mountains and a lake at sunset, digital art"
                        )

                        batch_negative_prompt_input = gr.Textbox(
                            label="Negative Prompt (optional)",
                            placeholder="What to avoid in the images...",
                            lines=2,
                            value="blurry, low quality, distorted"
                        )

                        batch_backend_selector = gr.Radio(
                            choices=["auto", "huggingface", "replicate", "local"],
                            value="auto",
                            label="Backend Selection",
                            info="Auto uses primary with fallback. Local = offline CPU inference"
                        )

                        batch_size_slider = gr.Slider(
                            minimum=2,
                            maximum=9,
                            value=4,
                            step=1,
                            label="Batch Size",
                            info="Number of variations to generate (2-9)"
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            # Aspect ratio presets
                            batch_aspect_ratio_preset = gr.Radio(
                                choices=["Square (1:1)", "Portrait (3:4)", "Landscape (4:3)", "Widescreen (16:9)", "Custom"],
                                value="Square (1:1)",
                                label="Aspect Ratio",
                                info="Preset ratios or custom dimensions"
                            )

                            batch_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale"
                            )

                            batch_num_steps = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1,
                                label="Inference Steps"
                            )

                            with gr.Row():
                                batch_width = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Width"
                                )

                                batch_height = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Height"
                                )

                        batch_generate_btn = gr.Button("🎲 Generate Batch", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        # Batch output
                        batch_gallery = gr.Gallery(
                            label="Generated Batch",
                            show_label=True,
                            columns=3,
                            rows=3,
                            object_fit="contain",
                            height="auto"
                        )

                        batch_info = gr.Textbox(
                            label="Batch Info",
                            lines=5,
                            interactive=False
                        )

        # Event handlers
        gen_event = generate_btn.click(
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

        # Update download button after generation
        gen_event.then(
            fn=download_image,
            inputs=[format_selector],
            outputs=download_btn
        ).then(
            fn=get_history_gallery,
            outputs=[history_gallery, history_count]
        )

        refresh_health.click(
            fn=get_health_status,
            outputs=[health_display]
        )

        # Update download when format changes
        format_selector.change(
            fn=download_image,
            inputs=[format_selector],
            outputs=download_btn
        )

        # History event handlers
        def handle_gallery_select(evt: gr.SelectData):
            """Handle gallery selection - return index and info."""
            info = get_history_info(evt)
            return evt.index, info

        history_gallery.select(
            fn=handle_gallery_select,
            outputs=[selected_history_index, history_info]
        )

        clear_history_btn.click(
            fn=clear_history,
            outputs=[history_gallery, history_count]
        ).then(
            fn=lambda: (-1, "Select an image to view details..."),
            outputs=[selected_history_index, history_info]
        )

        def use_selected_prompt(index):
            """Use prompt from the selected history item."""
            if index < 0:
                return ""
            total = history_manager.get_count()
            if total == 0:
                return ""
            actual_index = total - 1 - index
            entry = history_manager.get_by_index(actual_index)
            if entry is None:
                return ""
            return entry.generated_image.prompt

        use_prompt_btn.click(
            fn=use_selected_prompt,
            inputs=[selected_history_index],
            outputs=[prompt_input]
        )

        # Batch generation event handlers
        batch_gen_event = batch_generate_btn.click(
            fn=generate_batch_images,
            inputs=[
                batch_prompt_input,
                batch_size_slider,
                batch_backend_selector,
                batch_negative_prompt_input,
                batch_guidance_scale,
                batch_num_steps,
                batch_width,
                batch_height
            ],
            outputs=[batch_gallery, batch_info]
        )

        # Update history after batch generation
        batch_gen_event.then(
            fn=get_history_gallery,
            outputs=[history_gallery, history_count]
        )

        # Aspect ratio preset handlers
        def update_dimensions_from_preset(preset):
            """Update width and height based on aspect ratio preset."""
            presets = {
                "Square (1:1)": (512, 512),
                "Portrait (3:4)": (512, 704),
                "Landscape (4:3)": (704, 512),
                "Widescreen (16:9)": (896, 512),
                "Custom": (512, 512)  # Keep current values for custom
            }
            width, height = presets.get(preset, (512, 512))
            return width, height

        aspect_ratio_preset.change(
            fn=update_dimensions_from_preset,
            inputs=[aspect_ratio_preset],
            outputs=[width_slider, height_slider]
        )

        batch_aspect_ratio_preset.change(
            fn=update_dimensions_from_preset,
            inputs=[batch_aspect_ratio_preset],
            outputs=[batch_width, batch_height]
        )

        # Footer
        gr.Markdown(
            """
            ---
            💡 **Features:**
            - 🔄 **Multi-backend support**: HuggingFace + Replicate (Stage 2)
            - 🛡️ **Automatic fallback**: If primary fails, tries fallback (Stage 2)
            - 💚 **Health monitoring**: Check backend status in real-time (Stage 2)
            - 💾 **Image download**: Download with embedded metadata in PNG/JPEG/WebP (Stage 3)
            - 📸 **History gallery**: View and manage all generated images (Stage 3)
            - 🎲 **Batch generation**: Create multiple variations with different seeds (Stage 3)
            - 📐 **Aspect ratio presets**: Quick Square/Portrait/Landscape/Widescreen selection (Stage 3)

            **Tips:**
            - **Auto mode**: Uses configured primary backend with automatic fallback
            - **Specific backend**: Choose HuggingFace or Replicate to use only that service
            - **Download**: Select format and click download to save with metadata
            - **History**: Click images to view details, reuse prompts, or clear history
            - **Batch**: Generate 2-9 variations at once, all added to history
            - **Aspect ratios**: Use presets for common sizes or select "Custom" for manual control
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
