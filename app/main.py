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
from src.utils.prompt_enhancer import (
    get_prompt_enhancer,
    PromptStyle,
    PromptQuality,
    PromptLibrary
)

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

# Global prompt enhancer
prompt_enhancer = get_prompt_enhancer()


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
                settings.huggingface_token,
                model=settings.huggingface_model
            )
        elif settings.default_backend == "replicate":
            primary = BackendFactory.create_backend(
                "replicate",
                settings.replicate_token or "",
                model=settings.replicate_model
            )
        elif settings.default_backend == "local":
            primary = BackendFactory.create_backend(
                "local",
                model=settings.local_model
            )
        else:
            primary = BackendFactory.create_backend(
                "huggingface",
                settings.huggingface_token,
                model=settings.huggingface_model
            )

        # Create fallback backends if enabled
        fallbacks = []
        if settings.enable_fallback and settings.fallback_backend:
            try:
                if settings.fallback_backend == "replicate" and settings.replicate_token:
                    fallbacks.append(
                        BackendFactory.create_backend(
                            "replicate",
                            settings.replicate_token,
                            model=settings.replicate_model
                        )
                    )
                elif settings.fallback_backend == "huggingface" and settings.huggingface_token:
                    fallbacks.append(
                        BackendFactory.create_backend(
                            "huggingface",
                            settings.huggingface_token,
                            model=settings.huggingface_model
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
                        temp_backend = BackendFactory.create_backend("huggingface", settings.huggingface_token, model=settings.huggingface_model)
                        generator.primary_backend = temp_backend
                        use_fallback = False
                    elif backend_choice == "replicate" and settings.replicate_token:
                        temp_backend = BackendFactory.create_backend("replicate", settings.replicate_token, model=settings.replicate_model)
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


def generate_image_to_image(
    input_image: Optional[Image.Image],
    prompt: str,
    strength: float = 0.8,
    backend_choice: str = "auto",
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_steps: int = 4
) -> Tuple[Optional[Image.Image], str]:
    """Generate an image from an input image and text prompt.

    Args:
        input_image: PIL Image to transform
        prompt: Text description of desired transformation
        strength: How much to transform (0.0-1.0)
        backend_choice: Backend to use
        negative_prompt: What to avoid
        guidance_scale: How closely to follow prompt
        num_steps: Number of inference steps

    Returns:
        Tuple of (PIL Image object or None, status message)
    """
    if input_image is None:
        return None, "Error: Please upload an input image"

    if not prompt or prompt.strip() == "":
        return None, "Error: Please enter a prompt"

    if generator is None:
        return None, "❌ Error: Generator not initialized"

    try:
        logger.info(f"Generating image-to-image with prompt: {prompt[:50]}...")

        # Convert PIL Image to bytes
        import io
        img_byte_arr = io.BytesIO()
        input_image.save(img_byte_arr, format='PNG')
        init_image_bytes = img_byte_arr.getvalue()

        # Create generation request
        request = GenerationRequest(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip() if negative_prompt else None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            init_image=init_image_bytes,
            strength=strength,
            width=512,  # These are ignored for img2img but required by model
            height=512
        )

        # Determine whether to use fallback
        use_fallback = (backend_choice == "auto")

        # If user selected specific backend, temporarily switch primary
        if backend_choice != "auto":
            original_primary = generator.primary_backend

            try:
                if backend_choice == "huggingface" and settings.huggingface_token:
                    temp_backend = BackendFactory.create_backend(
                        "huggingface",
                        settings.huggingface_token,
                        model=settings.huggingface_model
                    )
                    generator.primary_backend = temp_backend
                    use_fallback = False
                elif backend_choice == "replicate" and settings.replicate_token:
                    temp_backend = BackendFactory.create_backend(
                        "replicate",
                        settings.replicate_token,
                        model=settings.replicate_model
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
        output_image = Image.open(io.BytesIO(result.image_data))

        # Create info message
        info_message = (
            f"✅ Image transformed successfully!\n"
            f"Backend: {result.backend}\n"
            f"Model: {result.metadata.get('model', 'N/A')}\n"
            f"Strength: {strength}\n"
            f"Steps: {num_steps}\n"
            f"Guidance Scale: {guidance_scale}"
        )

        return output_image, info_message

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
                        settings.huggingface_token,
                        model=settings.huggingface_model
                    )
                    generator.primary_backend = temp_backend
                    use_fallback = False
                elif backend_choice == "replicate" and settings.replicate_token:
                    temp_backend = BackendFactory.create_backend(
                        "replicate",
                        settings.replicate_token,
                        model=settings.replicate_model
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


def enhance_prompt_with_settings(
    prompt: str,
    style_choice: str,
    quality_choice: str,
    add_details: bool
) -> str:
    """Enhance a prompt with selected style and quality.

    Args:
        prompt: Original prompt text
        style_choice: Selected style name
        quality_choice: Selected quality name
        add_details: Whether to add detail enhancers

    Returns:
        Enhanced prompt string
    """
    if not prompt or prompt.strip() == "":
        return prompt

    # Map UI choices to enums
    style_map = {
        "None": None,
        "Photorealistic": PromptStyle.PHOTOREALISTIC,
        "Artistic": PromptStyle.ARTISTIC,
        "Anime": PromptStyle.ANIME,
        "Digital Art": PromptStyle.DIGITAL_ART,
        "Oil Painting": PromptStyle.OIL_PAINTING,
        "Watercolor": PromptStyle.WATERCOLOR,
        "Sketch": PromptStyle.SKETCH,
        "Cyberpunk": PromptStyle.CYBERPUNK,
        "Fantasy": PromptStyle.FANTASY,
        "Minimalist": PromptStyle.MINIMALIST,
    }

    quality_map = {
        "None": None,
        "Standard": PromptQuality.STANDARD,
        "High Quality": PromptQuality.HIGH_QUALITY,
        "Masterpiece": PromptQuality.MASTERPIECE,
        "Professional": PromptQuality.PROFESSIONAL,
    }

    style = style_map.get(style_choice)
    quality = quality_map.get(quality_choice)

    enhanced = prompt_enhancer.enhance_prompt(
        prompt,
        style=style,
        quality=quality,
        add_details=add_details
    )

    return enhanced


def apply_template(
    template_choice: str,
    **template_params
) -> str:
    """Apply a prompt template with given parameters.

    Args:
        template_choice: Selected template name
        **template_params: Template parameters

    Returns:
        Formatted prompt from template
    """
    if template_choice == "None":
        return ""

    template = PromptLibrary.get_template(template_choice.lower())
    if template is None:
        return ""

    try:
        # Filter out empty parameters
        params = {k: v for k, v in template_params.items() if v}
        return template.format(**params)
    except KeyError as e:
        return f"Error: Missing template parameter {e}"


def get_template_parameters(template_choice: str) -> dict:
    """Get required parameters for a template.

    Args:
        template_choice: Selected template name

    Returns:
        Dictionary with parameter info
    """
    if template_choice == "None":
        return {}

    template = PromptLibrary.get_template(template_choice.lower())
    if template is None:
        return {}

    # Extract parameters from template string
    import re
    params = re.findall(r'\{(\w+)\}', template.template)
    return {param: "" for param in params}


def generate_negative_prompt_from_defaults() -> str:
    """Generate negative prompt from default terms.

    Returns:
        Comma-separated negative prompt string
    """
    return prompt_enhancer.generate_negative_prompt()


def get_prompt_suggestions(prompt: str) -> str:
    """Get improvement suggestions for a prompt.

    Args:
        prompt: Prompt to analyze

    Returns:
        Formatted suggestions text
    """
    if not prompt or prompt.strip() == "":
        return "Enter a prompt to get suggestions"

    suggestions = prompt_enhancer.suggest_improvements(prompt)

    output = "## Analysis\n\n"

    if suggestions["issues"]:
        output += "**Issues Found:**\n"
        for issue in suggestions["issues"]:
            output += f"- {issue}\n"
        output += "\n"

    if suggestions["recommendations"]:
        output += "**Recommendations:**\n"
        for rec in suggestions["recommendations"]:
            output += f"- {rec}\n"
        output += "\n"

    if suggestions["enhanced_examples"]:
        output += "**Enhanced Examples:**\n\n"
        for i, example in enumerate(suggestions["enhanced_examples"], 1):
            output += f"{i}. `{example}`\n\n"

    return output


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

                        with gr.Accordion("✨ Prompt Enhancement", open=True):
                            gr.Markdown("Enhance your prompts with styles, quality presets, and templates")

                            # Style and Quality selectors
                            with gr.Row():
                                style_selector = gr.Dropdown(
                                    choices=["None", "Photorealistic", "Artistic", "Anime", "Digital Art",
                                            "Oil Painting", "Watercolor", "Sketch", "Cyberpunk", "Fantasy", "Minimalist"],
                                    value="None",
                                    label="Style Preset",
                                    info="Add style modifiers to your prompt"
                                )

                                quality_selector = gr.Dropdown(
                                    choices=["None", "Standard", "High Quality", "Masterpiece", "Professional"],
                                    value="None",
                                    label="Quality Level",
                                    info="Add quality enhancers"
                                )

                            add_details_checkbox = gr.Checkbox(
                                label="Add detail enhancers (sharp focus, detailed, intricate)",
                                value=True
                            )

                            with gr.Row():
                                enhance_button = gr.Button("✨ Enhance Prompt", variant="secondary", size="sm")
                                negative_defaults_button = gr.Button("🚫 Add Negative Defaults", variant="secondary", size="sm")
                                suggestions_button = gr.Button("💡 Get Suggestions", variant="secondary", size="sm")

                            # Template section
                            with gr.Accordion("📋 Use Template", open=False):
                                template_selector = gr.Dropdown(
                                    choices=["None", "Portrait", "Landscape", "Character", "Architecture",
                                            "Product", "Animal", "Abstract", "Food"],
                                    value="None",
                                    label="Template",
                                    info="Start with a pre-built template"
                                )

                                template_info = gr.Markdown("Select a template to see required parameters")

                                # Dynamic template parameter inputs
                                template_param1 = gr.Textbox(label="Parameter 1", visible=False)
                                template_param2 = gr.Textbox(label="Parameter 2", visible=False)
                                template_param3 = gr.Textbox(label="Parameter 3", visible=False)
                                template_param4 = gr.Textbox(label="Parameter 4", visible=False)
                                template_param5 = gr.Textbox(label="Parameter 5", visible=False)

                                apply_template_button = gr.Button("📋 Apply Template", variant="secondary", size="sm")

                            # Suggestions display
                            suggestions_display = gr.Markdown(visible=False, label="Suggestions")

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

            # Image-to-Image Tab
            with gr.Tab("🖼️ Image-to-Image"):
                gr.Markdown("Transform an existing image based on a text prompt. Upload an image and describe how you want to change it.")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Input image upload
                        img2img_input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            show_label=True
                        )

                        img2img_prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe how you want to transform the image...",
                            lines=3,
                            value="Turn this into a watercolor painting"
                        )

                        img2img_negative_prompt_input = gr.Textbox(
                            label="Negative Prompt (optional)",
                            placeholder="What to avoid in the image...",
                            lines=2,
                            value="blurry, low quality, distorted"
                        )

                        img2img_strength_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            label="Transformation Strength",
                            info="0 = keep original, 1 = completely new image"
                        )

                        img2img_backend_selector = gr.Radio(
                            choices=["auto", "huggingface", "replicate", "local"],
                            value="auto",
                            label="Backend Selection",
                            info="Auto uses primary with fallback"
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            img2img_guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale"
                            )

                            img2img_num_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=20,
                                step=1,
                                label="Inference Steps",
                                info="SDXL: 20-30 recommended, FLUX: 4-8"
                            )

                        img2img_generate_btn = gr.Button("🖼️ Transform Image", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        # Output
                        img2img_output_image = gr.Image(
                            label="Transformed Image",
                            type="pil",
                            show_label=True
                        )

                        img2img_output_info = gr.Textbox(
                            label="Generation Info",
                            lines=8,
                            interactive=False
                        )

                # Examples
                gr.Examples(
                    examples=[
                        ["Turn this into a watercolor painting", 0.7],
                        ["Make it look like a cyberpunk scene at night", 0.8],
                        ["Transform into an oil painting in Van Gogh style", 0.75],
                        ["Add dramatic sunset lighting", 0.6],
                    ],
                    inputs=[img2img_prompt_input, img2img_strength_slider],
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

        # Image-to-Image event handlers
        img2img_gen_event = img2img_generate_btn.click(
            fn=generate_image_to_image,
            inputs=[
                img2img_input_image,
                img2img_prompt_input,
                img2img_strength_slider,
                img2img_backend_selector,
                img2img_negative_prompt_input,
                img2img_guidance_scale,
                img2img_num_steps
            ],
            outputs=[img2img_output_image, img2img_output_info]
        )

        # Update history after img2img generation
        img2img_gen_event.then(
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

        # Prompt enhancement event handlers
        enhance_button.click(
            fn=enhance_prompt_with_settings,
            inputs=[prompt_input, style_selector, quality_selector, add_details_checkbox],
            outputs=[prompt_input]
        )

        negative_defaults_button.click(
            fn=generate_negative_prompt_from_defaults,
            outputs=[negative_prompt_input]
        )

        def show_suggestions(prompt):
            """Show suggestions and make visible."""
            suggestions = get_prompt_suggestions(prompt)
            return gr.Markdown(value=suggestions, visible=True)

        suggestions_button.click(
            fn=show_suggestions,
            inputs=[prompt_input],
            outputs=[suggestions_display]
        )

        # Template event handlers
        def update_template_ui(template_choice):
            """Update UI when template is selected."""
            if template_choice == "None":
                return (
                    "Select a template to see required parameters",
                    gr.Textbox(visible=False), gr.Textbox(visible=False),
                    gr.Textbox(visible=False), gr.Textbox(visible=False),
                    gr.Textbox(visible=False)
                )

            template = PromptLibrary.get_template(template_choice.lower())
            if template is None:
                return (
                    "Template not found",
                    gr.Textbox(visible=False), gr.Textbox(visible=False),
                    gr.Textbox(visible=False), gr.Textbox(visible=False),
                    gr.Textbox(visible=False)
                )

            # Extract parameters from template
            import re
            params = re.findall(r'\{(\w+)\}', template.template)

            info = f"**{template.name.title()}**: {template.description}\n\n"
            info += f"**Example**: {template.example}\n\n"
            info += f"**Required parameters**: {', '.join(params)}"

            # Show textboxes for each parameter
            textboxes = []
            for i in range(5):
                if i < len(params):
                    textboxes.append(gr.Textbox(label=params[i].replace('_', ' ').title(), visible=True, placeholder=f"Enter {params[i]}"))
                else:
                    textboxes.append(gr.Textbox(visible=False))

            return (info, *textboxes)

        template_selector.change(
            fn=update_template_ui,
            inputs=[template_selector],
            outputs=[template_info, template_param1, template_param2, template_param3, template_param4, template_param5]
        )

        def apply_template_to_prompt(template_choice, p1, p2, p3, p4, p5):
            """Apply selected template with parameters."""
            if template_choice == "None":
                return ""

            template = PromptLibrary.get_template(template_choice.lower())
            if template is None:
                return "Template not found"

            # Extract parameter names
            import re
            params = re.findall(r'\{(\w+)\}', template.template)

            # Build parameter dict with all parameters
            param_values = [p1, p2, p3, p4, p5]
            param_dict = {}
            missing_params = []

            for i, param_name in enumerate(params):
                if i < len(param_values) and param_values[i]:
                    param_dict[param_name] = param_values[i]
                else:
                    # Use placeholder for missing parameters
                    param_dict[param_name] = f"[{param_name.replace('_', ' ')}]"
                    missing_params.append(param_name)

            try:
                result = template.format(**param_dict)
                if missing_params:
                    # Add note about missing parameters
                    result += f" (Note: Fill in [{', '.join(missing_params)}] placeholders)"
                return result
            except KeyError as e:
                return f"Error: Missing required parameter: {e}"

        apply_template_button.click(
            fn=apply_template_to_prompt,
            inputs=[template_selector, template_param1, template_param2, template_param3, template_param4, template_param5],
            outputs=[prompt_input]
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
            - ✨ **Prompt enhancement**: 10 style presets, 4 quality levels, 8 templates (Improvement #2)
            - 🖼️ **Image-to-Image**: Transform existing images with text prompts (Improvement #3)

            **Tips:**
            - **Auto mode**: Uses configured primary backend with automatic fallback
            - **Specific backend**: Choose HuggingFace or Replicate to use only that service
            - **Prompt Enhancement**: Use style presets (Photorealistic, Anime, etc.) and quality levels to improve prompts
            - **Templates**: Start with pre-built templates for portraits, landscapes, characters, and more
            - **Suggestions**: Click "Get Suggestions" to analyze your prompt and see enhancement examples
            - **Negative defaults**: Auto-fill negative prompt with common unwanted terms
            - **Image-to-Image**: Upload an image and describe how to transform it. Strength controls how much it changes (0=original, 1=new)
            - **Download**: Select format and click download to save with metadata
            - **History**: Click images to view details, reuse prompts, or clear history
            - **Batch**: Generate 2-9 variations at once, all added to history
            - **Aspect ratios**: Use presets for common sizes or select "Custom" for manual control
            """
        )

    return demo


if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()

    logger.info("Launching Gradio application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
