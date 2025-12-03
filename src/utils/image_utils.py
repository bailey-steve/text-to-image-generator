"""Image utility functions for saving, format conversion, and metadata."""

import io
import json
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image, PngImagePlugin
from PIL.ExifTags import TAGS

from src.core.models import GeneratedImage


class ImageFormat:
    """Supported image formats."""
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"


def add_metadata_to_image(
    image: Image.Image,
    metadata: Dict[str, Any],
    format: str = ImageFormat.PNG
) -> bytes:
    """Add metadata to an image and return as bytes.

    Args:
        image: PIL Image object
        metadata: Dictionary of metadata to embed
        format: Output format (PNG, JPEG, or WEBP)

    Returns:
        Image bytes with embedded metadata
    """
    output = io.BytesIO()

    if format == ImageFormat.PNG:
        # For PNG, use pnginfo to store metadata
        pnginfo = PngImagePlugin.PngInfo()

        # Add each metadata field as a text chunk
        for key, value in metadata.items():
            if value is not None:
                pnginfo.add_text(key, str(value))

        # Add full metadata as JSON
        pnginfo.add_text("metadata_json", json.dumps(metadata, default=str))

        image.save(output, format="PNG", pnginfo=pnginfo)

    elif format == ImageFormat.JPEG:
        # JPEG doesn't support custom metadata easily, so we'll use EXIF
        # Convert to RGB if needed (JPEG doesn't support transparency)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image

        # Save with quality 95
        image.save(output, format="JPEG", quality=95)

    elif format == ImageFormat.WEBP:
        # WebP supports metadata
        # Convert RGBA to RGB for WebP if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image.save(output, format="WEBP", quality=95)

    output.seek(0)
    return output.getvalue()


def create_downloadable_image(
    generated_image: GeneratedImage,
    format: str = ImageFormat.PNG
) -> tuple[bytes, str]:
    """Create a downloadable image with metadata.

    Args:
        generated_image: GeneratedImage object
        format: Output format (PNG, JPEG, or WEBP)

    Returns:
        Tuple of (image_bytes, filename)
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(generated_image.image_data))

    # Prepare metadata
    metadata = {
        "prompt": generated_image.prompt,
        "backend": generated_image.backend,
        "timestamp": generated_image.timestamp.isoformat(),
        **generated_image.metadata
    }

    # Add metadata and convert to desired format
    image_bytes = add_metadata_to_image(image, metadata, format)

    # Generate filename
    timestamp_str = generated_image.timestamp.strftime("%Y%m%d_%H%M%S")
    # Clean prompt for filename (remove special chars, limit length)
    clean_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_'
                          for c in generated_image.prompt[:30])
    clean_prompt = clean_prompt.strip().replace(' ', '_')

    filename = f"{timestamp_str}_{clean_prompt}.{format.lower()}"

    return image_bytes, filename


def extract_metadata_from_image(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Extract metadata from an image file.

    Args:
        image_bytes: Image file bytes

    Returns:
        Dictionary of metadata if found, None otherwise
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Try PNG metadata
        if hasattr(image, 'text'):
            if 'metadata_json' in image.text:
                return json.loads(image.text['metadata_json'])
            # Otherwise return all text chunks
            elif image.text:
                return dict(image.text)

        # Try EXIF data
        exif_data = {}
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
            return exif_data

        return None
    except Exception:
        return None


def get_image_info(image_bytes: bytes) -> Dict[str, Any]:
    """Get basic information about an image.

    Args:
        image_bytes: Image file bytes

    Returns:
        Dictionary with image information (size, format, mode, etc.)
    """
    image = Image.open(io.BytesIO(image_bytes))

    return {
        "width": image.width,
        "height": image.height,
        "format": image.format,
        "mode": image.mode,
        "size_bytes": len(image_bytes)
    }
