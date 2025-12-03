"""History management for generated images."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image
import io

from src.core.models import GeneratedImage


@dataclass
class HistoryEntry:
    """A single entry in the generation history.

    Attributes:
        generated_image: The GeneratedImage object
        pil_image: PIL Image for display
        index: Position in history (0 = oldest)
    """
    generated_image: GeneratedImage
    pil_image: Image.Image
    index: int

    def get_display_info(self) -> str:
        """Get formatted display information.

        Returns:
            Formatted string with generation details
        """
        metadata = self.generated_image.metadata
        timestamp_str = self.generated_image.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        info = f"**#{self.index + 1}** - {timestamp_str}\n"
        info += f"**Prompt:** {self.generated_image.prompt}\n"
        info += f"**Backend:** {self.generated_image.backend}\n"
        info += f"**Model:** {metadata.get('model', 'N/A')}\n"

        if metadata.get('width') and metadata.get('height'):
            info += f"**Size:** {metadata['width']}x{metadata['height']}\n"

        if metadata.get('num_inference_steps'):
            info += f"**Steps:** {metadata['num_inference_steps']}\n"

        if metadata.get('guidance_scale'):
            info += f"**Guidance:** {metadata['guidance_scale']}\n"

        return info

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as a dictionary.

        Returns:
            Dictionary with all metadata
        """
        return {
            "index": self.index,
            "timestamp": self.generated_image.timestamp.isoformat(),
            "prompt": self.generated_image.prompt,
            "backend": self.generated_image.backend,
            **self.generated_image.metadata
        }


class ImageHistoryManager:
    """Manages history of generated images."""

    def __init__(self, max_history: int = 50):
        """Initialize history manager.

        Args:
            max_history: Maximum number of images to keep in history
        """
        self.max_history = max_history
        self.history: List[HistoryEntry] = []

    def add(self, generated_image: GeneratedImage) -> HistoryEntry:
        """Add a generated image to history.

        Args:
            generated_image: The GeneratedImage to add

        Returns:
            The created HistoryEntry
        """
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(generated_image.image_data))

        # Create history entry
        entry = HistoryEntry(
            generated_image=generated_image,
            pil_image=pil_image,
            index=len(self.history)
        )

        # Add to history
        self.history.append(entry)

        # Trim if exceeds max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            # Re-index
            for i, entry in enumerate(self.history):
                entry.index = i

        return entry

    def get_all(self) -> List[HistoryEntry]:
        """Get all history entries.

        Returns:
            List of all HistoryEntry objects
        """
        return self.history.copy()

    def get_latest(self, n: int = 1) -> List[HistoryEntry]:
        """Get the N most recent entries.

        Args:
            n: Number of entries to return

        Returns:
            List of most recent HistoryEntry objects
        """
        return self.history[-n:] if self.history else []

    def get_by_index(self, index: int) -> Optional[HistoryEntry]:
        """Get a specific history entry by index.

        Args:
            index: Index of the entry

        Returns:
            HistoryEntry if found, None otherwise
        """
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()

    def get_images_for_gallery(self) -> List[tuple[Image.Image, str]]:
        """Get images formatted for Gradio Gallery.

        Returns:
            List of (image, caption) tuples for Gradio Gallery
        """
        gallery_items = []
        for entry in reversed(self.history):  # Most recent first
            caption = f"#{entry.index + 1}: {entry.generated_image.prompt[:60]}..."
            gallery_items.append((entry.pil_image, caption))
        return gallery_items

    def get_count(self) -> int:
        """Get number of images in history.

        Returns:
            Number of history entries
        """
        return len(self.history)

    def export_metadata(self) -> List[Dict[str, Any]]:
        """Export all metadata as a list of dictionaries.

        Returns:
            List of metadata dictionaries
        """
        return [entry.get_metadata_dict() for entry in self.history]
