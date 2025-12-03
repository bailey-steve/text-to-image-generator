"""Unit tests for history manager."""

import pytest
from PIL import Image
import io
from datetime import datetime

from src.utils.history_manager import HistoryEntry, ImageHistoryManager
from src.core.models import GeneratedImage


class TestHistoryEntry:
    """Tests for HistoryEntry dataclass."""

    def test_get_display_info_basic(self):
        """Test basic display info formatting."""
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="A test image",
            backend="HuggingFace",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metadata={
                "model": "test-model",
                "width": 512,
                "height": 512,
                "num_inference_steps": 4,
                "guidance_scale": 7.5
            }
        )

        entry = HistoryEntry(
            generated_image=generated,
            pil_image=test_image,
            index=0
        )

        info = entry.get_display_info()

        assert "**#1**" in info
        assert "2024-01-01 12:00:00" in info
        assert "A test image" in info
        assert "HuggingFace" in info
        assert "test-model" in info
        assert "512x512" in info
        assert "Steps:** 4" in info
        assert "Guidance:** 7.5" in info

    def test_get_display_info_minimal(self):
        """Test display info with minimal metadata."""
        test_image = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="Minimal test",
            backend="Replicate",
            metadata={}
        )

        entry = HistoryEntry(
            generated_image=generated,
            pil_image=test_image,
            index=5
        )

        info = entry.get_display_info()

        assert "**#6**" in info
        assert "Minimal test" in info
        assert "Replicate" in info
        assert "**Model:** N/A" in info

    def test_get_metadata_dict(self):
        """Test metadata dictionary export."""
        test_image = Image.new('RGB', (100, 100), color='green')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="Export test",
            backend="HuggingFace",
            timestamp=datetime(2024, 6, 15, 10, 30, 45),
            metadata={"model": "flux", "steps": 8}
        )

        entry = HistoryEntry(
            generated_image=generated,
            pil_image=test_image,
            index=3
        )

        metadata = entry.get_metadata_dict()

        assert metadata["index"] == 3
        assert metadata["timestamp"] == "2024-06-15T10:30:45"
        assert metadata["prompt"] == "Export test"
        assert metadata["backend"] == "HuggingFace"
        assert metadata["model"] == "flux"
        assert metadata["steps"] == 8


class TestImageHistoryManager:
    """Tests for ImageHistoryManager class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        manager = ImageHistoryManager()

        assert manager.max_history == 50
        assert len(manager.history) == 0

    def test_init_custom_max(self):
        """Test initialization with custom max_history."""
        manager = ImageHistoryManager(max_history=10)

        assert manager.max_history == 10

    def test_add_single_image(self):
        """Test adding a single image."""
        manager = ImageHistoryManager()

        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt="Test",
            backend="HuggingFace",
            metadata={}
        )

        entry = manager.add(generated)

        assert len(manager.history) == 1
        assert entry.index == 0
        assert entry.generated_image == generated
        assert isinstance(entry.pil_image, Image.Image)

    def test_add_multiple_images(self):
        """Test adding multiple images."""
        manager = ImageHistoryManager()

        for i in range(5):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            entry = manager.add(generated)
            assert entry.index == i

        assert len(manager.history) == 5

    def test_add_exceeds_max_history(self):
        """Test that history is trimmed when exceeding max."""
        manager = ImageHistoryManager(max_history=3)

        # Add 5 images
        for i in range(5):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        # Should only keep last 3
        assert len(manager.history) == 3
        assert manager.history[0].generated_image.prompt == "Test 2"
        assert manager.history[1].generated_image.prompt == "Test 3"
        assert manager.history[2].generated_image.prompt == "Test 4"

        # Indices should be re-indexed
        assert manager.history[0].index == 0
        assert manager.history[1].index == 1
        assert manager.history[2].index == 2

    def test_get_all(self):
        """Test getting all history entries."""
        manager = ImageHistoryManager()

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        all_entries = manager.get_all()

        assert len(all_entries) == 3
        assert all_entries is not manager.history

    def test_get_latest_single(self):
        """Test getting latest single entry."""
        manager = ImageHistoryManager()

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        latest = manager.get_latest(n=1)

        assert len(latest) == 1
        assert latest[0].generated_image.prompt == "Test 2"

    def test_get_latest_multiple(self):
        """Test getting latest N entries."""
        manager = ImageHistoryManager()

        for i in range(5):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        latest = manager.get_latest(n=3)

        assert len(latest) == 3
        assert latest[0].generated_image.prompt == "Test 2"
        assert latest[1].generated_image.prompt == "Test 3"
        assert latest[2].generated_image.prompt == "Test 4"

    def test_get_latest_empty_history(self):
        """Test getting latest from empty history."""
        manager = ImageHistoryManager()

        latest = manager.get_latest(n=5)

        assert len(latest) == 0

    def test_get_by_index_valid(self):
        """Test getting entry by valid index."""
        manager = ImageHistoryManager()

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        entry = manager.get_by_index(1)

        assert entry is not None
        assert entry.generated_image.prompt == "Test 1"
        assert entry.index == 1

    def test_get_by_index_invalid(self):
        """Test getting entry by invalid index."""
        manager = ImageHistoryManager()

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        assert manager.get_by_index(-1) is None
        assert manager.get_by_index(10) is None

    def test_clear(self):
        """Test clearing history."""
        manager = ImageHistoryManager()

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        assert len(manager.history) == 3

        manager.clear()

        assert len(manager.history) == 0

    def test_get_images_for_gallery(self):
        """Test getting images formatted for Gradio Gallery."""
        manager = ImageHistoryManager()

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test prompt {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        gallery_items = manager.get_images_for_gallery()

        assert len(gallery_items) == 3

        # Should be reversed (most recent first)
        assert gallery_items[0][1] == "#3: Test prompt 2..."
        assert gallery_items[1][1] == "#2: Test prompt 1..."
        assert gallery_items[2][1] == "#1: Test prompt 0..."

        for image, caption in gallery_items:
            assert isinstance(image, Image.Image)
            assert isinstance(caption, str)

    def test_get_images_for_gallery_long_prompt(self):
        """Test that long prompts are truncated in gallery captions."""
        manager = ImageHistoryManager()

        long_prompt = "A" * 100

        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')

        generated = GeneratedImage(
            image_data=img_bytes.getvalue(),
            prompt=long_prompt,
            backend="HuggingFace",
            metadata={}
        )

        manager.add(generated)

        gallery_items = manager.get_images_for_gallery()
        caption = gallery_items[0][1]

        assert len(caption) < len(long_prompt) + 10

    def test_get_count(self):
        """Test getting count of history entries."""
        manager = ImageHistoryManager()

        assert manager.get_count() == 0

        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                metadata={}
            )

            manager.add(generated)

        assert manager.get_count() == 3

    def test_export_metadata(self):
        """Test exporting metadata for all entries."""
        manager = ImageHistoryManager()

        for i in range(2):
            test_image = Image.new('RGB', (100, 100), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')

            generated = GeneratedImage(
                image_data=img_bytes.getvalue(),
                prompt=f"Test {i}",
                backend="HuggingFace",
                timestamp=datetime(2024, 1, i+1, 12, 0, 0),
                metadata={"model": f"model-{i}", "steps": i+1}
            )

            manager.add(generated)

        metadata_list = manager.export_metadata()

        assert len(metadata_list) == 2
        assert metadata_list[0]["index"] == 0
        assert metadata_list[0]["prompt"] == "Test 0"
        assert metadata_list[0]["model"] == "model-0"
        assert metadata_list[1]["index"] == 1
        assert metadata_list[1]["prompt"] == "Test 1"
        assert metadata_list[1]["model"] == "model-1"
