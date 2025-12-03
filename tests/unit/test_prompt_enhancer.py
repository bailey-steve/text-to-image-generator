"""Unit tests for prompt enhancer."""

import pytest
from src.utils.prompt_enhancer import (
    PromptEnhancer,
    PromptLibrary,
    PromptTemplate,
    PromptStyle,
    PromptQuality,
    get_prompt_enhancer,
    reset_prompt_enhancer
)


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_create_template(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test",
            category="test_cat",
            template="{subject} in {style}",
            description="Test template",
            tags=["test", "example"],
            example="cat in watercolor"
        )

        assert template.name == "test"
        assert template.category == "test_cat"
        assert "test" in template.tags

    def test_format_template(self):
        """Test formatting a template with values."""
        template = PromptTemplate(
            name="test",
            category="test",
            template="{subject} doing {action}",
            description="Test",
            tags=["test"],
            example="example"
        )

        result = template.format(subject="cat", action="sleeping")

        assert result == "cat doing sleeping"


class TestPromptLibrary:
    """Tests for PromptLibrary class."""

    def test_has_templates(self):
        """Test that library has pre-defined templates."""
        assert len(PromptLibrary.TEMPLATES) > 0

    def test_get_template_by_name(self):
        """Test getting a template by name."""
        template = PromptLibrary.get_template("portrait")

        assert template is not None
        assert template.name == "portrait"
        assert "subject" in template.template

    def test_get_nonexistent_template(self):
        """Test getting a template that doesn't exist."""
        template = PromptLibrary.get_template("nonexistent")

        assert template is None

    def test_get_templates_by_category(self):
        """Test getting templates by category."""
        people_templates = PromptLibrary.get_templates_by_category("people")

        assert len(people_templates) > 0
        assert all(t.category == "people" for t in people_templates)

    def test_search_templates_by_name(self):
        """Test searching templates by name."""
        results = PromptLibrary.search_templates("portrait")

        assert len(results) > 0
        assert any("portrait" in t.name.lower() for t in results)

    def test_search_templates_by_tag(self):
        """Test searching templates by tag."""
        results = PromptLibrary.search_templates("nature")

        assert len(results) > 0

    def test_get_all_categories(self):
        """Test getting all categories."""
        categories = PromptLibrary.get_all_categories()

        assert len(categories) > 0
        assert "people" in categories or "nature" in categories

    def test_style_modifiers_exist(self):
        """Test that style modifiers are defined."""
        assert len(PromptLibrary.STYLE_MODIFIERS) > 0
        assert PromptStyle.PHOTOREALISTIC in PromptLibrary.STYLE_MODIFIERS

    def test_quality_modifiers_exist(self):
        """Test that quality modifiers are defined."""
        assert len(PromptLibrary.QUALITY_MODIFIERS) > 0
        assert PromptQuality.MASTERPIECE in PromptLibrary.QUALITY_MODIFIERS

    def test_negative_prompt_defaults(self):
        """Test that default negative prompts exist."""
        assert len(PromptLibrary.NEGATIVE_PROMPT_DEFAULTS) > 0
        assert "blurry" in PromptLibrary.NEGATIVE_PROMPT_DEFAULTS


class TestPromptEnhancer:
    """Tests for PromptEnhancer class."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_prompt_enhancer()
        self.enhancer = PromptEnhancer()

    def teardown_method(self):
        """Clean up after tests."""
        reset_prompt_enhancer()

    def test_initialization(self):
        """Test enhancer initialization."""
        enhancer = PromptEnhancer()

        assert enhancer is not None

    def test_enhance_prompt_basic(self):
        """Test basic prompt enhancement."""
        original = "a cat sitting"
        enhanced = self.enhancer.enhance_prompt(original)

        assert len(enhanced) > len(original)
        assert "cat sitting" in enhanced

    def test_enhance_prompt_with_style(self):
        """Test enhancement with style modifier."""
        original = "a beautiful sunset"
        enhanced = self.enhancer.enhance_prompt(
            original,
            style=PromptStyle.PHOTOREALISTIC
        )

        assert "photorealistic" in enhanced.lower()
        assert "sunset" in enhanced

    def test_enhance_prompt_with_quality(self):
        """Test enhancement with quality modifier."""
        original = "a mountain landscape"
        enhanced = self.enhancer.enhance_prompt(
            original,
            quality=PromptQuality.MASTERPIECE
        )

        assert "masterpiece" in enhanced.lower() or "quality" in enhanced.lower()

    def test_enhance_prompt_with_style_and_quality(self):
        """Test enhancement with both style and quality."""
        original = "a futuristic city"
        enhanced = self.enhancer.enhance_prompt(
            original,
            style=PromptStyle.DIGITAL_ART,
            quality=PromptQuality.HIGH_QUALITY
        )

        assert len(enhanced) > len(original)
        assert "digital art" in enhanced.lower()
        assert "quality" in enhanced.lower()

    def test_enhance_prompt_with_details(self):
        """Test that detail enhancers are added."""
        original = "a simple object"
        enhanced = self.enhancer.enhance_prompt(original, add_details=True)

        assert "detailed" in enhanced.lower() or "sharp focus" in enhanced.lower()

    def test_enhance_prompt_without_details(self):
        """Test enhancement without detail additions."""
        original = "a simple object"
        enhanced = self.enhancer.enhance_prompt(original, add_details=False)

        # Should still be enhanced but maybe not have "detailed"
        assert len(enhanced) >= len(original)

    def test_clean_prompt_removes_extra_spaces(self):
        """Test that extra spaces are removed."""
        dirty = "a  cat   with    spaces"
        clean = self.enhancer._clean_prompt(dirty)

        assert "  " not in clean
        assert clean == "a cat with spaces"

    def test_clean_prompt_removes_duplicate_commas(self):
        """Test that duplicate commas are removed."""
        dirty = "cat,, dog,, bird"
        clean = self.enhancer._clean_prompt(dirty)

        assert ",," not in clean

    def test_clean_prompt_removes_leading_trailing_commas(self):
        """Test that leading/trailing commas are removed."""
        dirty = ", cat, dog, "
        clean = self.enhancer._clean_prompt(dirty)

        assert not clean.startswith(",")
        assert not clean.endswith(",")

    def test_generate_negative_prompt_with_defaults(self):
        """Test generating negative prompt with defaults."""
        negative = self.enhancer.generate_negative_prompt()

        assert len(negative) > 0
        assert "blurry" in negative
        assert "low quality" in negative

    def test_generate_negative_prompt_with_custom(self):
        """Test generating negative prompt with custom terms."""
        custom = ["cartoon", "anime"]
        negative = self.enhancer.generate_negative_prompt(
            custom_negatives=custom,
            include_defaults=False
        )

        assert "cartoon" in negative
        assert "anime" in negative
        assert "blurry" not in negative

    def test_generate_negative_prompt_removes_duplicates(self):
        """Test that duplicate negatives are removed."""
        custom = ["blurry", "blurry", "ugly"]
        negative = self.enhancer.generate_negative_prompt(
            custom_negatives=custom,
            include_defaults=True
        )

        # Count occurrences of "blurry"
        count = negative.lower().count("blurry")
        assert count == 1

    def test_suggest_improvements_short_prompt(self):
        """Test suggestions for a short prompt."""
        short_prompt = "cat"
        suggestions = self.enhancer.suggest_improvements(short_prompt)

        assert "original" in suggestions
        assert len(suggestions["issues"]) > 0
        assert any("short" in issue.lower() for issue in suggestions["issues"])

    def test_suggest_improvements_no_style(self):
        """Test suggestions for prompt without style."""
        prompt = "a beautiful detailed scene"
        suggestions = self.enhancer.suggest_improvements(prompt)

        assert any("style" in issue.lower() for issue in suggestions["issues"])

    def test_suggest_improvements_includes_examples(self):
        """Test that suggestions include enhanced examples."""
        prompt = "a cat"
        suggestions = self.enhancer.suggest_improvements(prompt)

        assert "enhanced_examples" in suggestions
        assert len(suggestions["enhanced_examples"]) > 0
        assert all(len(ex) > len(prompt) for ex in suggestions["enhanced_examples"])

    def test_repr(self):
        """Test string representation."""
        enhancer = PromptEnhancer()
        repr_str = repr(enhancer)

        assert "PromptEnhancer" in repr_str
        assert "templates" in repr_str
        assert "styles" in repr_str

    def test_global_prompt_enhancer(self):
        """Test global prompt enhancer singleton."""
        enhancer1 = get_prompt_enhancer()
        enhancer2 = get_prompt_enhancer()

        assert enhancer1 is enhancer2

    def test_reset_global_prompt_enhancer(self):
        """Test resetting global prompt enhancer."""
        enhancer1 = get_prompt_enhancer()
        reset_prompt_enhancer()
        enhancer2 = get_prompt_enhancer()

        assert enhancer1 is not enhancer2

    def test_enhance_preserves_original_meaning(self):
        """Test that enhancement doesn't change original meaning."""
        original = "a red car in the rain"
        enhanced = self.enhancer.enhance_prompt(original)

        assert "red" in enhanced
        assert "car" in enhanced
        assert "rain" in enhanced

    def test_all_styles_work(self):
        """Test that all style enums work."""
        prompt = "test prompt"

        for style in PromptStyle:
            enhanced = self.enhancer.enhance_prompt(prompt, style=style)
            assert len(enhanced) > 0
            assert "test prompt" in enhanced

    def test_all_qualities_work(self):
        """Test that all quality enums work."""
        prompt = "test prompt"

        for quality in PromptQuality:
            enhanced = self.enhancer.enhance_prompt(prompt, quality=quality)
            assert len(enhanced) > 0
            assert "test prompt" in enhanced
