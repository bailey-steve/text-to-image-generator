"""Prompt enhancement utilities for better image generation."""

import logging
import re
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PromptStyle(Enum):
    """Available prompt styles."""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    DIGITAL_ART = "digital_art"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    MINIMALIST = "minimalist"


class PromptQuality(Enum):
    """Quality enhancement levels."""
    STANDARD = "standard"
    HIGH_QUALITY = "high_quality"
    MASTERPIECE = "masterpiece"
    PROFESSIONAL = "professional"


@dataclass
class PromptTemplate:
    """Template for common prompt patterns."""
    name: str
    category: str
    template: str
    description: str
    tags: List[str]
    example: str

    def format(self, **kwargs) -> str:
        """Format template with provided values.

        Args:
            **kwargs: Values to substitute in template

        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)


class PromptLibrary:
    """Library of prompt templates and patterns."""

    TEMPLATES = [
        PromptTemplate(
            name="portrait",
            category="people",
            template="portrait of {subject}, {style}, {quality}, detailed face, {lighting}",
            description="High-quality portrait generation",
            tags=["portrait", "face", "person"],
            example="portrait of a wise old wizard, oil painting style, masterpiece quality, detailed face, dramatic lighting"
        ),
        PromptTemplate(
            name="landscape",
            category="nature",
            template="{location} landscape, {time_of_day}, {style}, {quality}, {weather}",
            description="Beautiful landscape scenes",
            tags=["landscape", "nature", "scenery"],
            example="mountain landscape, golden hour, digital art style, high quality, clear sky"
        ),
        PromptTemplate(
            name="character",
            category="fantasy",
            template="{character_type} character, {pose}, {style}, detailed, {background}",
            description="Character design and concept art",
            tags=["character", "design", "concept"],
            example="fantasy warrior character, dynamic pose, anime style, detailed, mystical forest background"
        ),
        PromptTemplate(
            name="architecture",
            category="buildings",
            template="{building_type}, {architectural_style}, {time_of_day}, {quality}, {atmosphere}",
            description="Architectural visualization",
            tags=["architecture", "building", "structure"],
            example="futuristic skyscraper, modern architectural style, sunset, high quality, vibrant atmosphere"
        ),
        PromptTemplate(
            name="product",
            category="commercial",
            template="{product} on {surface}, {lighting}, {style}, {quality}, clean composition",
            description="Product photography style",
            tags=["product", "commercial", "photography"],
            example="luxury watch on marble surface, studio lighting, photorealistic style, professional quality, clean composition"
        ),
        PromptTemplate(
            name="animal",
            category="nature",
            template="{animal} {action}, {environment}, {style}, {quality}, detailed fur/feathers",
            description="Animal and wildlife images",
            tags=["animal", "wildlife", "nature"],
            example="majestic lion roaring, savanna environment, photorealistic style, high quality, detailed fur"
        ),
        PromptTemplate(
            name="abstract",
            category="artistic",
            template="abstract {concept}, {colors}, {style}, {quality}, {mood}",
            description="Abstract and artistic compositions",
            tags=["abstract", "artistic", "creative"],
            example="abstract technology concept, vibrant neon colors, digital art style, masterpiece quality, futuristic mood"
        ),
        PromptTemplate(
            name="food",
            category="commercial",
            template="{dish}, {presentation}, {style}, {quality}, {lighting}, appetizing",
            description="Food photography and styling",
            tags=["food", "culinary", "photography"],
            example="gourmet pasta dish, elegant presentation, photorealistic style, professional quality, natural lighting, appetizing"
        ),
    ]

    STYLE_MODIFIERS = {
        PromptStyle.PHOTOREALISTIC: "photorealistic, highly detailed, 8k uhd, dslr, soft lighting, high quality",
        PromptStyle.ARTISTIC: "artistic, painterly, creative composition, expressive",
        PromptStyle.ANIME: "anime style, manga illustration, vibrant colors, clean linework",
        PromptStyle.DIGITAL_ART: "digital art, concept art, trending on artstation, highly detailed",
        PromptStyle.OIL_PAINTING: "oil painting, traditional art, brush strokes, rich colors",
        PromptStyle.WATERCOLOR: "watercolor painting, soft colors, flowing, artistic",
        PromptStyle.SKETCH: "pencil sketch, line art, hand-drawn, artistic study",
        PromptStyle.CYBERPUNK: "cyberpunk style, neon lights, futuristic, dystopian",
        PromptStyle.FANTASY: "fantasy art, magical, ethereal, imaginative",
        PromptStyle.MINIMALIST: "minimalist, simple, clean, modern aesthetic",
    }

    QUALITY_MODIFIERS = {
        PromptQuality.STANDARD: "",
        PromptQuality.HIGH_QUALITY: "high quality, detailed, well-composed",
        PromptQuality.MASTERPIECE: "masterpiece, best quality, highly detailed, award-winning",
        PromptQuality.PROFESSIONAL: "professional, studio quality, expertly crafted, polished",
    }

    LIGHTING_TERMS = [
        "natural lighting", "studio lighting", "golden hour",
        "dramatic lighting", "soft lighting", "rim lighting",
        "ambient light", "backlit", "cinematic lighting",
    ]

    NEGATIVE_PROMPT_DEFAULTS = [
        "blurry", "low quality", "distorted", "deformed",
        "bad anatomy", "poorly drawn", "ugly", "duplicate",
        "watermark", "signature", "text", "cropped",
    ]

    @classmethod
    def get_template(cls, name: str) -> Optional[PromptTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate if found, None otherwise
        """
        for template in cls.TEMPLATES:
            if template.name == name:
                return template
        return None

    @classmethod
    def get_templates_by_category(cls, category: str) -> List[PromptTemplate]:
        """Get all templates in a category.

        Args:
            category: Category name

        Returns:
            List of templates in the category
        """
        return [t for t in cls.TEMPLATES if t.category == category]

    @classmethod
    def search_templates(cls, query: str) -> List[PromptTemplate]:
        """Search templates by name, tags, or description.

        Args:
            query: Search query

        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        results = []

        for template in cls.TEMPLATES:
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)

        return results

    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all available categories.

        Returns:
            List of category names
        """
        return list(set(t.category for t in cls.TEMPLATES))


class PromptEnhancer:
    """Enhances prompts for better image generation results."""

    def __init__(self):
        """Initialize the prompt enhancer."""
        logger.info("PromptEnhancer initialized")

    def enhance_prompt(
        self,
        prompt: str,
        style: Optional[PromptStyle] = None,
        quality: Optional[PromptQuality] = None,
        add_details: bool = True
    ) -> str:
        """Enhance a prompt with style and quality modifiers.

        Args:
            prompt: Original prompt
            style: Desired style
            quality: Desired quality level
            add_details: Whether to add detail enhancers

        Returns:
            Enhanced prompt
        """
        enhanced = prompt.strip()

        # Add style modifiers
        if style and style in PromptLibrary.STYLE_MODIFIERS:
            style_text = PromptLibrary.STYLE_MODIFIERS[style]
            enhanced = f"{enhanced}, {style_text}"

        # Add quality modifiers
        if quality and quality in PromptLibrary.QUALITY_MODIFIERS:
            quality_text = PromptLibrary.QUALITY_MODIFIERS[quality]
            if quality_text:
                enhanced = f"{enhanced}, {quality_text}"

        # Add detail enhancers
        if add_details:
            enhanced = self._add_detail_enhancers(enhanced)

        # Clean up the prompt
        enhanced = self._clean_prompt(enhanced)

        logger.debug(f"Enhanced prompt: '{prompt}' -> '{enhanced}'")
        return enhanced

    def _add_detail_enhancers(self, prompt: str) -> str:
        """Add detail-enhancing keywords.

        Args:
            prompt: Prompt to enhance

        Returns:
            Prompt with added details
        """
        # Check if detail keywords already present
        detail_keywords = ["detailed", "intricate", "sharp focus"]
        has_details = any(keyword in prompt.lower() for keyword in detail_keywords)

        if not has_details:
            prompt = f"{prompt}, detailed, sharp focus"

        return prompt

    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt text.

        Args:
            prompt: Prompt to clean

        Returns:
            Cleaned prompt
        """
        # Remove extra spaces
        prompt = re.sub(r'\s+', ' ', prompt)

        # Remove duplicate commas
        prompt = re.sub(r',\s*,', ',', prompt)

        # Remove leading/trailing commas
        prompt = prompt.strip(', ')

        # Ensure single space after commas
        prompt = re.sub(r',\s*', ', ', prompt)

        return prompt

    def generate_negative_prompt(
        self,
        custom_negatives: Optional[List[str]] = None,
        include_defaults: bool = True
    ) -> str:
        """Generate a negative prompt to avoid unwanted elements.

        Args:
            custom_negatives: Custom negative terms to add
            include_defaults: Whether to include default negative terms

        Returns:
            Negative prompt string
        """
        negatives = []

        if include_defaults:
            negatives.extend(PromptLibrary.NEGATIVE_PROMPT_DEFAULTS)

        if custom_negatives:
            negatives.extend(custom_negatives)

        # Remove duplicates while preserving order
        seen = set()
        unique_negatives = []
        for neg in negatives:
            if neg.lower() not in seen:
                seen.add(neg.lower())
                unique_negatives.append(neg)

        return ", ".join(unique_negatives)

    def suggest_improvements(self, prompt: str) -> Dict[str, any]:
        """Suggest improvements for a prompt.

        Args:
            prompt: Original prompt

        Returns:
            Dictionary with suggestions
        """
        suggestions = {
            "original": prompt,
            "issues": [],
            "recommendations": [],
            "enhanced_examples": []
        }

        # Check for common issues
        if len(prompt.split()) < 3:
            suggestions["issues"].append("Prompt is very short - consider adding more details")
            suggestions["recommendations"].append("Add descriptive details about style, quality, or composition")

        if not any(style in prompt.lower() for style in ["photo", "art", "painting", "digital", "anime"]):
            suggestions["issues"].append("No clear style specified")
            suggestions["recommendations"].append("Add a style modifier (photorealistic, digital art, anime, etc.)")

        if "high quality" not in prompt.lower() and "detailed" not in prompt.lower():
            suggestions["issues"].append("No quality modifiers present")
            suggestions["recommendations"].append("Consider adding quality enhancers like 'high quality' or 'detailed'")

        # Generate example enhancements
        suggestions["enhanced_examples"].append(
            self.enhance_prompt(prompt, style=PromptStyle.PHOTOREALISTIC, quality=PromptQuality.HIGH_QUALITY)
        )
        suggestions["enhanced_examples"].append(
            self.enhance_prompt(prompt, style=PromptStyle.DIGITAL_ART, quality=PromptQuality.MASTERPIECE)
        )

        return suggestions

    def __repr__(self) -> str:
        """String representation."""
        return "PromptEnhancer(templates={}, styles={})".format(
            len(PromptLibrary.TEMPLATES),
            len(PromptLibrary.STYLE_MODIFIERS)
        )


# Global prompt enhancer instance
_global_enhancer: Optional[PromptEnhancer] = None


def get_prompt_enhancer() -> PromptEnhancer:
    """Get or create the global prompt enhancer instance.

    Returns:
        Global PromptEnhancer instance
    """
    global _global_enhancer

    if _global_enhancer is None:
        _global_enhancer = PromptEnhancer()

    return _global_enhancer


def reset_prompt_enhancer() -> None:
    """Reset the global prompt enhancer instance (useful for testing)."""
    global _global_enhancer
    _global_enhancer = None
