"""Application configuration management."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    These settings are automatically loaded from the .env file or environment variables.
    All sensitive data (API keys) should be stored in environment variables, not hardcoded.

    Attributes:
        huggingface_token: HuggingFace API token for inference
        replicate_token: Replicate API token (optional, for Stage 2)
        default_backend: Which backend to use by default
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        max_retries: Maximum number of retries for API calls
        timeout: Request timeout in seconds
        run_integration_tests: Whether to run integration tests
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # API Keys
    huggingface_token: str = ""
    replicate_token: Optional[str] = None

    # Model Configuration
    huggingface_model: str = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL supports img2img
    replicate_model: str = "black-forest-labs/flux-schnell"  # FLUX supports both

    # Application Settings
    default_backend: str = "huggingface"
    fallback_backend: Optional[str] = None
    enable_fallback: bool = False
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: int = 60

    # Local Backend Settings (Stage 4)
    local_model_cache_dir: Optional[str] = None  # Defaults to ~/.cache/huggingface
    local_model: str = "stabilityai/sd-turbo"     # Default local model

    # Production Settings (Stage 6)
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100  # Max requests per window
    rate_limit_window: int = 60  # Window size in seconds
    enable_health_checks: bool = True
    enable_metrics: bool = True
    production_mode: bool = False  # Enables strict production behaviors

    # Testing
    run_integration_tests: bool = False

    def validate_required_keys(self) -> None:
        """Validate that required API keys are present.

        Raises:
            ValueError: If required API keys are missing
        """
        # Local backend doesn't require API keys
        if self.default_backend == "local":
            return

        if not self.huggingface_token and self.default_backend == "huggingface":
            raise ValueError(
                "HUGGINGFACE_TOKEN is required when using the HuggingFace backend. "
                "Please set it in your .env file or environment variables. "
                "Get your token from: https://huggingface.co/settings/tokens"
            )

        if not self.replicate_token and self.default_backend == "replicate":
            raise ValueError(
                "REPLICATE_TOKEN is required when using the Replicate backend. "
                "Please set it in your .env file or environment variables. "
                "Get your token from: https://replicate.com/account/api-tokens"
            )

        # Validate fallback backend has token (local doesn't need token)
        if self.enable_fallback and self.fallback_backend:
            if self.fallback_backend == "replicate" and not self.replicate_token:
                raise ValueError(
                    "REPLICATE_TOKEN is required when using Replicate as fallback backend."
                )
            elif self.fallback_backend == "huggingface" and not self.huggingface_token:
                raise ValueError(
                    "HUGGINGFACE_TOKEN is required when using HuggingFace as fallback backend."
                )


# Global settings instance
settings = Settings()
