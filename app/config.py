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

    # Application Settings
    default_backend: str = "huggingface"
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: int = 60

    # Testing
    run_integration_tests: bool = False

    def validate_required_keys(self) -> None:
        """Validate that required API keys are present.

        Raises:
            ValueError: If required API keys are missing
        """
        if not self.huggingface_token and self.default_backend == "huggingface":
            raise ValueError(
                "HUGGINGFACE_TOKEN is required when using the HuggingFace backend. "
                "Please set it in your .env file or environment variables. "
                "Get your token from: https://huggingface.co/settings/tokens"
            )


# Global settings instance
settings = Settings()
