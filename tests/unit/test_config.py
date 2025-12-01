"""Unit tests for configuration management."""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        from app.config import Settings

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test_token"}, clear=True):
            settings = Settings(_env_file=None)

            assert settings.huggingface_token == "test_token"
            assert settings.replicate_token is None
            assert settings.default_backend == "huggingface"
            assert settings.log_level == "INFO"
            assert settings.max_retries == 3
            assert settings.timeout == 60
            assert settings.run_integration_tests is False

    def test_custom_values(self):
        """Test that custom values can be set via environment variables."""
        from app.config import Settings

        env_vars = {
            "HUGGINGFACE_TOKEN": "hf_custom_token",
            "REPLICATE_TOKEN": "r8_custom_token",
            "DEFAULT_BACKEND": "replicate",
            "LOG_LEVEL": "DEBUG",
            "MAX_RETRIES": "5",
            "TIMEOUT": "120",
            "RUN_INTEGRATION_TESTS": "true"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

            assert settings.huggingface_token == "hf_custom_token"
            assert settings.replicate_token == "r8_custom_token"
            assert settings.default_backend == "replicate"
            assert settings.log_level == "DEBUG"
            assert settings.max_retries == 5
            assert settings.timeout == 120
            assert settings.run_integration_tests is True

    def test_validate_required_keys_success(self):
        """Test validation succeeds when required keys are present."""
        from app.config import Settings

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_valid_token"}, clear=True):
            settings = Settings(_env_file=None)
            settings.validate_required_keys()  # Should not raise

    def test_validate_required_keys_missing_huggingface(self):
        """Test validation fails when HuggingFace token is missing."""
        from app.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
                settings.validate_required_keys()

    def test_validate_required_keys_empty_huggingface(self):
        """Test validation fails when HuggingFace token is empty."""
        from app.config import Settings

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": ""}, clear=True):
            settings = Settings(_env_file=None)

            with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
                settings.validate_required_keys()

    def test_integer_parsing(self):
        """Test that integer values are parsed correctly."""
        from app.config import Settings

        with patch.dict(os.environ, {
            "HUGGINGFACE_TOKEN": "test",
            "MAX_RETRIES": "10",
            "TIMEOUT": "300"
        }, clear=True):
            settings = Settings(_env_file=None)

            assert settings.max_retries == 10
            assert settings.timeout == 300

    def test_boolean_true_values(self):
        """Test that boolean true values are parsed correctly."""
        from app.config import Settings

        true_values = ["true", "True", "TRUE", "1"]

        for value in true_values:
            with patch.dict(os.environ, {
                "HUGGINGFACE_TOKEN": "test",
                "RUN_INTEGRATION_TESTS": value
            }, clear=True):
                settings = Settings(_env_file=None)
                assert settings.run_integration_tests is True, \
                    f"Failed for value: {value}"

    def test_boolean_false_values(self):
        """Test that boolean false values are parsed correctly."""
        from app.config import Settings

        false_values = ["false", "False", "FALSE", "0"]

        for value in false_values:
            with patch.dict(os.environ, {
                "HUGGINGFACE_TOKEN": "test",
                "RUN_INTEGRATION_TESTS": value
            }, clear=True):
                settings = Settings(_env_file=None)
                assert settings.run_integration_tests is False, \
                    f"Failed for value: {value}"

    def test_boolean_default_false(self):
        """Test that boolean defaults to false when not set."""
        from app.config import Settings

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test"}, clear=True):
            settings = Settings(_env_file=None)
            assert settings.run_integration_tests is False

    def test_optional_fields(self):
        """Test that optional fields can be None."""
        from app.config import Settings

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test"}, clear=True):
            settings = Settings(_env_file=None)

            assert settings.replicate_token is None

    def test_settings_model_config(self):
        """Test that Settings has correct model configuration."""
        from app.config import Settings

        # Check that the model config exists
        assert hasattr(Settings, 'model_config')

        # Verify case sensitivity setting
        config = Settings.model_config
        assert config.get('case_sensitive') is False
        assert config.get('extra') == 'ignore'
