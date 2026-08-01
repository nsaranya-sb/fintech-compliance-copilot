"""Unit tests for configuration settings."""

import os
from unittest.mock import patch
from src.config import Settings, get_settings


def test_show_debug_sidebar_default_false():
    """Verify SHOW_DEBUG_SIDEBAR defaults to False."""
    settings = Settings()
    assert settings.SHOW_DEBUG_SIDEBAR is False


def test_show_debug_sidebar_env_override():
    """Verify SHOW_DEBUG_SIDEBAR can be enabled via environment variable."""
    with patch.dict(os.environ, {"SHOW_DEBUG_SIDEBAR": "true"}):
        settings = get_settings()
        assert settings.SHOW_DEBUG_SIDEBAR is True
