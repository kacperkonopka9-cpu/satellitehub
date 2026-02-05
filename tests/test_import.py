"""Test that SatelliteHub package imports correctly."""

import re

import pytest

import satellitehub


@pytest.mark.unit
def test_package_version_exists() -> None:
    """Verify package exposes a valid semver version string."""
    assert hasattr(satellitehub, "__version__")
    assert isinstance(satellitehub.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", satellitehub.__version__)


@pytest.mark.unit
def test_public_api_exports() -> None:
    """Verify public API symbols are accessible."""
    assert hasattr(satellitehub, "Config")
    assert hasattr(satellitehub, "configure")
    assert hasattr(satellitehub, "SatelliteHubError")
    assert hasattr(satellitehub, "ConfigurationError")
    assert hasattr(satellitehub, "ProviderError")
    assert hasattr(satellitehub, "CacheError")


@pytest.mark.unit
def test_exception_hierarchy() -> None:
    """Verify exception inheritance chain."""
    assert issubclass(satellitehub.ConfigurationError, satellitehub.SatelliteHubError)
    assert issubclass(satellitehub.ProviderError, satellitehub.SatelliteHubError)
    assert issubclass(satellitehub.CacheError, satellitehub.SatelliteHubError)
    assert issubclass(satellitehub.SatelliteHubError, Exception)
