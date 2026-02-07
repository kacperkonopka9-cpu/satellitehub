"""Tests for the provider registry get_provider() function."""

from __future__ import annotations

import pytest

from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError
from satellitehub.providers import get_provider
from satellitehub.providers.base import DataProvider
from satellitehub.providers.cds import CDSProvider
from satellitehub.providers.cdse import CDSEProvider
from satellitehub.providers.imgw import IMGWProvider
from satellitehub.providers.landsat import LandsatProvider


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level config before each test."""
    import satellitehub.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", Config())


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset provider registry before each test."""
    import satellitehub.providers as _prov

    monkeypatch.setattr(_prov, "_REGISTRY_INITIALIZED", False)
    monkeypatch.setattr(_prov, "_PROVIDER_REGISTRY", {})


# ── Registry returns correct provider types ─────────────────────────


@pytest.mark.unit
class TestRegistryReturns:
    """Verify get_provider() returns correct provider instances."""

    def test_cdse_returns_cdse_provider(self) -> None:
        provider = get_provider("cdse", Config())
        assert isinstance(provider, CDSEProvider)

    def test_cds_returns_cds_provider(self) -> None:
        provider = get_provider("cds", Config())
        assert isinstance(provider, CDSProvider)

    def test_imgw_returns_imgw_provider(self) -> None:
        provider = get_provider("imgw", Config())
        assert isinstance(provider, IMGWProvider)

    def test_landsat_returns_landsat_provider(self) -> None:
        provider = get_provider("landsat", Config())
        assert isinstance(provider, LandsatProvider)

    def test_returns_data_provider_instance(self) -> None:
        provider = get_provider("cdse", Config())
        assert isinstance(provider, DataProvider)


# ── Case-insensitive lookup ──────────────────────────────────────────


@pytest.mark.unit
class TestCaseInsensitive:
    """Verify provider lookup is case-insensitive."""

    def test_uppercase_cdse(self) -> None:
        provider = get_provider("CDSE", Config())
        assert isinstance(provider, CDSEProvider)

    def test_mixed_case_cdse(self) -> None:
        provider = get_provider("Cdse", Config())
        assert isinstance(provider, CDSEProvider)

    def test_uppercase_cds(self) -> None:
        provider = get_provider("CDS", Config())
        assert isinstance(provider, CDSProvider)

    def test_uppercase_imgw(self) -> None:
        provider = get_provider("IMGW", Config())
        assert isinstance(provider, IMGWProvider)

    def test_uppercase_landsat(self) -> None:
        provider = get_provider("LANDSAT", Config())
        assert isinstance(provider, LandsatProvider)


# ── Unknown provider ─────────────────────────────────────────────────


@pytest.mark.unit
class TestUnknownProvider:
    """Verify unknown provider names raise ConfigurationError."""

    def test_unknown_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            get_provider("nonexistent", Config())

    def test_error_mentions_valid_providers(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            get_provider("nonexistent", Config())
        msg = str(exc_info.value)
        assert "cds" in msg
        assert "cdse" in msg
        assert "imgw" in msg
        assert "landsat" in msg

    def test_error_includes_unknown_name(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            get_provider("nonexistent", Config())
        assert "nonexistent" in str(exc_info.value)

    def test_error_has_three_part_attributes(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            get_provider("nonexistent", Config())
        err = exc_info.value
        assert err.what != ""
        assert err.cause != ""
        assert err.fix != ""


# ── Provider config and name ─────────────────────────────────────────


@pytest.mark.unit
class TestProviderConfigAndName:
    """Verify returned provider has correct config reference and name."""

    def test_provider_has_correct_config(self) -> None:
        cfg = Config(cache_size_mb=4321)
        provider = get_provider("cdse", cfg)
        assert provider._config is cfg

    def test_provider_has_correct_name(self) -> None:
        provider = get_provider("cdse", Config())
        assert provider.name == "cdse"

    def test_cds_provider_has_correct_name(self) -> None:
        provider = get_provider("cds", Config())
        assert provider.name == "cds"

    def test_imgw_provider_has_correct_name(self) -> None:
        provider = get_provider("imgw", Config())
        assert provider.name == "imgw"

    def test_landsat_provider_has_correct_name(self) -> None:
        provider = get_provider("landsat", Config())
        assert provider.name == "landsat"

    def test_each_call_returns_new_instance(self) -> None:
        cfg = Config()
        p1 = get_provider("cdse", cfg)
        p2 = get_provider("cdse", cfg)
        assert p1 is not p2
