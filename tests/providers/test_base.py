"""Tests for DataProvider ABC and provider-domain types."""

from __future__ import annotations

from typing import Any

import pytest

from satellitehub._types import TimeRange
from satellitehub.config import Config
from satellitehub.providers.base import (
    CatalogEntry,
    DataProvider,
    ProviderCredentials,
    ProviderStatus,
)
from satellitehub.providers.cds import CDSProvider
from satellitehub.providers.cdse import CDSEProvider
from satellitehub.providers.imgw import IMGWProvider
from satellitehub.providers.landsat import LandsatProvider


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level config before each test."""
    import satellitehub.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", Config())


# ── DataProvider ABC enforcement ─────────────────────────────────────


@pytest.mark.unit
class TestDataProviderABC:
    """Verify DataProvider cannot be instantiated without full implementation."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            DataProvider(config=Config())  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self) -> None:
        class Incomplete(DataProvider):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete(config=Config())  # type: ignore[abstract]

    def test_missing_one_method_raises(self) -> None:
        class MissingDownload(DataProvider):
            _name = "test"

            def authenticate(self, credentials: ProviderCredentials) -> None:
                pass

            def search(
                self,
                location: Any,
                time_range: TimeRange,
                **params: Any,
            ) -> list[CatalogEntry]:
                return []

            # download() missing

            def check_status(self) -> ProviderStatus:
                return ProviderStatus()

        with pytest.raises(TypeError, match="abstract"):
            MissingDownload(config=Config())  # type: ignore[abstract]


# ── ProviderCredentials ──────────────────────────────────────────────


@pytest.mark.unit
class TestProviderCredentials:
    """Verify ProviderCredentials is frozen and has expected fields."""

    def test_default_fields(self) -> None:
        creds = ProviderCredentials()
        assert creds.username == ""
        assert creds.password == ""
        assert creds.api_key == ""

    def test_custom_fields(self) -> None:
        creds = ProviderCredentials(username="placeholder", password="placeholder")
        assert creds.username == "placeholder"
        assert creds.password == "placeholder"

    def test_frozen_immutable(self) -> None:
        creds = ProviderCredentials(username="placeholder")
        with pytest.raises(Exception):  # noqa: B017, PT011
            creds.username = "other"  # type: ignore[misc]


# ── CatalogEntry ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestCatalogEntry:
    """Verify CatalogEntry dataclass fields and defaults."""

    def test_default_fields(self) -> None:
        entry = CatalogEntry()
        assert entry.provider == ""
        assert entry.product_id == ""
        assert entry.timestamp == ""
        assert entry.cloud_cover == 0.0
        assert entry.geometry == {}
        assert entry.bands_available == []
        assert entry.metadata == {}

    def test_custom_fields(self) -> None:
        entry = CatalogEntry(
            provider="cdse",
            product_id="S2A_TEST",
            timestamp="2024-01-01T10:30:00Z",
            cloud_cover=0.15,
            geometry={"type": "Point", "coordinates": [0.0, 0.0]},
            bands_available=["B4", "B8"],
            metadata={"key": "value"},
        )
        assert entry.provider == "cdse"
        assert entry.product_id == "S2A_TEST"
        assert entry.cloud_cover == 0.15
        assert entry.bands_available == ["B4", "B8"]

    def test_mutable_defaults_independent(self) -> None:
        entry1 = CatalogEntry()
        entry2 = CatalogEntry()
        entry1.bands_available.append("B4")
        assert entry2.bands_available == []


# ── ProviderStatus ───────────────────────────────────────────────────


@pytest.mark.unit
class TestProviderStatus:
    """Verify ProviderStatus dataclass fields and defaults."""

    def test_default_fields(self) -> None:
        status = ProviderStatus()
        assert status.available is False
        assert status.message == ""
        assert status.last_checked == ""

    def test_custom_fields(self) -> None:
        status = ProviderStatus(
            available=True,
            message="OK",
            last_checked="2024-01-01T00:00:00Z",
        )
        assert status.available is True
        assert status.message == "OK"
        assert status.last_checked == "2024-01-01T00:00:00Z"


# ── Concrete provider stubs ─────────────────────────────────────────


@pytest.mark.unit
class TestConcreteProviders:
    """Verify concrete providers instantiate and implement the ABC."""

    def test_cdse_instantiates(self) -> None:
        provider = CDSEProvider(config=Config())
        assert isinstance(provider, DataProvider)

    def test_cds_instantiates(self) -> None:
        provider = CDSProvider(config=Config())
        assert isinstance(provider, DataProvider)

    def test_imgw_instantiates(self) -> None:
        provider = IMGWProvider(config=Config())
        assert isinstance(provider, DataProvider)

    def test_cdse_has_name(self) -> None:
        provider = CDSEProvider(config=Config())
        assert provider.name == "cdse"

    def test_cds_has_name(self) -> None:
        provider = CDSProvider(config=Config())
        assert provider.name == "cds"

    def test_imgw_has_name(self) -> None:
        provider = IMGWProvider(config=Config())
        assert provider.name == "imgw"

    def test_cdse_stores_config(self) -> None:
        cfg = Config()
        provider = CDSEProvider(config=cfg)
        assert provider._config is cfg

    def test_cds_stores_config(self) -> None:
        cfg = Config()
        provider = CDSProvider(config=cfg)
        assert provider._config is cfg

    def test_imgw_stores_config(self) -> None:
        cfg = Config()
        provider = IMGWProvider(config=cfg)
        assert provider._config is cfg

    def test_cdse_session_initialized(self) -> None:
        """CDSE creates a session in __init__ for vcrpy interceptability."""
        provider = CDSEProvider(config=Config())
        assert provider._session is not None

    def test_cdse_implements_all_methods(self) -> None:
        """CDSEProvider doesn't raise TypeError — all abstract methods implemented."""
        provider = CDSEProvider(config=Config())
        assert hasattr(provider, "authenticate")
        assert hasattr(provider, "search")
        assert hasattr(provider, "download")
        assert hasattr(provider, "check_status")

    def test_cds_implements_all_methods(self) -> None:
        provider = CDSProvider(config=Config())
        assert hasattr(provider, "authenticate")
        assert hasattr(provider, "search")
        assert hasattr(provider, "download")
        assert hasattr(provider, "check_status")

    def test_imgw_implements_all_methods(self) -> None:
        provider = IMGWProvider(config=Config())
        assert hasattr(provider, "authenticate")
        assert hasattr(provider, "search")
        assert hasattr(provider, "download")
        assert hasattr(provider, "check_status")

    def test_landsat_instantiates(self) -> None:
        provider = LandsatProvider(config=Config())
        assert isinstance(provider, DataProvider)

    def test_landsat_has_name(self) -> None:
        provider = LandsatProvider(config=Config())
        assert provider.name == "landsat"

    def test_landsat_stores_config(self) -> None:
        cfg = Config()
        provider = LandsatProvider(config=cfg)
        assert provider._config is cfg

    def test_landsat_implements_all_methods(self) -> None:
        provider = LandsatProvider(config=Config())
        assert hasattr(provider, "authenticate")
        assert hasattr(provider, "search")
        assert hasattr(provider, "download")
        assert hasattr(provider, "check_status")
