"""Tests for Location model, location() factory, UTM zone, and location_hash."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from satellitehub._types import RawData
from satellitehub.config import Config, configure, get_default_config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.location import (
    DataTier,
    Location,
    _compute_utm_epsg,
    _compute_utm_zone,
    location,
)
from satellitehub.providers.base import CatalogEntry, DataProvider
from satellitehub.providers.cds import CDSProvider
from satellitehub.providers.cdse import CDSEProvider
from satellitehub.providers.imgw import IMGWProvider
from satellitehub.results import BaseResult, ChangeResult, VegetationResult


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


# ── Location factory: valid coordinates ──────────────────────────────


@pytest.mark.unit
class TestLocationValid:
    """Verify location() creates Location with correct properties."""

    def test_valid_coordinates(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        assert loc.lat == 51.25
        assert loc.lon == 22.57

    @pytest.mark.parametrize("lat", [-90.0, 0.0, 90.0])
    def test_latitude_boundaries_accepted(self, lat: float) -> None:
        loc = location(lat=lat, lon=0.0)
        assert loc.lat == lat

    @pytest.mark.parametrize("lon", [-180.0, 0.0, 180.0])
    def test_longitude_boundaries_accepted(self, lon: float) -> None:
        loc = location(lat=0.0, lon=lon)
        assert loc.lon == lon

    def test_returns_location_instance(self) -> None:
        loc = location(lat=0.0, lon=0.0)
        assert isinstance(loc, Location)


# ── Location factory: invalid coordinates ────────────────────────────


@pytest.mark.unit
class TestLocationInvalid:
    """Verify location() raises ConfigurationError for invalid coords."""

    def test_lat_too_high(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid latitude"):
            location(lat=91, lon=0.0)

    def test_lat_too_low(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid latitude"):
            location(lat=-91, lon=0.0)

    def test_lon_too_high(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid longitude"):
            location(lat=0.0, lon=181)

    def test_lon_too_low(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid longitude"):
            location(lat=0.0, lon=-181)

    def test_error_includes_invalid_value(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            location(lat=999, lon=0.0)
        assert "999" in str(exc_info.value)

    def test_error_has_three_part_attributes(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            location(lat=91, lon=0.0)
        err = exc_info.value
        assert err.what != ""
        assert err.cause != ""
        assert err.fix != ""


# ── Config snapshot (AD-1) ───────────────────────────────────────────


@pytest.mark.unit
class TestConfigSnapshot:
    """Verify AD-1 configuration injection pattern."""

    def test_default_config_captured(self) -> None:
        loc = location(lat=0.0, lon=0.0)
        default = get_default_config()
        assert loc.config.cache_size_mb == default.cache_size_mb

    def test_explicit_config_overrides_default(self) -> None:
        explicit = Config(cache_size_mb=1234)
        loc = location(lat=0.0, lon=0.0, config=explicit)
        assert loc.config.cache_size_mb == 1234

    def test_subsequent_configure_does_not_affect_location(self) -> None:
        configure(cache_size_mb=1234)
        loc = location(lat=51.25, lon=22.57)
        configure(cache_size_mb=9999)
        assert loc.config.cache_size_mb == 1234

    def test_location_captures_snapshot_identity(self) -> None:
        cfg = Config(cache_size_mb=7777)
        loc = location(lat=0.0, lon=0.0, config=cfg)
        assert loc.config is cfg

    def test_two_locations_independent_configs(self) -> None:
        cfg1 = Config(cache_size_mb=1000)
        cfg2 = Config(cache_size_mb=2000)
        loc1 = location(lat=0.0, lon=0.0, config=cfg1)
        loc2 = location(lat=0.0, lon=0.0, config=cfg2)
        assert loc1.config.cache_size_mb == 1000
        assert loc2.config.cache_size_mb == 2000


# ── UTM zone determination ───────────────────────────────────────────


@pytest.mark.unit
class TestUtmZone:
    """Verify UTM zone computation with known reference points."""

    @pytest.mark.parametrize(
        ("lat", "lon", "expected_zone"),
        [
            (51.25, 22.57, 34),  # Warsaw
            (40.71, -74.01, 18),  # New York
            (-33.87, 151.21, 56),  # Sydney
            (51.5, 0.0, 31),  # Prime Meridian
            (35.68, 139.69, 54),  # Tokyo
            (0.0, 0.0, 31),  # Equator / Prime Meridian
        ],
    )
    def test_standard_zones(self, lat: float, lon: float, expected_zone: int) -> None:
        assert _compute_utm_zone(lat, lon) == expected_zone

    def test_norway_exception(self) -> None:
        # Southwest Norway: lon=5 should be zone 32 (widened), not zone 31
        assert _compute_utm_zone(60.0, 5.0) == 32

    def test_svalbard_exception_zone_33(self) -> None:
        assert _compute_utm_zone(75.0, 15.0) == 33

    def test_svalbard_exception_zone_35(self) -> None:
        assert _compute_utm_zone(75.0, 25.0) == 35

    def test_svalbard_exception_zone_37(self) -> None:
        assert _compute_utm_zone(75.0, 35.0) == 37

    def test_svalbard_exception_zone_31(self) -> None:
        assert _compute_utm_zone(75.0, 5.0) == 31

    def test_lon_180_clamped_to_zone_60(self) -> None:
        assert _compute_utm_zone(0.0, 180.0) == 60

    def test_lon_minus_180_is_zone_1(self) -> None:
        assert _compute_utm_zone(0.0, -180.0) == 1


@pytest.mark.unit
class TestUtmEpsg:
    """Verify EPSG code construction from UTM zone and hemisphere."""

    def test_northern_hemisphere(self) -> None:
        assert _compute_utm_epsg(51.25, 22.57) == 32634

    def test_southern_hemisphere(self) -> None:
        assert _compute_utm_epsg(-33.87, 151.21) == 32756

    def test_equator_north(self) -> None:
        assert _compute_utm_epsg(0.0, 0.0) == 32631

    def test_location_exposes_utm(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        assert loc.utm_zone == 34
        assert loc.utm_epsg == 32634


# ── Location hash ────────────────────────────────────────────────────


@pytest.mark.unit
class TestLocationHash:
    """Verify deterministic SHA-256 hash for cache keys."""

    def test_deterministic(self) -> None:
        loc1 = location(lat=51.25, lon=22.57)
        loc2 = location(lat=51.25, lon=22.57)
        assert loc1.location_hash == loc2.location_hash

    def test_different_coords_different_hash(self) -> None:
        loc1 = location(lat=51.25, lon=22.57)
        loc2 = location(lat=52.0, lon=21.0)
        assert loc1.location_hash != loc2.location_hash

    def test_hash_is_hex_string(self) -> None:
        loc = location(lat=0.0, lon=0.0)
        assert isinstance(loc.location_hash, str)
        assert len(loc.location_hash) == 64  # SHA-256 hex digest


# ── Equality and hashing ─────────────────────────────────────────────


@pytest.mark.unit
class TestEquality:
    """Verify __eq__ and __hash__ for Location objects."""

    def test_same_coords_are_equal(self) -> None:
        loc1 = location(lat=51.25, lon=22.57)
        loc2 = location(lat=51.25, lon=22.57)
        assert loc1 == loc2

    def test_different_coords_not_equal(self) -> None:
        loc1 = location(lat=51.25, lon=22.57)
        loc2 = location(lat=52.0, lon=21.0)
        assert loc1 != loc2

    def test_not_equal_to_non_location(self) -> None:
        loc = location(lat=0.0, lon=0.0)
        assert loc != "not a location"

    def test_hashable_and_usable_as_dict_key(self) -> None:
        loc1 = location(lat=51.25, lon=22.57)
        loc2 = location(lat=51.25, lon=22.57)
        d: dict[Location, str] = {loc1: "warsaw"}
        assert d[loc2] == "warsaw"

    def test_equal_locations_same_hash(self) -> None:
        loc1 = location(lat=51.25, lon=22.57)
        loc2 = location(lat=51.25, lon=22.57)
        assert hash(loc1) == hash(loc2)


# ── Repr ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRepr:
    """Verify Location repr includes key information."""

    def test_repr_includes_lat_lon_zone(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        r = repr(loc)
        assert "51.25" in r
        assert "22.57" in r
        assert "34" in r
        assert "Location(" in r


# ── Import test ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_location_importable_from_package() -> None:
    """Verify AC #4: from satellitehub import location succeeds."""
    import satellitehub

    assert hasattr(satellitehub, "location")
    assert callable(satellitehub.location)


# ── Provider caching on Location ─────────────────────────────────────


@pytest.mark.unit
class TestProviderCaching:
    """Verify provider instances are cached per Location (AC #3)."""

    def test_get_provider_returns_cdse(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        provider = loc.get_provider("cdse")
        assert isinstance(provider, CDSEProvider)

    def test_get_provider_returns_cds(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        provider = loc.get_provider("cds")
        assert isinstance(provider, CDSProvider)

    def test_get_provider_returns_imgw(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        provider = loc.get_provider("imgw")
        assert isinstance(provider, IMGWProvider)

    def test_same_provider_cached(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        p1 = loc.get_provider("cdse")
        p2 = loc.get_provider("cdse")
        assert p1 is p2

    def test_different_providers_different_instances(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        cdse = loc.get_provider("cdse")
        cds = loc.get_provider("cds")
        assert cdse is not cds

    def test_provider_uses_location_config(self) -> None:
        cfg = Config(cache_size_mb=5555)
        loc = location(lat=51.25, lon=22.57, config=cfg)
        provider = loc.get_provider("cdse")
        assert provider._config is cfg

    def test_two_locations_independent_providers(self) -> None:
        cfg1 = Config(cache_size_mb=1000)
        cfg2 = Config(cache_size_mb=2000)
        loc1 = location(lat=51.25, lon=22.57, config=cfg1)
        loc2 = location(lat=52.0, lon=21.0, config=cfg2)
        p1 = loc1.get_provider("cdse")
        p2 = loc2.get_provider("cdse")
        assert p1 is not p2
        assert p1._config is cfg1
        assert p2._config is cfg2

    def test_case_insensitive_caching(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        p1 = loc.get_provider("cdse")
        p2 = loc.get_provider("CDSE")
        p3 = loc.get_provider("Cdse")
        assert p1 is p2
        assert p1 is p3

    def test_unknown_provider_raises(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        with pytest.raises(ConfigurationError):
            loc.get_provider("nonexistent")

    def test_provider_is_data_provider(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        provider = loc.get_provider("cdse")
        assert isinstance(provider, DataProvider)


# ── Data tier access ──────────────────────────────────────────────────


@pytest.mark.unit
class TestDataTierAccess:
    """Tests for Location.data property and DataTier accessor."""

    def test_data_property_returns_data_tier(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        assert isinstance(loc.data, DataTier)

    def test_data_property_cached(self) -> None:
        loc = location(lat=51.25, lon=22.57)
        tier1 = loc.data
        tier2 = loc.data
        assert tier1 is tier2

    def test_sentinel2_returns_base_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = location(lat=51.25, lon=22.57)
        test_data = np.ones((2, 10, 10), dtype=np.float32)
        raw = RawData(
            data=test_data,
            metadata={
                "timestamp": "2024-01-01T10:00:00Z",
                "bands": ["B04", "B08"],
            },
        )

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result = loc.data.sentinel2(bands=["B04", "B08"])
        assert isinstance(result, BaseResult)
        assert result.confidence == 1.0
        assert result.data.shape == (2, 10, 10)
        assert result.metadata.source == "cdse"

    def test_sentinel2_empty_result_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = location(lat=51.25, lon=22.57)
        empty_raw = RawData(data=np.array([], dtype=np.float32), metadata={})

        def fake_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> RawData:
            return empty_raw

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", fake_acquire)

        result = loc.data.sentinel2()
        assert result.confidence == 0.0
        assert result.data.size == 0
        assert len(result.warnings) == 1
        assert "No Sentinel-2 data found" in result.warnings[0]

    def test_sentinel2_provider_error_returns_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = location(lat=51.25, lon=22.57)

        import satellitehub._pipeline as _pl

        def raise_provider_error(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> None:
            raise ProviderError(
                what="CDSE unreachable",
                cause="Timeout",
                fix="Retry",
            )

        monkeypatch.setattr(_pl, "_acquire", raise_provider_error)

        result = loc.data.sentinel2()
        assert result.confidence == 0.0
        assert len(result.warnings) == 1
        assert "CDSE unreachable" in result.warnings[0]

    def test_sentinel2_config_error_returns_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = location(lat=51.25, lon=22.57)

        import satellitehub._pipeline as _pl

        def raise_config_error(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> None:
            raise ConfigurationError(
                what="Missing credentials",
                cause="No file",
                fix="Configure credentials",
            )

        monkeypatch.setattr(_pl, "_acquire", raise_config_error)

        result = loc.data.sentinel2()
        assert result.confidence == 0.0
        assert len(result.warnings) == 1
        assert "Missing credentials" in result.warnings[0]


# ── available_data() ──────────────────────────────────────────────────


@pytest.mark.unit
class TestAvailableData:
    """Tests for Location.available_data() method."""

    def _inject_mock_provider(self, loc: Location, mock_provider: MagicMock) -> None:
        """Pre-populate the Location._providers cache with mock providers."""
        for name in ("cdse", "cds", "imgw"):
            loc._providers[name] = mock_provider

    def test_returns_summary_dict(self) -> None:
        loc = location(lat=51.25, lon=22.57)

        mock_provider = MagicMock()
        mock_provider.search.return_value = [
            CatalogEntry(
                provider="cdse",
                product_id="P1",
                timestamp="2024-01-01T10:00:00Z",
                cloud_cover=0.15,
            ),
            CatalogEntry(
                provider="cdse",
                product_id="P2",
                timestamp="2024-01-05T10:00:00Z",
                cloud_cover=0.25,
            ),
        ]

        self._inject_mock_provider(loc, mock_provider)

        summary = loc.available_data(last_days=30)
        assert "providers" in summary
        assert "total_passes" in summary
        assert "date_range" in summary
        assert "warnings" in summary
        # 3 providers x 2 entries each = 6 total
        assert summary["total_passes"] == 6

    def test_provider_error_adds_warning(self) -> None:
        loc = location(lat=51.25, lon=22.57)

        mock_ok = MagicMock()
        mock_ok.search.return_value = []

        mock_fail = MagicMock()
        mock_fail.search.side_effect = ProviderError(
            what="CDSE down", cause="503", fix="Wait"
        )

        loc._providers["cdse"] = mock_fail
        loc._providers["cds"] = mock_ok
        loc._providers["imgw"] = mock_ok

        summary = loc.available_data(last_days=30)
        assert len(summary["warnings"]) >= 1
        assert "cdse" in summary["warnings"][0].lower()
        assert summary["providers"]["cdse"] == []

    def test_date_range_tuple(self) -> None:
        loc = location(lat=51.25, lon=22.57)

        mock_provider = MagicMock()
        mock_provider.search.return_value = []
        self._inject_mock_provider(loc, mock_provider)

        summary = loc.available_data(last_days=30)
        start, end = summary["date_range"]
        assert isinstance(start, str)
        assert isinstance(end, str)
        assert len(start) == 10  # YYYY-MM-DD
        assert len(end) == 10


# ── vegetation_health() semantic method ─────────────────────────────


@pytest.mark.unit
class TestVegetationHealth:
    """Tests for Location.vegetation_health() semantic method (Story 3.4)."""

    def test_vegetation_health_standard_case_returns_populated_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #1: Full pipeline returns VegetationResult with all fields populated."""
        loc = location(lat=51.25, lon=22.57)

        # Create synthetic raw data: 3 bands (B04, B08, SCL), 5x5 pixels
        # B04 (red) = 0.1, B08 (NIR) = 0.5 → NDVI = (0.5-0.1)/(0.5+0.1) = 0.667
        raw_data = np.zeros((3, 5, 5), dtype=np.float32)
        raw_data[0] = 0.1  # B04 (red)
        raw_data[1] = 0.5  # B08 (NIR)
        raw_data[2] = 4.0  # SCL = 4 (vegetation, cloud-free)

        raw = RawData(
            data=raw_data,
            metadata={
                "timestamp": "2026-01-15T10:00:00Z",
                "bounds": {"minx": 22.0, "miny": 51.0, "maxx": 23.0, "maxy": 52.0},
            },
        )

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result = loc.vegetation_health(last_days=30)

        assert isinstance(result, VegetationResult)
        assert result.confidence > 0
        assert result.mean_ndvi == pytest.approx(0.667, abs=0.01)
        assert result.ndvi_std == pytest.approx(0.0, abs=0.01)
        assert result.observation_count == 1
        assert result.cloud_free_count == 1
        assert result.metadata.source == "cdse"
        assert result.data.shape == (5, 5)

    def test_vegetation_health_no_data_returns_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4: No data returns confidence=0.0 with descriptive warnings."""
        loc = location(lat=51.25, lon=22.57)

        empty_raw = RawData(data=np.array([], dtype=np.float32), metadata={})

        import satellitehub._pipeline as _pl

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> RawData:
            return empty_raw

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_health(last_days=30)

        assert result.confidence == 0.0
        assert result.data.size == 0
        assert np.isnan(result.mean_ndvi)
        assert np.isnan(result.ndvi_std)
        assert result.observation_count == 0
        assert result.cloud_free_count == 0

    def test_vegetation_health_no_data_includes_fallback_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4: Warnings include FR14 fallback suggestion."""
        loc = location(lat=51.25, lon=22.57)

        empty_raw = RawData(data=np.array([], dtype=np.float32), metadata={})

        import satellitehub._pipeline as _pl

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> RawData:
            return empty_raw

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_health(last_days=30)

        # Check for FR14 fallback suggestion
        fallback_found = any(
            "field.data.sentinel2()" in w and "relaxed cloud_max" in w
            for w in result.warnings
        )
        assert fallback_found, f"FR14 fallback not found in warnings: {result.warnings}"

    def test_vegetation_health_all_cloudy_returns_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4: All-cloudy data returns confidence=0.0."""
        loc = location(lat=51.25, lon=22.57)

        # Create raw data where all pixels are cloudy (SCL=9 = cloud high)
        raw_data = np.zeros((3, 5, 5), dtype=np.float32)
        raw_data[0] = 0.1  # B04
        raw_data[1] = 0.5  # B08
        raw_data[2] = 9.0  # SCL = 9 (cloud high)

        raw = RawData(data=raw_data, metadata={"timestamp": "2026-01-15T10:00:00Z"})

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result = loc.vegetation_health(last_days=30)

        assert result.confidence == 0.0
        assert result.cloud_free_count == 0

    def test_vegetation_health_all_cloudy_includes_fallback_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4: All-cloudy warnings include FR14 fallback suggestion."""
        loc = location(lat=51.25, lon=22.57)

        raw_data = np.zeros((3, 5, 5), dtype=np.float32)
        raw_data[2] = 9.0  # All cloudy

        raw = RawData(data=raw_data, metadata={})

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result = loc.vegetation_health(last_days=30)

        fallback_found = any(
            "field.data.sentinel2()" in w and "relaxed cloud_max" in w
            for w in result.warnings
        )
        assert fallback_found, f"FR14 fallback not found in warnings: {result.warnings}"

    def test_vegetation_health_partial_cloud_returns_proportional_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #5: Partial cloud cover returns result with proportional confidence."""
        loc = location(lat=51.25, lon=22.57)

        # Create raw data with 50% cloudy pixels
        raw_data = np.zeros((3, 4, 4), dtype=np.float32)
        raw_data[0] = 0.1  # B04
        raw_data[1] = 0.5  # B08
        # Half valid (SCL=4), half cloudy (SCL=9)
        raw_data[2, :2, :] = 4.0  # Top half valid
        raw_data[2, 2:, :] = 9.0  # Bottom half cloudy

        raw = RawData(data=raw_data, metadata={"timestamp": "2026-01-15T10:00:00Z"})

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result = loc.vegetation_health(last_days=30)

        # Should have some valid data
        assert result.confidence > 0
        assert result.cloud_free_count == 1
        # Mean NDVI computed from valid pixels
        assert not np.isnan(result.mean_ndvi)

    def test_vegetation_health_reproducibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #6 (NFR20): Same inputs produce bit-identical results."""
        loc = location(lat=51.25, lon=22.57)

        raw_data = np.zeros((3, 5, 5), dtype=np.float32)
        raw_data[0] = 0.1
        raw_data[1] = 0.5
        raw_data[2] = 4.0

        raw = RawData(data=raw_data, metadata={"timestamp": "2026-01-15T10:00:00Z"})

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result1 = loc.vegetation_health(last_days=30)
        result2 = loc.vegetation_health(last_days=30)

        # Bit-identical checks
        assert result1.confidence == result2.confidence
        assert result1.mean_ndvi == result2.mean_ndvi
        assert result1.ndvi_std == result2.ndvi_std
        assert result1.observation_count == result2.observation_count
        assert result1.cloud_free_count == result2.cloud_free_count
        assert np.array_equal(result1.data, result2.data, equal_nan=True)

    def test_vegetation_health_returns_vegetation_result_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify vegetation_health returns VegetationResult, not BaseResult."""
        loc = location(lat=51.25, lon=22.57)

        raw_data = np.zeros((3, 5, 5), dtype=np.float32)
        raw_data[0] = 0.1
        raw_data[1] = 0.5
        raw_data[2] = 4.0

        raw = RawData(data=raw_data, metadata={})

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(
            _pl,
            "_acquire",
            lambda location, provider_name, product, bands, cloud_max, last_days: raw,
        )

        result = loc.vegetation_health(last_days=30)

        assert type(result).__name__ == "VegetationResult"
        # VegetationResult-specific fields exist
        assert hasattr(result, "mean_ndvi")
        assert hasattr(result, "ndvi_std")
        assert hasattr(result, "trend")
        assert hasattr(result, "observation_count")
        assert hasattr(result, "cloud_free_count")

    def test_vegetation_health_provider_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #3: ProviderError from CDSE bubbles up (infrastructure failure)."""
        loc = location(lat=51.25, lon=22.57)

        import satellitehub._pipeline as _pl

        def raise_provider_error(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> None:
            raise ProviderError(
                what="CDSE unreachable",
                cause="Connection timeout after 3 retries",
                fix="Check network connectivity or try again later",
            )

        monkeypatch.setattr(_pl, "_acquire", raise_provider_error)

        with pytest.raises(ProviderError, match="CDSE unreachable"):
            loc.vegetation_health(last_days=30)

    def test_vegetation_health_configuration_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #3: ConfigurationError for missing credentials bubbles up."""
        loc = location(lat=51.25, lon=22.57)

        import satellitehub._pipeline as _pl

        def raise_config_error(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int,
        ) -> None:
            raise ConfigurationError(
                what="Copernicus credentials not configured",
                cause="No credentials file found at configured path",
                fix="Call sh.configure(copernicus_credentials='path/to/creds.json')",
            )

        monkeypatch.setattr(_pl, "_acquire", raise_config_error)

        with pytest.raises(ConfigurationError, match="credentials not configured"):
            loc.vegetation_health(last_days=30)


@pytest.mark.integration
class TestVegetationHealthPerformance:
    """Performance tests for vegetation_health() (AC #2, #3).

    These tests require live CDSE API access and are marked as integration.
    Run with: pytest -m integration --run-integration

    Note: NFR1 (cached <2s) and NFR2 (uncached <2min) performance bounds
    are verified manually during integration testing due to external
    service dependencies.
    """

    @pytest.mark.skip(reason="Manual only - requires live CDSE API")
    def test_vegetation_health_cached_performance_under_2s(self) -> None:
        """AC #2 (NFR1): Cached result returned within 2 seconds.

        Manual test procedure:
        1. Configure valid CDSE credentials
        2. Call vegetation_health() twice for same location/period
        3. Verify second call completes in <2s
        """
        pass

    @pytest.mark.skip(reason="Manual only - requires live CDSE API")
    def test_vegetation_health_uncached_performance_under_2min(self) -> None:
        """AC #3 (NFR2): Fresh download completes within 2 minutes.

        Manual test procedure:
        1. Configure valid CDSE credentials
        2. Clear cache for target location
        3. Call vegetation_health() and measure total time
        4. Verify completion in <2min with INFO logs for >10s operations
        """
        pass


# ── vegetation_change() semantic method ──────────────────────────────


@pytest.mark.unit
class TestVegetationChange:
    """Tests for Location.vegetation_change() semantic method (Story 3.5)."""

    def _make_raw_with_ndvi(
        self, red_val: float, nir_val: float, scl_val: float = 4.0
    ) -> RawData:
        """Create synthetic raw data with specified spectral values.

        Args:
            red_val: Value for B04 (red) band.
            nir_val: Value for B08 (NIR) band.
            scl_val: Value for SCL band (default 4 = vegetation, cloud-free).

        Returns:
            RawData with 3 bands (B04, B08, SCL), 5x5 pixels.
        """
        raw_data = np.zeros((3, 5, 5), dtype=np.float32)
        raw_data[0] = red_val  # B04 (red)
        raw_data[1] = nir_val  # B08 (NIR)
        raw_data[2] = scl_val  # SCL
        return RawData(
            data=raw_data,
            metadata={"timestamp": "2026-01-15T10:00:00Z"},
        )

    def test_vegetation_change_both_periods_valid_returns_delta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #1: Both periods valid returns populated ChangeResult with delta."""
        loc = location(lat=51.25, lon=22.57)

        # Period 1: NDVI = (0.5-0.1)/(0.5+0.1) ≈ 0.667
        raw_p1 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.5)
        # Period 2: NDVI = (0.3-0.1)/(0.3+0.1) = 0.5
        raw_p2 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.3)

        call_count = {"count": 0}

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            call_count["count"] += 1
            return raw_p1 if call_count["count"] == 1 else raw_p2

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert isinstance(result, ChangeResult)
        assert result.period_1_confidence > 0
        assert result.period_2_confidence > 0
        assert result.confidence > 0
        # Expected delta: 0.5 - 0.667 ≈ -0.167 (declining)
        assert result.delta < 0
        assert result.direction == "declining"
        assert result.period_1_range == ("2025-01-01", "2025-01-31")
        assert result.period_2_range == ("2026-01-01", "2026-01-31")

    def test_vegetation_change_period_1_missing_returns_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #3: Period 1 missing data returns confidence=0.0 for that period."""
        loc = location(lat=51.25, lon=22.57)

        empty_raw = RawData(data=np.array([], dtype=np.float32), metadata={})
        raw_p2 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.5)

        call_count = {"count": 0}

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            call_count["count"] += 1
            return empty_raw if call_count["count"] == 1 else raw_p2

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert result.period_1_confidence == 0.0
        assert result.period_2_confidence > 0
        assert result.confidence == 0.0  # min of both
        assert result.direction == "unknown"
        assert any("Period 1" in w for w in result.warnings)

    def test_vegetation_change_period_2_missing_returns_zero_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #3: Period 2 missing data returns confidence=0.0 for that period."""
        loc = location(lat=51.25, lon=22.57)

        raw_p1 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.5)
        empty_raw = RawData(data=np.array([], dtype=np.float32), metadata={})

        call_count = {"count": 0}

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            call_count["count"] += 1
            return raw_p1 if call_count["count"] == 1 else empty_raw

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert result.period_1_confidence > 0
        assert result.period_2_confidence == 0.0
        assert result.confidence == 0.0  # min of both
        assert result.direction == "unknown"
        assert any("Period 2" in w for w in result.warnings)

    def test_vegetation_change_both_missing_includes_fallback_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4: Both periods missing includes FR14 fallback warning."""
        loc = location(lat=51.25, lon=22.57)

        empty_raw = RawData(data=np.array([], dtype=np.float32), metadata={})

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            return empty_raw

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert result.confidence == 0.0
        assert result.direction == "unknown"
        # FR14 fallback warning present
        fallback_found = any(
            "field.data.sentinel2()" in w and "relaxed cloud_max" in w
            for w in result.warnings
        )
        assert fallback_found, f"FR14 fallback not found: {result.warnings}"

    def test_vegetation_change_reproducibility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #5 (NFR20): Same inputs produce bit-identical results."""
        loc = location(lat=51.25, lon=22.57)

        raw_p1 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.5)
        raw_p2 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.3)

        def make_mock_acquire() -> object:
            call_count = {"count": 0}

            def mock_acquire(
                location: object,
                provider_name: str,
                product: str,
                bands: object,
                cloud_max: float,
                last_days: int | None = None,
                time_range: tuple[str, str] | None = None,
            ) -> RawData:
                call_count["count"] += 1
                return raw_p1 if call_count["count"] % 2 == 1 else raw_p2

            return mock_acquire

        import satellitehub._pipeline as _pl

        # First call
        monkeypatch.setattr(_pl, "_acquire", make_mock_acquire())
        result1 = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        # Second call with fresh mock
        monkeypatch.setattr(_pl, "_acquire", make_mock_acquire())
        result2 = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        # Bit-identical checks
        assert result1.confidence == result2.confidence
        assert result1.period_1_ndvi == result2.period_1_ndvi
        assert result1.period_2_ndvi == result2.period_2_ndvi
        assert result1.delta == result2.delta
        assert result1.direction == result2.direction
        assert np.array_equal(result1.data, result2.data, equal_nan=True)

    def test_vegetation_change_declining_direction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test declining vegetation scenario (period_2 NDVI < period_1 NDVI)."""
        loc = location(lat=51.25, lon=22.57)

        # Period 1: healthy vegetation (high NDVI)
        # NDVI = (0.7-0.1)/(0.7+0.1) = 0.75
        raw_p1 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.7)
        # Period 2: degraded vegetation (lower NDVI)
        # NDVI = (0.3-0.1)/(0.3+0.1) = 0.5
        raw_p2 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.3)

        call_count = {"count": 0}

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            call_count["count"] += 1
            return raw_p1 if call_count["count"] == 1 else raw_p2

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert result.direction == "declining"
        assert result.delta < 0

    def test_vegetation_change_improving_direction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test improving vegetation scenario (period_2 NDVI > period_1 NDVI)."""
        loc = location(lat=51.25, lon=22.57)

        # Period 1: sparse vegetation (low NDVI)
        # NDVI = (0.3-0.1)/(0.3+0.1) = 0.5
        raw_p1 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.3)
        # Period 2: healthy vegetation (high NDVI)
        # NDVI = (0.7-0.1)/(0.7+0.1) = 0.75
        raw_p2 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.7)

        call_count = {"count": 0}

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            call_count["count"] += 1
            return raw_p1 if call_count["count"] == 1 else raw_p2

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert result.direction == "improving"
        assert result.delta > 0

    def test_vegetation_change_stable_direction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test stable vegetation scenario (delta within threshold)."""
        loc = location(lat=51.25, lon=22.57)

        # Both periods: nearly identical NDVI (within stable_threshold=0.02)
        # NDVI ≈ 0.667 for both
        raw_p1 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.5)
        raw_p2 = self._make_raw_with_ndvi(red_val=0.1, nir_val=0.5)

        call_count = {"count": 0}

        def mock_acquire(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> RawData:
            call_count["count"] += 1
            return raw_p1 if call_count["count"] == 1 else raw_p2

        import satellitehub._pipeline as _pl

        monkeypatch.setattr(_pl, "_acquire", mock_acquire)

        result = loc.vegetation_change(
            period_1=("2025-01-01", "2025-01-31"),
            period_2=("2026-01-01", "2026-01-31"),
        )

        assert result.direction == "stable"
        assert abs(result.delta) < 0.02  # Within stable threshold

    def test_vegetation_change_provider_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #1 Raises: ProviderError from CDSE bubbles up."""
        loc = location(lat=51.25, lon=22.57)

        import satellitehub._pipeline as _pl

        def raise_provider_error(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> None:
            raise ProviderError(
                what="CDSE unreachable",
                cause="Connection timeout after 3 retries",
                fix="Check network connectivity or try again later",
            )

        monkeypatch.setattr(_pl, "_acquire", raise_provider_error)

        with pytest.raises(ProviderError, match="CDSE unreachable"):
            loc.vegetation_change(
                period_1=("2025-01-01", "2025-01-31"),
                period_2=("2026-01-01", "2026-01-31"),
            )

    def test_vegetation_change_configuration_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #1 Raises: ConfigurationError for missing credentials bubbles up."""
        loc = location(lat=51.25, lon=22.57)

        import satellitehub._pipeline as _pl

        def raise_config_error(
            location: object,
            provider_name: str,
            product: str,
            bands: object,
            cloud_max: float,
            last_days: int | None = None,
            time_range: tuple[str, str] | None = None,
        ) -> None:
            raise ConfigurationError(
                what="Copernicus credentials not configured",
                cause="No credentials file found at configured path",
                fix="Call sh.configure(copernicus_credentials='path/to/creds.json')",
            )

        monkeypatch.setattr(_pl, "_acquire", raise_config_error)

        with pytest.raises(ConfigurationError, match="credentials not configured"):
            loc.vegetation_change(
                period_1=("2025-01-01", "2025-01-31"),
                period_2=("2026-01-01", "2026-01-31"),
            )
