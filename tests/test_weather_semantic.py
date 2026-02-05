"""Unit tests for weather semantic method and data tier methods."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from satellitehub._types import RawData
from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.location import Location, location
from satellitehub.results import BaseResult, WeatherResult

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config() -> Config:
    """Create a mock Config for testing."""
    return Config()


@pytest.fixture
def test_location(mock_config: Config) -> Location:
    """Create a test location in Poland (Warsaw area)."""
    return location(lat=52.23, lon=21.01, config=mock_config)


@pytest.fixture
def test_location_outside_poland(mock_config: Config) -> Location:
    """Create a test location outside Poland (London)."""
    return location(lat=51.51, lon=-0.13, config=mock_config)


@pytest.fixture
def mock_era5_raw() -> RawData:
    """Create mock ERA5 RawData."""
    return RawData(
        data=np.array([10.5, 11.2, 9.8], dtype=np.float32),
        metadata={
            "timestamps": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "variables": ["2m_temperature"],
        },
    )


@pytest.fixture
def mock_imgw_raw() -> RawData:
    """Create mock IMGW RawData."""
    return RawData(
        data=np.array([8.5], dtype=np.float32),
        metadata={
            "measurements": {"temperature": 8.5, "precipitation": 2.0},
            "observation_time": "2024-01-15T12:00:00Z",
            "station_name": "Warszawa-Okecie",
        },
    )


# ── Weather semantic method tests ─────────────────────────────────────────────


class TestWeatherSemanticMethod:
    """Test Location.weather() semantic method."""

    def test_weather_returns_weather_result(
        self, test_location: Location, mock_era5_raw: RawData
    ) -> None:
        """weather() returns a WeatherResult."""
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=mock_era5_raw
        ):
            result = test_location.weather(last_days=30)

        assert isinstance(result, WeatherResult)

    def test_weather_with_no_data_returns_zero_confidence(
        self, test_location: Location
    ) -> None:
        """weather() with no data returns confidence=0.0."""
        empty_raw = RawData(
            data=np.array([], dtype=np.float32),
            metadata={},
        )
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=empty_raw
        ):
            result = test_location.weather(last_days=30)

        assert result.confidence == 0.0
        assert len(result.warnings) > 0

    def test_weather_handles_provider_error_gracefully(
        self, test_location: Location
    ) -> None:
        """weather() handles ProviderError without raising."""
        with patch(
            "satellitehub._pipeline._acquire_weather",
            side_effect=ProviderError(
                what="Test error", cause="Test cause", fix="Test fix"
            ),
        ):
            result = test_location.weather(last_days=30)

        assert result.confidence == 0.0
        assert math.isnan(result.mean_temperature)

    def test_weather_handles_configuration_error_gracefully(
        self, test_location: Location
    ) -> None:
        """weather() handles ConfigurationError without raising."""
        with patch(
            "satellitehub._pipeline._acquire_weather",
            side_effect=ConfigurationError(
                what="Test error", cause="Test cause", fix="Test fix"
            ),
        ):
            result = test_location.weather(last_days=30)

        assert result.confidence == 0.0

    def test_weather_only_tries_imgw_for_polish_locations(
        self, test_location_outside_poland: Location, mock_era5_raw: RawData
    ) -> None:
        """weather() only tries IMGW for locations in Poland."""
        call_count: dict[str, int] = {"cds": 0, "imgw": 0}

        def mock_acquire(
            location: Any,
            provider_name: str,
            **kwargs: Any,
        ) -> RawData:
            call_count[provider_name] = call_count.get(provider_name, 0) + 1
            if provider_name == "cds":
                return mock_era5_raw
            return RawData(data=np.array([], dtype=np.float32), metadata={})

        with patch(
            "satellitehub._pipeline._acquire_weather", side_effect=mock_acquire
        ):
            test_location_outside_poland.weather(last_days=30)

        assert call_count["cds"] == 1
        assert call_count.get("imgw", 0) == 0

    def test_weather_tries_imgw_for_polish_locations(
        self, test_location: Location, mock_era5_raw: RawData, mock_imgw_raw: RawData
    ) -> None:
        """weather() tries IMGW for locations in Poland."""
        call_count: dict[str, int] = {"cds": 0, "imgw": 0}

        def mock_acquire(
            location: Any,
            provider_name: str,
            **kwargs: Any,
        ) -> RawData:
            call_count[provider_name] = call_count.get(provider_name, 0) + 1
            if provider_name == "cds":
                return mock_era5_raw
            return mock_imgw_raw

        with patch(
            "satellitehub._pipeline._acquire_weather", side_effect=mock_acquire
        ):
            test_location.weather(last_days=30)

        assert call_count["cds"] == 1
        assert call_count["imgw"] == 1


# ── DataTier.era5() tests ─────────────────────────────────────────────────────


class TestDataTierEra5:
    """Test DataTier.era5() method."""

    def test_era5_returns_base_result(
        self, test_location: Location, mock_era5_raw: RawData
    ) -> None:
        """era5() returns a BaseResult."""
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=mock_era5_raw
        ):
            result = test_location.data.era5(last_days=30)

        assert isinstance(result, BaseResult)

    def test_era5_with_no_data_returns_zero_confidence(
        self, test_location: Location
    ) -> None:
        """era5() with no data returns confidence=0.0."""
        empty_raw = RawData(
            data=np.array([], dtype=np.float32),
            metadata={},
        )
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=empty_raw
        ):
            result = test_location.data.era5(last_days=30)

        assert result.confidence == 0.0
        assert len(result.warnings) > 0

    def test_era5_handles_provider_error(self, test_location: Location) -> None:
        """era5() handles ProviderError without raising."""
        with patch(
            "satellitehub._pipeline._acquire_weather",
            side_effect=ProviderError(
                what="Test error", cause="Test cause", fix="Test fix"
            ),
        ):
            result = test_location.data.era5(last_days=30)

        assert result.confidence == 0.0
        assert len(result.warnings) > 0

    def test_era5_accepts_variables_parameter(
        self, test_location: Location, mock_era5_raw: RawData
    ) -> None:
        """era5() accepts variables parameter."""
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=mock_era5_raw
        ) as mock_acquire:
            test_location.data.era5(
                variables=["2m_temperature", "total_precipitation"],
                last_days=90,
            )

        mock_acquire.assert_called_once()
        call_kwargs = mock_acquire.call_args.kwargs
        assert call_kwargs["variables"] == ["2m_temperature", "total_precipitation"]


# ── DataTier.imgw() tests ─────────────────────────────────────────────────────


class TestDataTierImgw:
    """Test DataTier.imgw() method."""

    def test_imgw_returns_base_result(
        self, test_location: Location, mock_imgw_raw: RawData
    ) -> None:
        """imgw() returns a BaseResult."""
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=mock_imgw_raw
        ):
            result = test_location.data.imgw(last_days=30)

        assert isinstance(result, BaseResult)

    def test_imgw_with_no_data_returns_zero_confidence(
        self, test_location: Location
    ) -> None:
        """imgw() with no data returns confidence=0.0."""
        empty_raw = RawData(
            data=np.array([], dtype=np.float32),
            metadata={},
        )
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=empty_raw
        ):
            result = test_location.data.imgw(last_days=30)

        assert result.confidence == 0.0
        assert len(result.warnings) > 0

    def test_imgw_handles_provider_error(self, test_location: Location) -> None:
        """imgw() handles ProviderError without raising."""
        with patch(
            "satellitehub._pipeline._acquire_weather",
            side_effect=ProviderError(
                what="Test error", cause="Test cause", fix="Test fix"
            ),
        ):
            result = test_location.data.imgw(last_days=30)

        assert result.confidence == 0.0
        assert len(result.warnings) > 0

    def test_imgw_accepts_station_type_parameter(
        self, test_location: Location, mock_imgw_raw: RawData
    ) -> None:
        """imgw() accepts station_type parameter."""
        with patch(
            "satellitehub._pipeline._acquire_weather", return_value=mock_imgw_raw
        ) as mock_acquire:
            test_location.data.imgw(station_type="synop", last_days=30)

        mock_acquire.assert_called_once()
        call_kwargs = mock_acquire.call_args.kwargs
        assert call_kwargs["station_type"] == "synop"


# ── Location.bounds property tests ────────────────────────────────────────────


class TestLocationBounds:
    """Test Location.bounds property."""

    def test_bounds_returns_tuple(self, test_location: Location) -> None:
        """Bounds returns a 4-tuple."""
        bounds = test_location.bounds
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4

    def test_bounds_contains_coordinates(self, test_location: Location) -> None:
        """Bounds contains the location coordinates."""
        bounds = test_location.bounds
        # For a point, bounds are (lon, lat, lon, lat)
        assert bounds[0] == test_location.lon  # minx
        assert bounds[1] == test_location.lat  # miny
        assert bounds[2] == test_location.lon  # maxx
        assert bounds[3] == test_location.lat  # maxy


# ── Pipeline _acquire_weather tests ───────────────────────────────────────────


class TestAcquireWeather:
    """Test _acquire_weather pipeline function."""

    def test_acquire_weather_returns_empty_on_no_entries(self) -> None:
        """_acquire_weather returns empty RawData when no entries found."""
        from satellitehub._pipeline import _acquire_weather

        mock_provider = MagicMock()
        mock_provider.search.return_value = []

        # Create location with mocked cache and provider
        loc = location(lat=52.23, lon=21.01)
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache._build_cache_key.return_value = "test-key"
        loc._cache_manager = mock_cache
        loc._providers["cds"] = mock_provider

        with patch(
            "satellitehub._pipeline.resolve_credentials_path", return_value=None
        ):
            result = _acquire_weather(
                location=loc,
                provider_name="cds",
                last_days=30,
            )

        assert result.data.size == 0

    def test_acquire_weather_uses_cache_hit(
        self, mock_era5_raw: RawData
    ) -> None:
        """_acquire_weather returns cached data on cache hit."""
        from satellitehub._pipeline import (
            _acquire_weather,
            _serialize_raw_data,
        )

        cached_data = _serialize_raw_data(mock_era5_raw)

        # Create location with mocked cache
        loc = location(lat=52.23, lon=21.01)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_data
        mock_cache._build_cache_key.return_value = "test-key"
        loc._cache_manager = mock_cache

        # Provider should not be called
        mock_provider = MagicMock()
        loc._providers["cds"] = mock_provider

        result = _acquire_weather(
            location=loc,
            provider_name="cds",
            last_days=30,
        )

        # Provider should NOT be called when cache hit
        mock_provider.search.assert_not_called()
        mock_provider.download.assert_not_called()
        assert result.data.size > 0
