"""Unit tests for WeatherResult and weather aggregation functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from satellitehub import WeatherResult
from satellitehub.analysis.weather import (
    _calculate_confidence,
    aggregate_era5_daily,
    aggregate_imgw_daily,
    build_weather_result,
    merge_weather_sources,
)
from satellitehub.results import (
    ResultMetadata,
    _interpret_precipitation,
    _interpret_temperature,
)


# ── Test fixtures ─────────────────────────────────────────────────────────────


@dataclass
class MockLocation:
    """Mock Location for testing."""

    bounds: tuple[float, float, float, float] | None = None


@dataclass
class MockRawData:
    """Mock RawData for testing aggregation functions."""

    data: np.ndarray
    metadata: dict[str, Any]


# ── WeatherResult construction tests ──────────────────────────────────────────


class TestWeatherResultConstruction:
    """Test WeatherResult dataclass construction."""

    def test_default_values(self) -> None:
        """WeatherResult has correct default values."""
        result = WeatherResult(data=np.array([], dtype=np.float32))

        assert math.isnan(result.mean_temperature)
        assert math.isnan(result.total_precipitation)
        assert result.data_source == ""
        assert math.isnan(result.temperature_min)
        assert math.isnan(result.temperature_max)
        assert result.observation_count == 0
        assert result.confidence == 0.0
        assert result.warnings == []

    def test_full_construction(self) -> None:
        """WeatherResult accepts all fields."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            confidence=0.92,
            mean_temperature=4.2,
            total_precipitation=45.2,
            data_source="era5+imgw",
            temperature_min=-2.5,
            temperature_max=12.3,
            observation_count=31,
            warnings=["Test warning"],
        )

        assert result.mean_temperature == 4.2
        assert result.total_precipitation == 45.2
        assert result.data_source == "era5+imgw"
        assert result.temperature_min == -2.5
        assert result.temperature_max == 12.3
        assert result.observation_count == 31
        assert result.confidence == 0.92
        assert "Test warning" in result.warnings

    def test_inherits_base_result(self) -> None:
        """WeatherResult extends BaseResult."""
        result = WeatherResult(
            data=np.array([[1, 2], [3, 4]], dtype=np.float32),
            confidence=0.75,
            metadata=ResultMetadata(source="test"),
        )

        assert hasattr(result, "data")
        assert hasattr(result, "confidence")
        assert hasattr(result, "metadata")
        assert hasattr(result, "warnings")
        assert result.metadata.source == "test"


class TestWeatherResultRepr:
    """Test WeatherResult narrative __repr__ display."""

    def test_repr_basic(self) -> None:
        """__repr__ returns string starting with class name."""
        result = WeatherResult(data=np.array([], dtype=np.float32))
        repr_str = repr(result)

        assert repr_str.startswith("WeatherResult(")
        assert repr_str.endswith(")")

    def test_repr_shows_temperature(self) -> None:
        """__repr__ shows mean temperature with interpretation."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            mean_temperature=12.5,
        )
        repr_str = repr(result)

        assert "12.5" in repr_str
        assert "mild" in repr_str

    def test_repr_shows_precipitation(self) -> None:
        """__repr__ shows precipitation with interpretation."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            total_precipitation=25.0,
        )
        repr_str = repr(result)

        assert "25.0" in repr_str
        assert "moderate rainfall" in repr_str

    def test_repr_shows_temperature_range(self) -> None:
        """__repr__ shows temperature range when available."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            temperature_min=-2.5,
            temperature_max=12.3,
        )
        repr_str = repr(result)

        assert "-2.5" in repr_str
        assert "12.3" in repr_str
        assert "temperature_range" in repr_str

    def test_repr_shows_data_source(self) -> None:
        """__repr__ shows data source."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            data_source="era5+imgw",
        )
        repr_str = repr(result)

        assert "era5+imgw" in repr_str

    def test_repr_shows_location(self) -> None:
        """__repr__ shows location from metadata bounds."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            metadata=ResultMetadata(
                bounds={"minx": 22.0, "miny": 51.0, "maxx": 23.0, "maxy": 52.0}
            ),
        )
        repr_str = repr(result)

        assert "location:" in repr_str
        assert "N" in repr_str
        assert "E" in repr_str

    def test_repr_shows_period(self) -> None:
        """__repr__ shows period from timestamps."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            metadata=ResultMetadata(
                timestamps=["2024-01-01", "2024-01-15", "2024-01-31"]
            ),
        )
        repr_str = repr(result)

        assert "2024-01-01" in repr_str
        assert "2024-01-31" in repr_str

    def test_repr_shows_warnings(self) -> None:
        """__repr__ shows warnings with warning symbol."""
        result = WeatherResult(
            data=np.array([], dtype=np.float32),
            warnings=["Low confidence warning"],
        )
        repr_str = repr(result)

        assert "Low confidence warning" in repr_str
        assert "\u26a0" in repr_str  # Warning symbol

    def test_repr_handles_nan_values(self) -> None:
        """__repr__ handles NaN temperature and precipitation gracefully."""
        result = WeatherResult(data=np.array([], dtype=np.float32))
        repr_str = repr(result)

        assert "N/A" in repr_str


# ── Temperature interpretation tests ──────────────────────────────────────────


class TestInterpretTemperature:
    """Test _interpret_temperature helper function."""

    def test_cold_threshold(self) -> None:
        """Temperature below 5°C is cold."""
        assert _interpret_temperature(4.9) == "cold"
        assert _interpret_temperature(0.0) == "cold"
        assert _interpret_temperature(-10.0) == "cold"

    def test_mild_threshold(self) -> None:
        """Temperature 5-15°C is mild."""
        assert _interpret_temperature(5.0) == "mild"
        assert _interpret_temperature(10.0) == "mild"
        assert _interpret_temperature(14.9) == "mild"

    def test_warm_threshold(self) -> None:
        """Temperature 15-25°C is warm."""
        assert _interpret_temperature(15.0) == "warm"
        assert _interpret_temperature(20.0) == "warm"
        assert _interpret_temperature(24.9) == "warm"

    def test_hot_threshold(self) -> None:
        """Temperature above 25°C is hot."""
        assert _interpret_temperature(25.0) == "hot"
        assert _interpret_temperature(30.0) == "hot"
        assert _interpret_temperature(40.0) == "hot"

    def test_nan_returns_no_data(self) -> None:
        """NaN temperature returns 'no data'."""
        assert _interpret_temperature(float("nan")) == "no data"


# ── Precipitation interpretation tests ────────────────────────────────────────


class TestInterpretPrecipitation:
    """Test _interpret_precipitation helper function."""

    def test_dry_threshold(self) -> None:
        """Precipitation below 1mm is dry."""
        assert _interpret_precipitation(0.0) == "dry conditions"
        assert _interpret_precipitation(0.5) == "dry conditions"
        assert _interpret_precipitation(0.9) == "dry conditions"

    def test_light_rainfall_threshold(self) -> None:
        """Precipitation 1-10mm is light rainfall."""
        assert _interpret_precipitation(1.0) == "light rainfall"
        assert _interpret_precipitation(5.0) == "light rainfall"
        assert _interpret_precipitation(9.9) == "light rainfall"

    def test_moderate_rainfall_threshold(self) -> None:
        """Precipitation 10-30mm is moderate rainfall."""
        assert _interpret_precipitation(10.0) == "moderate rainfall"
        assert _interpret_precipitation(20.0) == "moderate rainfall"
        assert _interpret_precipitation(29.9) == "moderate rainfall"

    def test_heavy_rainfall_threshold(self) -> None:
        """Precipitation above 30mm is heavy rainfall."""
        assert _interpret_precipitation(30.0) == "heavy rainfall"
        assert _interpret_precipitation(50.0) == "heavy rainfall"
        assert _interpret_precipitation(100.0) == "heavy rainfall"

    def test_nan_returns_no_data(self) -> None:
        """NaN precipitation returns 'no data'."""
        assert _interpret_precipitation(float("nan")) == "no data"


# ── ERA5 aggregation tests ────────────────────────────────────────────────────


class TestAggregateEra5Daily:
    """Test aggregate_era5_daily function."""

    def test_empty_arrays(self) -> None:
        """Returns empty DataFrame for empty arrays."""
        raw_data = MockRawData(data=np.array([], dtype=np.float32), metadata={})
        df = aggregate_era5_daily(raw_data)

        assert df.empty
        assert "timestamp" in df.columns
        assert "temperature_mean" in df.columns
        assert "source" in df.columns

    def test_no_timestamps(self) -> None:
        """Returns empty DataFrame when no timestamps in metadata."""
        raw_data = MockRawData(
            data=np.array([10.0, 12.0], dtype=np.float32),
            metadata={},
        )
        df = aggregate_era5_daily(raw_data)

        assert df.empty

    def test_single_day_aggregation(self) -> None:
        """Aggregates single day correctly."""
        raw_data = MockRawData(
            data=np.array([[10.0], [12.0]], dtype=np.float32),
            metadata={"timestamps": ["2024-01-15T00:00:00Z", "2024-01-15T12:00:00Z"]},
        )
        df = aggregate_era5_daily(raw_data)

        assert len(df) == 1
        assert df.iloc[0]["timestamp"] == "2024-01-15"
        assert df.iloc[0]["source"] == "era5"

    def test_multi_day_aggregation(self) -> None:
        """Aggregates multiple days correctly."""
        raw_data = MockRawData(
            data=np.array([[10.0], [15.0]], dtype=np.float32),
            metadata={
                "timestamps": [
                    "2024-01-15T00:00:00Z",
                    "2024-01-16T00:00:00Z",
                ]
            },
        )
        df = aggregate_era5_daily(raw_data)

        assert len(df) == 2
        assert "2024-01-15" in df["timestamp"].values
        assert "2024-01-16" in df["timestamp"].values

    def test_output_columns(self) -> None:
        """Output has required columns."""
        raw_data = MockRawData(
            data=np.array([10.0], dtype=np.float32),
            metadata={"timestamps": ["2024-01-15T00:00:00Z"]},
        )
        df = aggregate_era5_daily(raw_data)

        expected_cols = [
            "timestamp",
            "temperature_min",
            "temperature_max",
            "temperature_mean",
            "precipitation",
            "source",
        ]
        for col in expected_cols:
            assert col in df.columns


# ── IMGW aggregation tests ────────────────────────────────────────────────────


class TestAggregateImgwDaily:
    """Test aggregate_imgw_daily function."""

    def test_empty_measurements(self) -> None:
        """Returns empty DataFrame for empty measurements."""
        raw_data = MockRawData(data=np.array([], dtype=np.float32), metadata={})
        df = aggregate_imgw_daily(raw_data)

        assert df.empty

    def test_single_measurement(self) -> None:
        """Aggregates single IMGW measurement."""
        raw_data = MockRawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "measurements": {"temperature": 8.5, "precipitation": 2.3},
                "observation_time": "2024-01-15T12:00:00Z",
            },
        )
        df = aggregate_imgw_daily(raw_data)

        assert len(df) == 1
        assert df.iloc[0]["timestamp"] == "2024-01-15"
        assert df.iloc[0]["temperature_mean"] == 8.5
        assert df.iloc[0]["precipitation"] == 2.3
        assert df.iloc[0]["source"] == "imgw"

    def test_output_columns(self) -> None:
        """Output has required columns."""
        raw_data = MockRawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "measurements": {"temperature": 10.0},
                "observation_time": "2024-01-15T12:00:00Z",
            },
        )
        df = aggregate_imgw_daily(raw_data)

        expected_cols = [
            "timestamp",
            "temperature_min",
            "temperature_max",
            "temperature_mean",
            "precipitation",
            "source",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_missing_temperature(self) -> None:
        """Handles missing temperature gracefully."""
        raw_data = MockRawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "measurements": {"precipitation": 5.0},
                "observation_time": "2024-01-15T12:00:00Z",
            },
        )
        df = aggregate_imgw_daily(raw_data)

        assert len(df) == 1
        assert math.isnan(df.iloc[0]["temperature_mean"])


# ── Merge sources tests ───────────────────────────────────────────────────────


class TestMergeWeatherSources:
    """Test merge_weather_sources function."""

    def test_both_empty(self) -> None:
        """Returns empty DataFrame when both inputs empty."""
        era5_df = pd.DataFrame()
        imgw_df = pd.DataFrame()
        merged = merge_weather_sources(era5_df, imgw_df)

        assert merged.empty

    def test_era5_only(self) -> None:
        """Returns ERA5 data when IMGW is empty."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 10.0,
                    "source": "era5",
                }
            ]
        )
        imgw_df = pd.DataFrame()
        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 1
        assert merged.iloc[0]["source"] == "era5"

    def test_imgw_only(self) -> None:
        """Returns IMGW data when ERA5 is empty."""
        era5_df = pd.DataFrame()
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 8.5,
                    "source": "imgw",
                }
            ]
        )
        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 1
        assert merged.iloc[0]["source"] == "imgw"

    def test_prefers_imgw_on_overlap(self) -> None:
        """IMGW data preferred for overlapping dates."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 10.0,
                    "source": "era5",
                }
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 8.5,
                    "source": "imgw",
                }
            ]
        )
        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 1
        assert merged.iloc[0]["source"] == "imgw"
        assert merged.iloc[0]["temperature_mean"] == 8.5

    def test_combines_non_overlapping(self) -> None:
        """Combines data from non-overlapping dates."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-14",
                    "temperature_mean": 9.0,
                    "source": "era5",
                }
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 8.5,
                    "source": "imgw",
                }
            ]
        )
        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 2
        sources = merged["source"].tolist()
        assert "era5" in sources
        assert "imgw" in sources

    def test_sorted_by_timestamp(self) -> None:
        """Result is sorted by timestamp."""
        era5_df = pd.DataFrame(
            [
                {"timestamp": "2024-01-16", "temperature_mean": 11.0, "source": "era5"}
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {"timestamp": "2024-01-14", "temperature_mean": 8.0, "source": "imgw"}
            ]
        )
        merged = merge_weather_sources(era5_df, imgw_df)

        assert merged.iloc[0]["timestamp"] == "2024-01-14"
        assert merged.iloc[1]["timestamp"] == "2024-01-16"


# ── Confidence calculation tests ──────────────────────────────────────────────


class TestCalculateConfidence:
    """Test _calculate_confidence function."""

    def test_zero_requested_days(self) -> None:
        """Returns 0 for zero requested days."""
        assert _calculate_confidence(30, 5, 0) == 0.0

    def test_full_era5_coverage(self) -> None:
        """Full ERA5 coverage gives 0.8 base confidence."""
        confidence = _calculate_confidence(31, 0, 31)
        assert confidence == pytest.approx(0.8)

    def test_era5_plus_imgw_boost(self) -> None:
        """IMGW adds up to 0.2 boost."""
        confidence = _calculate_confidence(31, 31, 31)
        assert confidence == pytest.approx(1.0)

    def test_partial_coverage(self) -> None:
        """Partial coverage reduces confidence proportionally."""
        confidence = _calculate_confidence(15, 0, 31)
        expected = (15 / 31) * 0.8
        assert confidence == pytest.approx(expected)

    def test_capped_at_one(self) -> None:
        """Confidence never exceeds 1.0."""
        confidence = _calculate_confidence(100, 100, 31)
        assert confidence <= 1.0


# ── Build weather result tests ────────────────────────────────────────────────


class TestBuildWeatherResult:
    """Test build_weather_result function."""

    def test_era5_only(self) -> None:
        """Builds result with ERA5 data only."""
        era5_data = MockRawData(
            data=np.array([10.0], dtype=np.float32),
            metadata={"timestamps": ["2024-01-15T00:00:00Z"]},
        )
        location = MockLocation(bounds=(22.0, 51.0, 23.0, 52.0))

        result = build_weather_result(
            era5_data=era5_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-15"),
        )

        assert result.data_source == "era5"
        assert result.observation_count >= 0

    def test_imgw_only(self) -> None:
        """Builds result with IMGW data only."""
        imgw_data = MockRawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "measurements": {"temperature": 8.5, "precipitation": 2.0},
                "observation_time": "2024-01-15T12:00:00Z",
            },
        )
        location = MockLocation(bounds=(22.0, 51.0, 23.0, 52.0))

        result = build_weather_result(
            era5_data=None,
            imgw_data=imgw_data,
            location=location,
            time_range=("2024-01-15", "2024-01-15"),
        )

        assert result.data_source == "imgw"
        assert result.mean_temperature == 8.5
        assert any("IMGW data only" in w for w in result.warnings)

    def test_both_sources(self) -> None:
        """Builds result with both ERA5 and IMGW data."""
        era5_data = MockRawData(
            data=np.array([10.0], dtype=np.float32),
            metadata={"timestamps": ["2024-01-14T00:00:00Z"]},
        )
        imgw_data = MockRawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "measurements": {"temperature": 8.5},
                "observation_time": "2024-01-15T12:00:00Z",
            },
        )
        location = MockLocation(bounds=(22.0, 51.0, 23.0, 52.0))

        result = build_weather_result(
            era5_data=era5_data,
            imgw_data=imgw_data,
            location=location,
            time_range=("2024-01-14", "2024-01-15"),
        )

        assert result.data_source == "era5+imgw"

    def test_no_data(self) -> None:
        """Builds result with no data gives zero confidence."""
        location = MockLocation(bounds=(22.0, 51.0, 23.0, 52.0))

        result = build_weather_result(
            era5_data=None,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-31"),
        )

        assert result.confidence == 0.0
        assert result.data_source == ""
        assert math.isnan(result.mean_temperature)
        assert any("No weather data available" in w for w in result.warnings)

    def test_metadata_populated(self) -> None:
        """Metadata is populated correctly."""
        era5_data = MockRawData(
            data=np.array([10.0], dtype=np.float32),
            metadata={"timestamps": ["2024-01-15T00:00:00Z"]},
        )
        location = MockLocation(bounds=(22.0, 51.0, 23.0, 52.0))

        result = build_weather_result(
            era5_data=era5_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-15"),
        )

        assert result.metadata.source == "era5"
        assert result.metadata.bounds["minx"] == 22.0

    def test_low_confidence_warning(self) -> None:
        """Adds warning for low confidence."""
        era5_data = MockRawData(
            data=np.array([10.0], dtype=np.float32),
            metadata={"timestamps": ["2024-01-15T00:00:00Z"]},
        )
        location = MockLocation(bounds=(22.0, 51.0, 23.0, 52.0))

        # Request 31 days but only provide 1 day of data
        result = build_weather_result(
            era5_data=era5_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-01", "2024-01-31"),
        )

        assert result.confidence < 0.5
        assert any("Low confidence" in w for w in result.warnings)


# ── Export tests ──────────────────────────────────────────────────────────────


class TestWeatherResultExport:
    """Test WeatherResult is exported from satellitehub."""

    def test_import_from_package(self) -> None:
        """WeatherResult can be imported from satellitehub."""
        from satellitehub import WeatherResult as ImportedResult

        assert ImportedResult is WeatherResult

    def test_in_all(self) -> None:
        """WeatherResult is in __all__."""
        import satellitehub

        assert "WeatherResult" in satellitehub.__all__
