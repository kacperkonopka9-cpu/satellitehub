"""Tests for the weather analysis module."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from satellitehub._types import RawData
from satellitehub.analysis.weather import (
    _calculate_confidence,
    aggregate_era5_daily,
    aggregate_imgw_daily,
    build_weather_result,
    merge_weather_sources,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def era5_raw_data() -> RawData:
    """Create sample ERA5 RawData with temperature and precipitation."""
    # 4 timestamps per day for 2 days = 8 total
    data = np.array(
        [
            [5.0, 0.001],  # Day 1, 00:00
            [3.0, 0.002],  # Day 1, 06:00
            [8.0, 0.000],  # Day 1, 12:00
            [6.0, 0.001],  # Day 1, 18:00
            [4.0, 0.003],  # Day 2, 00:00
            [2.0, 0.001],  # Day 2, 06:00
            [7.0, 0.000],  # Day 2, 12:00
            [5.0, 0.002],  # Day 2, 18:00
        ],
        dtype=np.float32,
    )
    return RawData(
        data=data,
        metadata={
            "timestamps": [
                "2024-01-15T00:00:00Z",
                "2024-01-15T06:00:00Z",
                "2024-01-15T12:00:00Z",
                "2024-01-15T18:00:00Z",
                "2024-01-16T00:00:00Z",
                "2024-01-16T06:00:00Z",
                "2024-01-16T12:00:00Z",
                "2024-01-16T18:00:00Z",
            ],
            "variables": ["2m_temperature", "total_precipitation"],
        },
    )


@pytest.fixture
def era5_empty_data() -> RawData:
    """Create empty ERA5 RawData."""
    return RawData(
        data=np.array([], dtype=np.float32),
        metadata={"timestamps": [], "variables": []},
    )


@pytest.fixture
def imgw_raw_data() -> RawData:
    """Create sample IMGW RawData with station measurements."""
    return RawData(
        data=np.array([], dtype=np.float32),  # IMGW uses metadata
        metadata={
            "station_id": "12375",
            "station_name": "Warszawa",
            "observation_time": "2024-01-15T12:00:00Z",
            "measurements": {
                "temperature": 6.5,
                "precipitation": 1.2,
                "humidity": 75.0,
                "pressure": 1015.5,
            },
        },
    )


@pytest.fixture
def imgw_empty_data() -> RawData:
    """Create IMGW RawData with no measurements."""
    return RawData(
        data=np.array([], dtype=np.float32),
        metadata={"station_id": "12375", "measurements": {}},
    )


def _make_location(lat: float = 52.23, lon: float = 21.01) -> Any:
    """Create a mock Location object."""
    loc = MagicMock()
    loc.lat = lat
    loc.lon = lon
    loc.bounds = (lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)
    return loc


# ---------------------------------------------------------------------------
# aggregate_era5_daily Tests
# ---------------------------------------------------------------------------


class TestAggregateERA5Daily:
    """Test aggregate_era5_daily function."""

    def test_aggregates_to_daily_records(self, era5_raw_data: RawData) -> None:
        """Aggregates sub-daily data to daily records."""
        df = aggregate_era5_daily(era5_raw_data)

        assert len(df) == 2
        assert list(df["timestamp"]) == ["2024-01-15", "2024-01-16"]

    def test_calculates_daily_temperature_stats(
        self, era5_raw_data: RawData
    ) -> None:
        """Calculates min, max, mean temperature for each day."""
        df = aggregate_era5_daily(era5_raw_data)

        # Day 1: temps = [5, 3, 8, 6] -> min=3, max=8, mean=5.5
        day1 = df[df["timestamp"] == "2024-01-15"].iloc[0]
        assert day1["temperature_min"] == pytest.approx(3.0)
        assert day1["temperature_max"] == pytest.approx(8.0)
        assert day1["temperature_mean"] == pytest.approx(5.5)

        # Day 2: temps = [4, 2, 7, 5] -> min=2, max=7, mean=4.5
        day2 = df[df["timestamp"] == "2024-01-16"].iloc[0]
        assert day2["temperature_min"] == pytest.approx(2.0)
        assert day2["temperature_max"] == pytest.approx(7.0)
        assert day2["temperature_mean"] == pytest.approx(4.5)

    def test_sums_daily_precipitation(self, era5_raw_data: RawData) -> None:
        """Sums precipitation for each day and converts to mm."""
        df = aggregate_era5_daily(era5_raw_data)

        # Day 1: precip = [0.001, 0.002, 0.000, 0.001] m -> 4mm
        day1 = df[df["timestamp"] == "2024-01-15"].iloc[0]
        assert day1["precipitation"] == pytest.approx(4.0)

        # Day 2: precip = [0.003, 0.001, 0.000, 0.002] m -> 6mm
        day2 = df[df["timestamp"] == "2024-01-16"].iloc[0]
        assert day2["precipitation"] == pytest.approx(6.0)

    def test_sets_source_to_era5(self, era5_raw_data: RawData) -> None:
        """All records have source='era5'."""
        df = aggregate_era5_daily(era5_raw_data)

        assert all(df["source"] == "era5")

    def test_returns_empty_for_no_data(self, era5_empty_data: RawData) -> None:
        """Returns empty DataFrame when no data."""
        df = aggregate_era5_daily(era5_empty_data)

        assert df.empty
        assert list(df.columns) == [
            "timestamp",
            "temperature_min",
            "temperature_max",
            "temperature_mean",
            "precipitation",
            "source",
        ]

    def test_returns_empty_for_no_timestamps(self) -> None:
        """Returns empty DataFrame when no timestamps in metadata."""
        raw_data = RawData(
            data=np.array([[5.0, 0.001]], dtype=np.float32),
            metadata={"timestamps": [], "variables": ["2m_temperature", "tp"]},
        )

        df = aggregate_era5_daily(raw_data)

        assert df.empty

    def test_handles_1d_array(self) -> None:
        """Handles 1D data array (single variable)."""
        raw_data = RawData(
            data=np.array([5.0, 3.0, 8.0, 6.0], dtype=np.float32),
            metadata={
                "timestamps": [
                    "2024-01-15T00:00:00Z",
                    "2024-01-15T06:00:00Z",
                    "2024-01-15T12:00:00Z",
                    "2024-01-15T18:00:00Z",
                ],
                "variables": ["2m_temperature"],
            },
        )

        df = aggregate_era5_daily(raw_data)

        assert len(df) == 1
        assert df.iloc[0]["temperature_min"] == pytest.approx(3.0)
        assert df.iloc[0]["temperature_max"] == pytest.approx(8.0)

    def test_handles_nan_values(self) -> None:
        """Handles NaN values in data array."""
        raw_data = RawData(
            data=np.array(
                [[5.0, 0.001], [np.nan, 0.002], [8.0, np.nan], [6.0, 0.001]],
                dtype=np.float32,
            ),
            metadata={
                "timestamps": [
                    "2024-01-15T00:00:00Z",
                    "2024-01-15T06:00:00Z",
                    "2024-01-15T12:00:00Z",
                    "2024-01-15T18:00:00Z",
                ],
                "variables": ["2m_temperature", "total_precipitation"],
            },
        )

        df = aggregate_era5_daily(raw_data)

        # Should ignore NaN when calculating stats
        assert df.iloc[0]["temperature_min"] == pytest.approx(5.0)  # min of [5, 8, 6]
        assert df.iloc[0]["temperature_max"] == pytest.approx(8.0)

    def test_handles_alternative_variable_names(self) -> None:
        """Handles alternative ERA5 variable names (t2m, tp)."""
        raw_data = RawData(
            data=np.array([[10.0, 0.005]], dtype=np.float32),
            metadata={
                "timestamps": ["2024-01-15T12:00:00Z"],
                "variables": ["t2m", "tp"],
            },
        )

        df = aggregate_era5_daily(raw_data)

        assert len(df) == 1
        assert df.iloc[0]["temperature_mean"] == pytest.approx(10.0)
        assert df.iloc[0]["precipitation"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# aggregate_imgw_daily Tests
# ---------------------------------------------------------------------------


class TestAggregateIMGWDaily:
    """Test aggregate_imgw_daily function."""

    def test_creates_single_daily_record(self, imgw_raw_data: RawData) -> None:
        """Creates single daily record from IMGW observation."""
        df = aggregate_imgw_daily(imgw_raw_data)

        assert len(df) == 1
        assert df.iloc[0]["timestamp"] == "2024-01-15"

    def test_extracts_temperature(self, imgw_raw_data: RawData) -> None:
        """Extracts temperature from measurements."""
        df = aggregate_imgw_daily(imgw_raw_data)

        # Single observation: min=max=mean=6.5
        assert df.iloc[0]["temperature_min"] == pytest.approx(6.5)
        assert df.iloc[0]["temperature_max"] == pytest.approx(6.5)
        assert df.iloc[0]["temperature_mean"] == pytest.approx(6.5)

    def test_extracts_precipitation(self, imgw_raw_data: RawData) -> None:
        """Extracts precipitation from measurements."""
        df = aggregate_imgw_daily(imgw_raw_data)

        assert df.iloc[0]["precipitation"] == pytest.approx(1.2)

    def test_sets_source_to_imgw(self, imgw_raw_data: RawData) -> None:
        """Sets source to 'imgw'."""
        df = aggregate_imgw_daily(imgw_raw_data)

        assert df.iloc[0]["source"] == "imgw"

    def test_returns_empty_for_no_measurements(
        self, imgw_empty_data: RawData
    ) -> None:
        """Returns empty DataFrame when no measurements."""
        df = aggregate_imgw_daily(imgw_empty_data)

        assert df.empty

    def test_handles_missing_precipitation(self) -> None:
        """Handles missing precipitation (defaults to 0)."""
        raw_data = RawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "observation_time": "2024-01-15T12:00:00Z",
                "measurements": {"temperature": 5.0},
            },
        )

        df = aggregate_imgw_daily(raw_data)

        assert df.iloc[0]["precipitation"] == 0.0

    def test_handles_missing_temperature(self) -> None:
        """Handles missing temperature (returns NaN)."""
        raw_data = RawData(
            data=np.array([], dtype=np.float32),
            metadata={
                "observation_time": "2024-01-15T12:00:00Z",
                "measurements": {"precipitation": 1.0},
            },
        )

        df = aggregate_imgw_daily(raw_data)

        assert np.isnan(df.iloc[0]["temperature_mean"])

    def test_uses_current_date_if_no_observation_time(self) -> None:
        """Uses current date if observation_time is missing."""
        raw_data = RawData(
            data=np.array([], dtype=np.float32),
            metadata={"measurements": {"temperature": 5.0}},
        )

        df = aggregate_imgw_daily(raw_data)

        # Should use today's date
        expected_date = datetime.now().strftime("%Y-%m-%d")
        assert df.iloc[0]["timestamp"] == expected_date


# ---------------------------------------------------------------------------
# merge_weather_sources Tests
# ---------------------------------------------------------------------------


class TestMergeWeatherSources:
    """Test merge_weather_sources function."""

    def test_prefers_imgw_for_overlapping_dates(self) -> None:
        """Prefers IMGW data for overlapping dates."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 5.0,
                    "temperature_min": 3.0,
                    "temperature_max": 7.0,
                    "precipitation": 2.0,
                    "source": "era5",
                }
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 6.5,
                    "temperature_min": 6.5,
                    "temperature_max": 6.5,
                    "precipitation": 1.2,
                    "source": "imgw",
                }
            ]
        )

        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 1
        assert merged.iloc[0]["source"] == "imgw"
        assert merged.iloc[0]["temperature_mean"] == pytest.approx(6.5)

    def test_combines_non_overlapping_dates(self) -> None:
        """Combines data from non-overlapping dates."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-14",
                    "temperature_mean": 4.0,
                    "temperature_min": 2.0,
                    "temperature_max": 6.0,
                    "precipitation": 1.0,
                    "source": "era5",
                }
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 6.5,
                    "temperature_min": 6.5,
                    "temperature_max": 6.5,
                    "precipitation": 1.2,
                    "source": "imgw",
                }
            ]
        )

        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 2
        assert merged.iloc[0]["timestamp"] == "2024-01-14"
        assert merged.iloc[0]["source"] == "era5"
        assert merged.iloc[1]["timestamp"] == "2024-01-15"
        assert merged.iloc[1]["source"] == "imgw"

    def test_sorted_by_timestamp(self) -> None:
        """Result is sorted by timestamp."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-17",
                    "temperature_mean": 5.0,
                    "temperature_min": 3.0,
                    "temperature_max": 7.0,
                    "precipitation": 2.0,
                    "source": "era5",
                },
                {
                    "timestamp": "2024-01-14",
                    "temperature_mean": 4.0,
                    "temperature_min": 2.0,
                    "temperature_max": 6.0,
                    "precipitation": 1.0,
                    "source": "era5",
                },
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 6.5,
                    "temperature_min": 6.5,
                    "temperature_max": 6.5,
                    "precipitation": 1.2,
                    "source": "imgw",
                }
            ]
        )

        merged = merge_weather_sources(era5_df, imgw_df)

        assert list(merged["timestamp"]) == [
            "2024-01-14",
            "2024-01-15",
            "2024-01-17",
        ]

    def test_returns_empty_for_both_empty(self) -> None:
        """Returns empty DataFrame when both inputs are empty."""
        era5_df = pd.DataFrame(
            columns=[
                "timestamp",
                "temperature_min",
                "temperature_max",
                "temperature_mean",
                "precipitation",
                "source",
            ]
        )
        imgw_df = era5_df.copy()

        merged = merge_weather_sources(era5_df, imgw_df)

        assert merged.empty

    def test_returns_era5_when_imgw_empty(self) -> None:
        """Returns ERA5 data when IMGW is empty."""
        era5_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 5.0,
                    "temperature_min": 3.0,
                    "temperature_max": 7.0,
                    "precipitation": 2.0,
                    "source": "era5",
                }
            ]
        )
        imgw_df = pd.DataFrame(
            columns=[
                "timestamp",
                "temperature_min",
                "temperature_max",
                "temperature_mean",
                "precipitation",
                "source",
            ]
        )

        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 1
        assert merged.iloc[0]["source"] == "era5"

    def test_returns_imgw_when_era5_empty(self) -> None:
        """Returns IMGW data when ERA5 is empty."""
        era5_df = pd.DataFrame(
            columns=[
                "timestamp",
                "temperature_min",
                "temperature_max",
                "temperature_mean",
                "precipitation",
                "source",
            ]
        )
        imgw_df = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-15",
                    "temperature_mean": 6.5,
                    "temperature_min": 6.5,
                    "temperature_max": 6.5,
                    "precipitation": 1.2,
                    "source": "imgw",
                }
            ]
        )

        merged = merge_weather_sources(era5_df, imgw_df)

        assert len(merged) == 1
        assert merged.iloc[0]["source"] == "imgw"


# ---------------------------------------------------------------------------
# _calculate_confidence Tests
# ---------------------------------------------------------------------------


class TestCalculateConfidence:
    """Test _calculate_confidence function."""

    def test_zero_requested_days(self) -> None:
        """Returns 0.0 for zero requested days."""
        confidence = _calculate_confidence(era5_days=10, imgw_days=5, requested_days=0)

        assert confidence == 0.0

    def test_full_era5_coverage(self) -> None:
        """Returns 0.8 for full ERA5 coverage without IMGW."""
        confidence = _calculate_confidence(
            era5_days=31, imgw_days=0, requested_days=31
        )

        assert confidence == pytest.approx(0.8)

    def test_era5_plus_imgw_boost(self) -> None:
        """ERA5 + IMGW gives confidence boost."""
        confidence_era5_only = _calculate_confidence(
            era5_days=31, imgw_days=0, requested_days=31
        )
        confidence_with_imgw = _calculate_confidence(
            era5_days=31, imgw_days=31, requested_days=31
        )

        assert confidence_with_imgw > confidence_era5_only
        assert confidence_with_imgw == pytest.approx(1.0)

    def test_partial_era5_coverage(self) -> None:
        """Returns proportional confidence for partial ERA5 coverage."""
        confidence = _calculate_confidence(
            era5_days=15, imgw_days=0, requested_days=30
        )

        # 50% ERA5 coverage * 0.8 = 0.4
        assert confidence == pytest.approx(0.4)

    def test_capped_at_one(self) -> None:
        """Confidence is capped at 1.0."""
        confidence = _calculate_confidence(
            era5_days=60, imgw_days=60, requested_days=30
        )

        assert confidence == 1.0

    def test_imgw_only(self) -> None:
        """IMGW only gives partial confidence."""
        confidence = _calculate_confidence(
            era5_days=0, imgw_days=30, requested_days=30
        )

        # 0 ERA5 + 20% IMGW boost = 0.2
        assert confidence == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# build_weather_result Tests
# ---------------------------------------------------------------------------


class TestBuildWeatherResult:
    """Test build_weather_result function."""

    def test_era5_only(self, era5_raw_data: RawData) -> None:
        """Builds result from ERA5 data only."""
        location = _make_location()

        result = build_weather_result(
            era5_data=era5_raw_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-16"),
        )

        assert result.data_source == "era5"
        assert result.observation_count == 2
        assert result.confidence > 0

    def test_imgw_only(self, imgw_raw_data: RawData) -> None:
        """Builds result from IMGW data only."""
        location = _make_location()

        result = build_weather_result(
            era5_data=None,
            imgw_data=imgw_raw_data,
            location=location,
            time_range=("2024-01-15", "2024-01-15"),
        )

        assert result.data_source == "imgw"
        assert result.observation_count == 1
        assert any("IMGW data only" in w for w in result.warnings)

    def test_both_sources(
        self, era5_raw_data: RawData, imgw_raw_data: RawData
    ) -> None:
        """Builds result from both ERA5 and IMGW data."""
        location = _make_location()

        result = build_weather_result(
            era5_data=era5_raw_data,
            imgw_data=imgw_raw_data,
            location=location,
            time_range=("2024-01-15", "2024-01-16"),
        )

        assert result.data_source == "era5+imgw"

    def test_no_data(self) -> None:
        """Handles case with no data from either source."""
        location = _make_location()

        result = build_weather_result(
            era5_data=None,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-16"),
        )

        assert result.data_source == ""
        assert result.observation_count == 0
        assert result.confidence == 0.0
        assert any("No weather data available" in w for w in result.warnings)

    def test_calculates_temperature_stats(self, era5_raw_data: RawData) -> None:
        """Calculates correct temperature statistics."""
        location = _make_location()

        result = build_weather_result(
            era5_data=era5_raw_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-16"),
        )

        # Day 1: mean=5.5, Day 2: mean=4.5 -> overall mean=5.0
        assert result.mean_temperature == pytest.approx(5.0)
        # Min across all: 2.0 (Day 2)
        assert result.temperature_min == pytest.approx(2.0)
        # Max across all: 8.0 (Day 1)
        assert result.temperature_max == pytest.approx(8.0)

    def test_calculates_total_precipitation(self, era5_raw_data: RawData) -> None:
        """Calculates total precipitation across all days."""
        location = _make_location()

        result = build_weather_result(
            era5_data=era5_raw_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-16"),
        )

        # Day 1: 4mm, Day 2: 6mm -> total 10mm
        assert result.total_precipitation == pytest.approx(10.0)

    def test_metadata_populated(self, era5_raw_data: RawData) -> None:
        """Populates result metadata correctly."""
        location = _make_location()

        result = build_weather_result(
            era5_data=era5_raw_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-15", "2024-01-16"),
        )

        assert result.metadata.source == "era5"
        assert result.metadata.observation_count == 2
        assert len(result.metadata.timestamps) == 2
        assert "minx" in result.metadata.bounds

    def test_low_confidence_warning(self) -> None:
        """Adds warning for low confidence results."""
        location = _make_location()
        # Only 1 day of data for 30 day request = low confidence
        sparse_data = RawData(
            data=np.array([[5.0, 0.001]], dtype=np.float32),
            metadata={
                "timestamps": ["2024-01-15T12:00:00Z"],
                "variables": ["2m_temperature", "total_precipitation"],
            },
        )

        result = build_weather_result(
            era5_data=sparse_data,
            imgw_data=None,
            location=location,
            time_range=("2024-01-01", "2024-01-31"),
        )

        assert result.confidence < 0.5
        assert any("Low confidence" in w for w in result.warnings)

    def test_handles_invalid_time_range(self, era5_raw_data: RawData) -> None:
        """Handles invalid time range gracefully."""
        location = _make_location()

        result = build_weather_result(
            era5_data=era5_raw_data,
            imgw_data=None,
            location=location,
            time_range=("invalid", "dates"),
        )

        # Should still work, but with 0 requested days
        assert result.confidence == 0.0
