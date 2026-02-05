"""Unit tests for result export methods (Story 5.1)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from satellitehub.results import (
    BaseResult,
    ChangeResult,
    ResultMetadata,
    VegetationResult,
    WeatherResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_metadata() -> ResultMetadata:
    """Create sample metadata with bounds and timestamps."""
    return ResultMetadata(
        source="cdse",
        timestamps=["2024-01-01T00:00:00Z", "2024-01-15T00:00:00Z"],
        observation_count=5,
        crs="EPSG:32634",
        bounds={"minx": 21.0, "miny": 52.0, "maxx": 21.1, "maxy": 52.1},
    )


@pytest.fixture
def base_result(sample_metadata: ResultMetadata) -> BaseResult:
    """Create a BaseResult with sample raster data."""
    data = np.random.rand(1, 100, 100).astype(np.float32)
    return BaseResult(
        data=data,
        confidence=0.75,
        metadata=sample_metadata,
        warnings=["Test warning"],
    )


@pytest.fixture
def vegetation_result(sample_metadata: ResultMetadata) -> VegetationResult:
    """Create a VegetationResult with sample NDVI data."""
    data = np.random.rand(1, 100, 100).astype(np.float32) * 0.8
    return VegetationResult(
        data=data,
        confidence=0.82,
        metadata=sample_metadata,
        mean_ndvi=0.45,
        ndvi_std=0.12,
        trend=0.02,
        observation_count=6,
        cloud_free_count=4,
    )


@pytest.fixture
def change_result(sample_metadata: ResultMetadata) -> ChangeResult:
    """Create a ChangeResult with sample change data."""
    data = np.random.rand(1, 100, 100).astype(np.float32) * 0.2 - 0.1
    return ChangeResult(
        data=data,
        confidence=0.68,
        metadata=sample_metadata,
        period_1_ndvi=0.52,
        period_2_ndvi=0.38,
        delta=-0.14,
        direction="declining",
        period_1_confidence=0.78,
        period_2_confidence=0.65,
        period_1_range=("2023-06-01", "2023-06-30"),
        period_2_range=("2024-06-01", "2024-06-30"),
    )


@pytest.fixture
def weather_result(sample_metadata: ResultMetadata) -> WeatherResult:
    """Create a WeatherResult with sample weather data."""
    # Create a DataFrame for time series
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
            "temperature_mean": np.random.rand(30) * 10 + 5,
            "temperature_min": np.random.rand(30) * 5,
            "temperature_max": np.random.rand(30) * 10 + 10,
            "precipitation": np.random.rand(30) * 5,
            "source": ["era5"] * 30,
        }
    )
    return WeatherResult(
        data=df,  # type: ignore[arg-type]
        confidence=0.85,
        metadata=sample_metadata,
        mean_temperature=12.5,
        total_precipitation=45.2,
        data_source="era5+imgw",
        temperature_min=2.1,
        temperature_max=18.7,
        observation_count=30,
    )


@pytest.fixture
def empty_base_result() -> BaseResult:
    """Create an empty BaseResult (confidence=0, no data)."""
    return BaseResult(
        data=np.array([], dtype=np.float32),
        confidence=0.0,
        metadata=ResultMetadata(),
        warnings=["No data available"],
    )


@pytest.fixture
def empty_vegetation_result() -> VegetationResult:
    """Create an empty VegetationResult."""
    return VegetationResult(
        data=np.array([], dtype=np.float32),
        confidence=0.0,
        metadata=ResultMetadata(),
        mean_ndvi=float("nan"),
        ndvi_std=float("nan"),
        observation_count=0,
        cloud_free_count=0,
    )


@pytest.fixture
def empty_weather_result() -> WeatherResult:
    """Create an empty WeatherResult."""
    return WeatherResult(
        data=np.array([], dtype=np.float32),
        confidence=0.0,
        metadata=ResultMetadata(),
        mean_temperature=float("nan"),
        total_precipitation=float("nan"),
        observation_count=0,
    )


# ── BaseResult.to_dataframe() tests ───────────────────────────────────────────


class TestBaseResultToDataFrame:
    """Tests for BaseResult.to_dataframe()."""

    def test_returns_dataframe(self, base_result: BaseResult) -> None:
        """to_dataframe() returns a pandas DataFrame."""
        df = base_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_contains_confidence(self, base_result: BaseResult) -> None:
        """DataFrame contains confidence column."""
        df = base_result.to_dataframe()
        assert "confidence" in df.columns
        assert df["confidence"].iloc[0] == 0.75

    def test_contains_location(self, base_result: BaseResult) -> None:
        """DataFrame contains latitude and longitude."""
        df = base_result.to_dataframe()
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        # Center of bounds (52.0, 52.1) and (21.0, 21.1)
        assert df["latitude"].iloc[0] == pytest.approx(52.05, abs=0.01)
        assert df["longitude"].iloc[0] == pytest.approx(21.05, abs=0.01)

    def test_contains_timestamps(self, base_result: BaseResult) -> None:
        """DataFrame contains period_start and period_end."""
        df = base_result.to_dataframe()
        assert "period_start" in df.columns
        assert "period_end" in df.columns

    def test_contains_data_summary(self, base_result: BaseResult) -> None:
        """DataFrame contains data statistics for non-empty results."""
        df = base_result.to_dataframe()
        assert "data_min" in df.columns
        assert "data_max" in df.columns
        assert "data_mean" in df.columns

    def test_empty_result_has_no_data_stats(
        self, empty_base_result: BaseResult
    ) -> None:
        """Empty result has no data statistics."""
        df = empty_base_result.to_dataframe()
        # Should not have data_min/max/mean if data is empty
        assert "data_min" not in df.columns or pd.isna(df["data_min"].iloc[0])


# ── BaseResult.to_geotiff() tests ─────────────────────────────────────────────


class TestBaseResultToGeoTiff:
    """Tests for BaseResult.to_geotiff()."""

    def test_writes_file(self, base_result: BaseResult) -> None:
        """to_geotiff() writes a file and returns Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.tif"
            result_path = base_result.to_geotiff(output_path)

            assert result_path == output_path
            assert output_path.exists()

    def test_returns_path_object(self, base_result: BaseResult) -> None:
        """to_geotiff() returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.tif"
            result_path = base_result.to_geotiff(output_path)
            assert isinstance(result_path, Path)

    def test_accepts_string_path(self, base_result: BaseResult) -> None:
        """to_geotiff() accepts string path argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/output.tif"
            result_path = base_result.to_geotiff(output_path)
            assert isinstance(result_path, Path)
            assert result_path.exists()

    def test_empty_result_raises_error(self, empty_base_result: BaseResult) -> None:
        """to_geotiff() raises ValueError for empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.tif"
            with pytest.raises(ValueError, match="empty result"):
                empty_base_result.to_geotiff(output_path)

    def test_1d_data_raises_error(self) -> None:
        """to_geotiff() raises ValueError for 1D data."""
        result = BaseResult(
            data=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            confidence=0.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.tif"
            with pytest.raises(ValueError, match="at least 2D"):
                result.to_geotiff(output_path)

    def test_geotiff_has_correct_crs(self, base_result: BaseResult) -> None:
        """GeoTIFF file has correct CRS metadata."""
        import rasterio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.tif"
            base_result.to_geotiff(output_path)

            with rasterio.open(output_path) as src:
                assert src.crs is not None
                assert "32634" in str(src.crs)


# ── BaseResult.to_png() tests ─────────────────────────────────────────────────


class TestBaseResultToPng:
    """Tests for BaseResult.to_png()."""

    def test_writes_file(self, base_result: BaseResult) -> None:
        """to_png() writes a file and returns Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.png"
            result_path = base_result.to_png(output_path)

            assert result_path == output_path
            assert output_path.exists()

    def test_returns_path_object(self, base_result: BaseResult) -> None:
        """to_png() returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.png"
            result_path = base_result.to_png(output_path)
            assert isinstance(result_path, Path)

    def test_empty_result_still_writes(self, empty_base_result: BaseResult) -> None:
        """to_png() handles empty results gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.png"
            empty_base_result.to_png(output_path)
            assert output_path.exists()


# ── VegetationResult.to_dataframe() tests ─────────────────────────────────────


class TestVegetationResultToDataFrame:
    """Tests for VegetationResult.to_dataframe()."""

    def test_contains_ndvi_fields(self, vegetation_result: VegetationResult) -> None:
        """DataFrame contains NDVI-specific fields."""
        df = vegetation_result.to_dataframe()
        assert "mean_ndvi" in df.columns
        assert "ndvi_std" in df.columns
        assert "ndvi_interpretation" in df.columns
        assert "trend" in df.columns

    def test_ndvi_values_correct(self, vegetation_result: VegetationResult) -> None:
        """NDVI values in DataFrame match result."""
        df = vegetation_result.to_dataframe()
        assert df["mean_ndvi"].iloc[0] == pytest.approx(0.45)
        assert df["ndvi_std"].iloc[0] == pytest.approx(0.12)

    def test_contains_observation_counts(
        self, vegetation_result: VegetationResult
    ) -> None:
        """DataFrame contains observation count fields."""
        df = vegetation_result.to_dataframe()
        assert "observation_count" in df.columns
        assert "cloud_free_count" in df.columns
        assert df["observation_count"].iloc[0] == 6
        assert df["cloud_free_count"].iloc[0] == 4

    def test_interpretation_correct(self, vegetation_result: VegetationResult) -> None:
        """NDVI interpretation is correct."""
        df = vegetation_result.to_dataframe()
        # 0.45 NDVI should be "moderate vegetation"
        assert df["ndvi_interpretation"].iloc[0] == "moderate vegetation"


# ── VegetationResult.to_png() tests ───────────────────────────────────────────


class TestVegetationResultToPng:
    """Tests for VegetationResult.to_png()."""

    def test_writes_file(self, vegetation_result: VegetationResult) -> None:
        """to_png() writes a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ndvi.png"
            result_path = vegetation_result.to_png(output_path)
            assert output_path.exists()
            assert result_path == output_path

    def test_empty_result_handled(
        self, empty_vegetation_result: VegetationResult
    ) -> None:
        """to_png() handles empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ndvi.png"
            empty_vegetation_result.to_png(output_path)
            assert output_path.exists()


# ── ChangeResult.to_dataframe() tests ─────────────────────────────────────────


class TestChangeResultToDataFrame:
    """Tests for ChangeResult.to_dataframe()."""

    def test_contains_change_fields(self, change_result: ChangeResult) -> None:
        """DataFrame contains change-specific fields."""
        df = change_result.to_dataframe()
        assert "period_1_ndvi" in df.columns
        assert "period_2_ndvi" in df.columns
        assert "delta" in df.columns
        assert "direction" in df.columns
        assert "change_interpretation" in df.columns

    def test_period_ranges_present(self, change_result: ChangeResult) -> None:
        """DataFrame contains period ranges."""
        df = change_result.to_dataframe()
        assert "period_1_start" in df.columns
        assert "period_1_end" in df.columns
        assert "period_2_start" in df.columns
        assert "period_2_end" in df.columns

    def test_confidence_values(self, change_result: ChangeResult) -> None:
        """DataFrame contains both period confidences."""
        df = change_result.to_dataframe()
        assert "period_1_confidence" in df.columns
        assert "period_2_confidence" in df.columns
        assert "combined_confidence" in df.columns


# ── ChangeResult.to_png() tests ───────────────────────────────────────────────


class TestChangeResultToPng:
    """Tests for ChangeResult.to_png()."""

    def test_writes_file(self, change_result: ChangeResult) -> None:
        """to_png() writes a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "change.png"
            change_result.to_png(output_path)
            assert output_path.exists()


# ── WeatherResult.to_dataframe() tests ────────────────────────────────────────


class TestWeatherResultToDataFrame:
    """Tests for WeatherResult.to_dataframe()."""

    def test_returns_time_series(self, weather_result: WeatherResult) -> None:
        """to_dataframe() returns the time series DataFrame."""
        df = weather_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        # Should have 30 rows (one per day)
        assert len(df) == 30

    def test_contains_weather_columns(self, weather_result: WeatherResult) -> None:
        """DataFrame contains weather columns."""
        df = weather_result.to_dataframe()
        assert "temperature_mean" in df.columns
        assert "precipitation" in df.columns

    def test_empty_result_returns_summary(
        self, empty_weather_result: WeatherResult
    ) -> None:
        """Empty result returns summary DataFrame."""
        df = empty_weather_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Single row summary


# ── WeatherResult.to_geotiff() tests ──────────────────────────────────────────


class TestWeatherResultToGeoTiff:
    """Tests for WeatherResult.to_geotiff()."""

    def test_raises_value_error(self, weather_result: WeatherResult) -> None:
        """to_geotiff() raises ValueError for weather data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "weather.tif"
            with pytest.raises(ValueError, match="time-series data"):
                weather_result.to_geotiff(output_path)

    def test_error_message_is_helpful(self, weather_result: WeatherResult) -> None:
        """Error message suggests alternatives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "weather.tif"
            with pytest.raises(ValueError) as excinfo:
                weather_result.to_geotiff(output_path)
            assert "to_dataframe()" in str(excinfo.value)
            assert "to_png()" in str(excinfo.value)


# ── WeatherResult.to_png() tests ──────────────────────────────────────────────


class TestWeatherResultToPng:
    """Tests for WeatherResult.to_png()."""

    def test_writes_file(self, weather_result: WeatherResult) -> None:
        """to_png() writes a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "weather.png"
            weather_result.to_png(output_path)
            assert output_path.exists()

    def test_empty_result_handled(self, empty_weather_result: WeatherResult) -> None:
        """to_png() handles empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "weather.png"
            empty_weather_result.to_png(output_path)
            assert output_path.exists()


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases across all result types."""

    def test_2d_data_for_geotiff(self) -> None:
        """2D data (no bands dimension) works for GeoTIFF."""
        result = BaseResult(
            data=np.random.rand(100, 100).astype(np.float32),
            confidence=0.5,
            metadata=ResultMetadata(
                bounds={"minx": 0, "miny": 0, "maxx": 1, "maxy": 1}
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.tif"
            result.to_geotiff(output_path)
            assert output_path.exists()

    def test_no_bounds_metadata(self) -> None:
        """Results without bounds metadata still work."""
        result = BaseResult(
            data=np.random.rand(1, 50, 50).astype(np.float32),
            confidence=0.6,
            metadata=ResultMetadata(),  # No bounds
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # DataFrame should have None for lat/lon
            df = result.to_dataframe()
            assert df["latitude"].iloc[0] is None

            # GeoTIFF should use default transform
            tif_path = Path(tmpdir) / "output.tif"
            result.to_geotiff(tif_path)
            assert tif_path.exists()

            # PNG should still work
            png_path = Path(tmpdir) / "output.png"
            result.to_png(png_path)
            assert png_path.exists()

    def test_nan_values_in_dataframe(self) -> None:
        """NaN values are handled correctly in DataFrame exports."""
        result = VegetationResult(
            data=np.array([[float("nan"), 0.5]], dtype=np.float32),
            confidence=0.3,
            mean_ndvi=float("nan"),
            ndvi_std=float("nan"),
        )
        df = result.to_dataframe()
        assert math.isnan(df["mean_ndvi"].iloc[0])
        assert df["ndvi_interpretation"].iloc[0] == "no data"

    def test_change_result_with_nan_delta(self) -> None:
        """ChangeResult with NaN delta exports correctly."""
        result = ChangeResult(
            data=np.array([], dtype=np.float32),
            confidence=0.0,
            period_1_ndvi=float("nan"),
            period_2_ndvi=float("nan"),
            delta=float("nan"),
            direction="unknown",
        )
        df = result.to_dataframe()
        assert math.isnan(df["delta"].iloc[0])
        assert df["change_interpretation"].iloc[0] == "change could not be determined"

        # PNG should still work
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "change.png"
            result.to_png(png_path)
            assert png_path.exists()
