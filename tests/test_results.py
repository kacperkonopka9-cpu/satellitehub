"""Tests for the result object model (BaseResult, ResultMetadata, VegetationResult)."""

import math
import time

import numpy as np
import pytest

from satellitehub.results import (
    BaseResult,
    ChangeResult,
    ResultMetadata,
    VegetationResult,
    _interpret_change,
    _interpret_ndvi,
)


class TestResultMetadata:
    """Tests for the ResultMetadata pydantic model."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        meta = ResultMetadata()
        assert meta.source == ""
        assert meta.timestamps == []
        assert meta.observation_count == 0
        assert meta.cloud_cover_pct == 0.0
        assert meta.crs == ""
        assert meta.bounds == {}
        assert meta.resolution_m is None
        assert meta.bands == []

    @pytest.mark.unit
    def test_construction_with_values(self) -> None:
        meta = ResultMetadata(
            source="cdse",
            timestamps=["2024-01-01T10:00:00Z"],
            observation_count=3,
            cloud_cover_pct=15.5,
            crs="EPSG:32634",
            bounds={"minx": 22.0, "miny": 51.0, "maxx": 23.0, "maxy": 52.0},
            resolution_m=10.0,
            bands=["B04", "B08"],
        )
        assert meta.source == "cdse"
        assert meta.observation_count == 3
        assert meta.cloud_cover_pct == 15.5
        assert meta.crs == "EPSG:32634"
        assert meta.resolution_m == 10.0
        assert meta.bands == ["B04", "B08"]
        assert len(meta.timestamps) == 1

    @pytest.mark.unit
    def test_json_serialization_round_trip(self) -> None:
        meta = ResultMetadata(
            source="cdse",
            observation_count=5,
            bands=["B04"],
        )
        dumped = meta.model_dump_json()
        restored = ResultMetadata.model_validate_json(dumped)
        assert restored == meta


class TestBaseResult:
    """Tests for the BaseResult dataclass."""

    @pytest.mark.unit
    def test_construction_with_data(self) -> None:
        data = np.zeros((2, 100, 100), dtype=np.float32)
        result = BaseResult(data=data, confidence=0.85)
        assert result.confidence == 0.85
        assert result.data.shape == (2, 100, 100)
        assert isinstance(result.metadata, ResultMetadata)
        assert result.warnings == []

    @pytest.mark.unit
    def test_defaults(self) -> None:
        data = np.array([], dtype=np.float32)
        result = BaseResult(data=data)
        assert result.confidence == 0.0
        assert result.metadata.source == ""
        assert result.warnings == []

    @pytest.mark.unit
    def test_empty_data_zero_confidence(self) -> None:
        result = BaseResult(
            data=np.array([], dtype=np.float32),
            confidence=0.0,
            warnings=["No data found for requested location and time range"],
        )
        assert result.confidence == 0.0
        assert len(result.warnings) == 1
        assert result.data.size == 0

    @pytest.mark.unit
    def test_repr_without_warnings(self) -> None:
        meta = ResultMetadata(observation_count=6)
        result = BaseResult(
            data=np.zeros((2, 10, 10), dtype=np.float32),
            confidence=0.78,
            metadata=meta,
        )
        r = repr(result)
        assert "BaseResult(" in r
        assert "confidence=0.78" in r
        assert "observations=6" in r
        assert "warnings" not in r

    @pytest.mark.unit
    def test_repr_with_warnings(self) -> None:
        result = BaseResult(
            data=np.array([], dtype=np.float32),
            confidence=0.0,
            warnings=["warn1", "warn2"],
        )
        r = repr(result)
        assert "warnings=2" in r

    @pytest.mark.unit
    def test_repr_does_not_show_raw_arrays(self) -> None:
        data = np.ones((3, 50, 50), dtype=np.float32)
        result = BaseResult(data=data, confidence=0.9)
        r = repr(result)
        # Should not contain array values or numpy repr
        assert "array(" not in r.lower()
        assert "1." not in r  # no float pixel values

    @pytest.mark.unit
    def test_data_type_is_numpy_array(self) -> None:
        data = np.zeros((1, 10, 10), dtype=np.float64)
        result = BaseResult(data=data)
        assert isinstance(result.data, np.ndarray)
        assert np.issubdtype(result.data.dtype, np.floating)

    @pytest.mark.unit
    def test_metadata_is_result_metadata(self) -> None:
        result = BaseResult(data=np.array([], dtype=np.float32))
        assert isinstance(result.metadata, ResultMetadata)

    @pytest.mark.unit
    def test_warnings_is_mutable_list(self) -> None:
        result = BaseResult(data=np.array([], dtype=np.float32))
        result.warnings.append("test warning")
        assert len(result.warnings) == 1


# ── _interpret_ndvi tests ────────────────────────────────────────


@pytest.mark.unit
class TestInterpretNdvi:
    """Tests for the _interpret_ndvi() helper function."""

    def test_healthy_vegetation(self) -> None:
        assert _interpret_ndvi(0.7) == "healthy vegetation"

    def test_moderate_vegetation(self) -> None:
        assert _interpret_ndvi(0.45) == "moderate vegetation"

    def test_sparse_stressed_vegetation(self) -> None:
        assert _interpret_ndvi(0.2) == "sparse/stressed vegetation"

    def test_bare_soil_water(self) -> None:
        assert _interpret_ndvi(0.05) == "bare soil/water"

    def test_boundary_healthy(self) -> None:
        assert _interpret_ndvi(0.6) == "healthy vegetation"

    def test_boundary_moderate(self) -> None:
        assert _interpret_ndvi(0.3) == "moderate vegetation"

    def test_boundary_sparse(self) -> None:
        assert _interpret_ndvi(0.1) == "sparse/stressed vegetation"

    def test_nan_returns_no_data(self) -> None:
        assert _interpret_ndvi(float("nan")) == "no data"

    def test_negative_value(self) -> None:
        assert _interpret_ndvi(-0.1) == "bare soil/water"


# ── VegetationResult tests ───────────────────────────────────────


@pytest.mark.unit
class TestVegetationResult:
    """Tests for VegetationResult dataclass construction."""

    def test_construction_with_all_fields(self) -> None:
        meta = ResultMetadata(
            source="cdse",
            timestamps=["2026-01-05T10:00:00Z", "2026-02-04T10:00:00Z"],
            observation_count=6,
            cloud_cover_pct=20.0,
            bounds={"minx": 22.0, "miny": 51.0, "maxx": 23.0, "maxy": 52.0},
        )
        result = VegetationResult(
            data=np.ones((10, 10), dtype=np.float32) * 0.42,
            confidence=0.78,
            metadata=meta,
            warnings=["2 passes excluded (cloud cover >80%)"],
            mean_ndvi=0.42,
            ndvi_std=0.08,
            trend=None,
            observation_count=6,
            cloud_free_count=4,
        )
        assert result.confidence == 0.78
        assert result.mean_ndvi == 0.42
        assert result.ndvi_std == 0.08
        assert result.trend is None
        assert result.observation_count == 6
        assert result.cloud_free_count == 4
        assert len(result.warnings) == 1

    def test_defaults(self) -> None:
        result = VegetationResult(data=np.array([], dtype=np.float32))
        assert result.confidence == 0.0
        assert math.isnan(result.mean_ndvi)
        assert math.isnan(result.ndvi_std)
        assert result.trend is None
        assert result.observation_count == 0
        assert result.cloud_free_count == 0
        assert result.warnings == []
        assert isinstance(result.metadata, ResultMetadata)

    def test_inherits_base_result(self) -> None:
        result = VegetationResult(data=np.array([], dtype=np.float32))
        assert isinstance(result, BaseResult)

    def test_public_import(self) -> None:
        from satellitehub import VegetationResult as VegResult

        assert VegResult is VegetationResult


@pytest.mark.unit
class TestVegetationResultRepr:
    """Tests for VegetationResult.__repr__ narrative display."""

    def _make_result(
        self,
        *,
        mean_ndvi: float = 0.42,
        ndvi_std: float = 0.08,
        confidence: float = 0.78,
        observation_count: int = 6,
        cloud_free_count: int = 4,
        trend: float | None = None,
        warnings: list[str] | None = None,
        bounds: dict[str, float] | None = None,
        timestamps: list[str] | None = None,
    ) -> VegetationResult:
        if bounds is None:
            bounds = {"minx": 22.0, "miny": 51.0, "maxx": 23.0, "maxy": 52.0}
        if timestamps is None:
            timestamps = ["2026-01-05T10:00:00Z", "2026-02-04T10:00:00Z"]
        if warnings is None:
            warnings = []
        meta = ResultMetadata(
            source="cdse",
            timestamps=timestamps,
            observation_count=observation_count,
            bounds=bounds,
        )
        return VegetationResult(
            data=np.ones((10, 10), dtype=np.float32) * mean_ndvi,
            confidence=confidence,
            metadata=meta,
            warnings=warnings,
            mean_ndvi=mean_ndvi,
            ndvi_std=ndvi_std,
            trend=trend,
            observation_count=observation_count,
            cloud_free_count=cloud_free_count,
        )

    def test_repr_contains_class_name(self) -> None:
        r = repr(self._make_result())
        assert r.startswith("VegetationResult(")

    def test_repr_contains_location(self) -> None:
        r = repr(self._make_result())
        assert "location:" in r
        assert "51.50" in r  # center lat
        assert "22.50" in r  # center lon
        assert "N" in r
        assert "E" in r

    def test_repr_contains_period(self) -> None:
        r = repr(self._make_result())
        assert "period:" in r
        assert "2026-01-05" in r
        assert "2026-02-04" in r

    def test_repr_contains_observations(self) -> None:
        r = repr(self._make_result())
        assert "4 of 6 passes (cloud-free)" in r

    def test_repr_contains_confidence(self) -> None:
        r = repr(self._make_result())
        assert "confidence: 0.78" in r

    def test_repr_contains_mean_ndvi_with_interpretation(self) -> None:
        r = repr(self._make_result())
        assert "mean_ndvi: 0.42" in r
        assert "0.08" in r
        assert "moderate vegetation" in r

    def test_repr_with_warnings(self) -> None:
        r = repr(
            self._make_result(
                warnings=["2 passes excluded (cloud cover >80%)"],
            )
        )
        assert "\u26a0" in r  # warning symbol
        assert "2 passes excluded" in r

    def test_repr_without_warnings_has_no_warning_lines(self) -> None:
        r = repr(self._make_result(warnings=[]))
        assert "\u26a0" not in r

    def test_repr_with_nan_mean_ndvi(self) -> None:
        r = repr(
            self._make_result(
                mean_ndvi=float("nan"),
                ndvi_std=float("nan"),
                confidence=0.0,
            )
        )
        assert "N/A (no valid data)" in r

    def test_repr_with_trend(self) -> None:
        r = repr(self._make_result(trend=-0.06))
        assert "declining" in r
        assert "-0.06" in r

    def test_repr_with_positive_trend(self) -> None:
        r = repr(self._make_result(trend=0.04))
        assert "improving" in r
        assert "+0.04" in r

    def test_repr_with_zero_trend(self) -> None:
        r = repr(self._make_result(trend=0.0))
        assert "stable" in r
        assert "+0.00" in r

    def test_repr_southern_western_hemisphere(self) -> None:
        # Argentina/South America coordinates
        r = repr(
            self._make_result(
                bounds={"minx": -70.0, "miny": -35.0, "maxx": -69.0, "maxy": -34.0}
            )
        )
        assert "location:" in r
        assert "34.50" in r  # center lat
        assert "69.50" in r  # center lon
        assert "S" in r
        assert "W" in r

    def test_repr_partial_bounds_omits_location(self) -> None:
        # Partial bounds (missing keys) should not display location
        r = repr(self._make_result(bounds={"minx": 22.0, "miny": 51.0}))
        assert "location:" not in r

    def test_repr_without_trend(self) -> None:
        r = repr(self._make_result(trend=None))
        assert "N/A (single period)" in r

    def test_repr_does_not_show_raw_arrays(self) -> None:
        r = repr(self._make_result())
        assert "array(" not in r.lower()

    def test_repr_without_bounds_omits_location(self) -> None:
        r = repr(self._make_result(bounds={}))
        assert "location:" not in r

    def test_repr_without_timestamps_omits_period(self) -> None:
        r = repr(self._make_result(timestamps=[]))
        assert "period:" not in r

    def test_repr_performance_under_200ms(self) -> None:
        result = self._make_result()
        start = time.perf_counter()
        for _ in range(100):
            repr(result)
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.200

    def test_repr_reproducibility(self) -> None:
        r1 = repr(self._make_result())
        r2 = repr(self._make_result())
        assert r1 == r2


# ── _interpret_change tests ────────────────────────────────────────


@pytest.mark.unit
class TestInterpretChange:
    """Tests for the _interpret_change() helper function."""

    def test_significant_improvement(self) -> None:
        assert _interpret_change(0.20) == "significant vegetation improvement detected"

    def test_improvement(self) -> None:
        assert _interpret_change(0.08) == "vegetation improvement detected"

    def test_stable_positive(self) -> None:
        assert _interpret_change(0.01) == "vegetation stable (no significant change)"

    def test_stable_negative(self) -> None:
        assert _interpret_change(-0.01) == "vegetation stable (no significant change)"

    def test_stable_zero(self) -> None:
        assert _interpret_change(0.0) == "vegetation stable (no significant change)"

    def test_decline(self) -> None:
        assert _interpret_change(-0.08) == "vegetation decline detected"

    def test_significant_decline(self) -> None:
        assert _interpret_change(-0.20) == "significant vegetation decline detected"

    def test_nan_returns_undetermined(self) -> None:
        assert _interpret_change(float("nan")) == "change could not be determined"

    def test_boundary_stable_threshold(self) -> None:
        # At exactly 0.02, should be stable (abs < 0.02 is stable)
        assert "stable" in _interpret_change(0.019)
        # Just above threshold
        assert "improvement" in _interpret_change(0.021)

    def test_boundary_significant_threshold(self) -> None:
        # At exactly 0.15, should be "improvement" not "significant"
        assert _interpret_change(0.15) == "vegetation improvement detected"
        # Just above threshold
        assert "significant" in _interpret_change(0.16)


# ── ChangeResult tests ───────────────────────────────────────


@pytest.mark.unit
class TestChangeResult:
    """Tests for ChangeResult dataclass construction."""

    def test_construction_with_all_fields(self) -> None:
        result = ChangeResult(
            data=np.array([], dtype=np.float32),
            confidence=0.65,
            period_1_ndvi=0.42,
            period_2_ndvi=0.27,
            delta=-0.15,
            direction="declining",
            period_1_confidence=0.78,
            period_2_confidence=0.65,
            period_1_range=("2025-01-01", "2025-01-31"),
            period_2_range=("2026-01-01", "2026-01-31"),
        )
        assert result.confidence == 0.65
        assert result.period_1_ndvi == 0.42
        assert result.period_2_ndvi == 0.27
        assert result.delta == -0.15
        assert result.direction == "declining"
        assert result.period_1_confidence == 0.78
        assert result.period_2_confidence == 0.65
        assert result.period_1_range == ("2025-01-01", "2025-01-31")
        assert result.period_2_range == ("2026-01-01", "2026-01-31")

    def test_defaults(self) -> None:
        result = ChangeResult(data=np.array([], dtype=np.float32))
        assert result.confidence == 0.0
        assert math.isnan(result.period_1_ndvi)
        assert math.isnan(result.period_2_ndvi)
        assert math.isnan(result.delta)
        assert result.direction == "unknown"
        assert result.period_1_confidence == 0.0
        assert result.period_2_confidence == 0.0
        assert result.period_1_range == ("", "")
        assert result.period_2_range == ("", "")

    def test_inherits_base_result(self) -> None:
        result = ChangeResult(data=np.array([], dtype=np.float32))
        assert isinstance(result, BaseResult)

    def test_public_import(self) -> None:
        from satellitehub import ChangeResult as ImportedChangeResult

        assert ImportedChangeResult is ChangeResult


@pytest.mark.unit
class TestChangeResultRepr:
    """Tests for ChangeResult.__repr__ narrative display."""

    def _make_result(
        self,
        *,
        period_1_ndvi: float = 0.42,
        period_2_ndvi: float = 0.27,
        delta: float = -0.15,
        direction: str = "declining",
        confidence: float = 0.65,
        period_1_confidence: float = 0.78,
        period_2_confidence: float = 0.65,
        period_1_range: tuple[str, str] = ("2025-01-01", "2025-01-31"),
        period_2_range: tuple[str, str] = ("2026-01-01", "2026-01-31"),
        warnings: list[str] | None = None,
    ) -> ChangeResult:
        if warnings is None:
            warnings = []
        return ChangeResult(
            data=np.array([], dtype=np.float32),
            confidence=confidence,
            warnings=warnings,
            period_1_ndvi=period_1_ndvi,
            period_2_ndvi=period_2_ndvi,
            delta=delta,
            direction=direction,
            period_1_confidence=period_1_confidence,
            period_2_confidence=period_2_confidence,
            period_1_range=period_1_range,
            period_2_range=period_2_range,
        )

    def test_repr_contains_class_name(self) -> None:
        r = repr(self._make_result())
        assert r.startswith("ChangeResult(")

    def test_repr_contains_period_1(self) -> None:
        r = repr(self._make_result())
        assert "period_1:" in r
        assert "2025-01-01" in r
        assert "2025-01-31" in r
        assert "NDVI: 0.42" in r
        assert "confidence: 0.78" in r

    def test_repr_contains_period_2(self) -> None:
        r = repr(self._make_result())
        assert "period_2:" in r
        assert "2026-01-01" in r
        assert "2026-01-31" in r
        assert "NDVI: 0.27" in r
        assert "confidence: 0.65" in r

    def test_repr_contains_change_delta(self) -> None:
        r = repr(self._make_result())
        assert "change:" in r
        assert "-0.15" in r

    def test_repr_contains_change_interpretation(self) -> None:
        r = repr(self._make_result())
        # Should contain the interpretation from _interpret_change
        assert "decline" in r.lower()

    def test_repr_contains_combined_confidence(self) -> None:
        r = repr(self._make_result())
        assert "combined_confidence: 0.65" in r

    def test_repr_with_warnings(self) -> None:
        r = repr(
            self._make_result(
                warnings=["Period 1 (2025-01-01 to 2025-01-31): insufficient data"],
            )
        )
        assert "\u26a0" in r  # warning symbol
        assert "insufficient data" in r

    def test_repr_without_warnings_has_no_warning_lines(self) -> None:
        r = repr(self._make_result(warnings=[]))
        assert "\u26a0" not in r

    def test_repr_with_nan_ndvi(self) -> None:
        r = repr(
            self._make_result(
                period_1_ndvi=float("nan"),
                period_2_ndvi=0.30,
                delta=float("nan"),
            )
        )
        assert "N/A" in r

    def test_repr_with_nan_delta(self) -> None:
        r = repr(self._make_result(delta=float("nan")))
        assert "insufficient data" in r.lower() or "n/a" in r.lower()

    def test_repr_improving_scenario(self) -> None:
        r = repr(
            self._make_result(
                period_1_ndvi=0.30,
                period_2_ndvi=0.50,
                delta=0.20,
                direction="improving",
            )
        )
        assert "+0.20" in r
        assert "improvement" in r.lower()

    def test_repr_stable_scenario(self) -> None:
        r = repr(
            self._make_result(
                period_1_ndvi=0.40,
                period_2_ndvi=0.41,
                delta=0.01,
                direction="stable",
            )
        )
        assert "stable" in r.lower()

    def test_repr_does_not_show_raw_arrays(self) -> None:
        r = repr(self._make_result())
        assert "array(" not in r.lower()

    def test_repr_performance_under_200ms(self) -> None:
        result = self._make_result()
        start = time.perf_counter()
        for _ in range(100):
            repr(result)
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.200

    def test_repr_reproducibility(self) -> None:
        r1 = repr(self._make_result())
        r2 = repr(self._make_result())
        assert r1 == r2
