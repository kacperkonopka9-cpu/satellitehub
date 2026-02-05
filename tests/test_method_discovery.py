"""Unit tests for method discovery API (Story 5.2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from satellitehub._types import MethodInfo
from satellitehub.config import Config
from satellitehub.location import Location, location

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config() -> Config:
    """Create a mock Config for testing."""
    return Config()


@pytest.fixture
def test_location(mock_config: Config) -> Location:
    """Create a test location."""
    return location(lat=52.23, lon=21.01, config=mock_config)


# ── MethodInfo tests ──────────────────────────────────────────────────────────


class TestMethodInfo:
    """Tests for MethodInfo dataclass."""

    def test_method_info_available(self) -> None:
        """MethodInfo shows available status correctly."""
        info = MethodInfo(
            name="vegetation_health",
            description="Compute NDVI",
            data_sources=["cdse"],
            available=True,
        )
        assert info.available is True
        assert info.unavailable_reason == ""
        assert "\u2713" in repr(info)  # Checkmark

    def test_method_info_unavailable(self) -> None:
        """MethodInfo shows unavailable status with reason."""
        info = MethodInfo(
            name="weather",
            description="Get weather data",
            data_sources=["cds"],
            available=False,
            unavailable_reason="CDS not configured",
        )
        assert info.available is False
        assert info.unavailable_reason == "CDS not configured"
        assert "\u2717" in repr(info)  # X mark
        assert "CDS not configured" in repr(info)

    def test_method_info_repr_format(self) -> None:
        """MethodInfo repr has expected format."""
        info = MethodInfo(
            name="test_method",
            description="Test description",
            data_sources=["source1", "source2"],
            available=True,
        )
        repr_str = repr(info)
        assert "test_method()" in repr_str
        assert "Test description" in repr_str
        assert "source1, source2" in repr_str


# ── Location.available_methods() tests ────────────────────────────────────────


class TestAvailableMethods:
    """Tests for Location.available_methods()."""

    def test_returns_list_of_method_info(self, test_location: Location) -> None:
        """available_methods() returns a list of MethodInfo objects."""
        methods = test_location.available_methods()
        assert isinstance(methods, list)
        assert all(isinstance(m, MethodInfo) for m in methods)

    def test_contains_core_methods(self, test_location: Location) -> None:
        """available_methods() lists vegetation_health, vegetation_change, weather."""
        methods = test_location.available_methods()
        method_names = [m.name for m in methods]
        assert "vegetation_health" in method_names
        assert "vegetation_change" in method_names
        assert "weather" in method_names

    def test_vegetation_methods_require_cdse(self, test_location: Location) -> None:
        """Vegetation methods require cdse data source."""
        methods = test_location.available_methods()
        veg_health = next(m for m in methods if m.name == "vegetation_health")
        veg_change = next(m for m in methods if m.name == "vegetation_change")

        assert "cdse" in veg_health.data_sources
        assert "cdse" in veg_change.data_sources

    def test_weather_requires_cds(self, test_location: Location) -> None:
        """Weather method requires cds data source."""
        methods = test_location.available_methods()
        weather = next(m for m in methods if m.name == "weather")
        assert "cds" in weather.data_sources

    def test_unavailable_when_no_copernicus_credentials(
        self, test_location: Location
    ) -> None:
        """Methods marked unavailable when Copernicus credentials missing."""
        with patch(
            "satellitehub.config.resolve_credentials_path",
            return_value=None,
        ):
            methods = test_location.available_methods()

        veg_health = next(m for m in methods if m.name == "vegetation_health")
        assert veg_health.available is False
        assert "Copernicus" in veg_health.unavailable_reason

    def test_available_when_credentials_configured(
        self, test_location: Location
    ) -> None:
        """Methods marked available when credentials are configured."""
        with patch(
            "satellitehub.config.resolve_credentials_path",
            return_value=Path("/fake/credentials.json"),
        ):
            methods = test_location.available_methods()

        veg_health = next(m for m in methods if m.name == "vegetation_health")
        assert veg_health.available is True
        assert veg_health.unavailable_reason == ""

    def test_weather_unavailable_without_cds(
        self, test_location: Location
    ) -> None:
        """Weather marked unavailable when CDS credentials missing."""
        # Both copernicus and cds return None
        with patch(
            "satellitehub.config.resolve_credentials_path",
            return_value=None,
        ):
            methods = test_location.available_methods()

        weather = next(m for m in methods if m.name == "weather")
        assert weather.available is False
        assert "CDS" in weather.unavailable_reason

    def test_methods_have_descriptions(self, test_location: Location) -> None:
        """All methods have non-empty descriptions."""
        methods = test_location.available_methods()
        for method in methods:
            assert method.description
            assert len(method.description) > 10  # Meaningful description

    def test_methods_have_data_sources(self, test_location: Location) -> None:
        """All methods list their data sources."""
        methods = test_location.available_methods()
        for method in methods:
            assert method.data_sources
            assert len(method.data_sources) >= 1


# ── Jupyter display tests ─────────────────────────────────────────────────────


class TestMethodInfoDisplay:
    """Tests for MethodInfo display formatting."""

    def test_repr_is_readable(self, test_location: Location) -> None:
        """Method repr is human-readable for terminal/Jupyter display."""
        methods = test_location.available_methods()
        for method in methods:
            repr_str = repr(method)
            # Should contain method name with parentheses
            assert f"{method.name}()" in repr_str
            # Should contain description
            assert method.description in repr_str
            # Should contain sources
            assert "sources:" in repr_str

    def test_multiple_methods_display(self, test_location: Location) -> None:
        """Multiple methods display nicely when printed."""
        methods = test_location.available_methods()
        # Each method should have a distinct repr
        reprs = [repr(m) for m in methods]
        assert len(reprs) == len(set(reprs))  # All unique
