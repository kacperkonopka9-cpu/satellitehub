"""Top-level semantic API functions for SatelliteHub.

Provides convenient module-level functions for common analysis tasks.
These functions accept either (lat, lon) coordinates or a Location object,
making the API more ergonomic for quick scripts and notebooks.

Example:
    >>> import satellitehub as sh
    >>> # Simple: just pass coordinates
    >>> result = sh.vegetation_health(52.23, 21.01, last_days=30)
    >>> print(f"NDVI: {result.mean_ndvi:.2f}")
    >>>
    >>> # Or use a Location for multiple analyses
    >>> loc = sh.location(52.23, 21.01)
    >>> veg = sh.vegetation_health(loc, last_days=30)
    >>> weather = sh.weather(loc, last_days=7)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, overload

from satellitehub.location import Location
from satellitehub.location import location as create_location
from satellitehub.results import ChangeResult, VegetationResult, WeatherResult

if TYPE_CHECKING:
    from satellitehub.config import Config


def _resolve_location(
    loc_or_lat: Location | float,
    lon: float | None = None,
    config: Config | None = None,
) -> Location:
    """Resolve input to a Location object.

    Args:
        loc_or_lat: Either a Location object or latitude float.
        lon: Longitude (required if loc_or_lat is a float).
        config: Optional config override.

    Returns:
        A Location object.

    Raises:
        TypeError: If loc_or_lat is a float but lon is not provided.
    """
    if isinstance(loc_or_lat, Location):
        return loc_or_lat
    if lon is None:
        raise TypeError(
            "longitude is required when first argument is latitude. "
            "Use: vegetation_health(lat, lon) or vegetation_health(location)"
        )
    return create_location(loc_or_lat, lon, config=config)


def _days_to_date_range(
    start_days_ago: int,
    end_days_ago: int,
) -> tuple[str, str]:
    """Convert days-ago values to ISO date strings.

    Args:
        start_days_ago: Start of period (days before today).
        end_days_ago: End of period (days before today).

    Returns:
        Tuple of (start_date, end_date) as ISO strings.
    """
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=start_days_ago)).strftime("%Y-%m-%d")
    end = (now - timedelta(days=end_days_ago)).strftime("%Y-%m-%d")
    return (start, end)


# ── Vegetation Health ─────────────────────────────────────────────────


@overload
def vegetation_health(
    loc_or_lat: Location,
    lon: None = None,
    *,
    last_days: int = ...,
    config: None = None,
) -> VegetationResult: ...


@overload
def vegetation_health(
    loc_or_lat: float,
    lon: float,
    *,
    last_days: int = ...,
    config: Config | None = None,
) -> VegetationResult: ...


def vegetation_health(
    loc_or_lat: Location | float,
    lon: float | None = None,
    *,
    last_days: int = 30,
    config: Config | None = None,
) -> VegetationResult:
    """Compute vegetation health (NDVI) for a location.

    Retrieves Sentinel-2 imagery, applies cloud masking, and computes
    the Normalized Difference Vegetation Index (NDVI).

    Args:
        loc_or_lat: A Location object, or latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees (required if first arg is latitude).
        last_days: Number of days to look back. Defaults to 30.
        config: Optional configuration override.

    Returns:
        VegetationResult with NDVI values, confidence, and metadata.

    Example:
        >>> import satellitehub as sh
        >>> # Using coordinates directly
        >>> result = sh.vegetation_health(52.23, 21.01, last_days=30)
        >>> print(f"Mean NDVI: {result.mean_ndvi:.2f}")
        >>>
        >>> # Using a Location object
        >>> loc = sh.location(52.23, 21.01)
        >>> result = sh.vegetation_health(loc, last_days=60)
    """
    location = _resolve_location(loc_or_lat, lon, config)
    return location.vegetation_health(last_days=last_days)


# ── Vegetation Change Detection ───────────────────────────────────────


@overload
def change_detection(
    loc_or_lat: Location,
    lon: None = None,
    *,
    period1: tuple[str, str] | None = None,
    period2: tuple[str, str] | None = None,
    period1_days: tuple[int, int] | None = None,
    period2_days: tuple[int, int] | None = None,
    config: None = None,
) -> ChangeResult: ...


@overload
def change_detection(
    loc_or_lat: float,
    lon: float,
    *,
    period1: tuple[str, str] | None = None,
    period2: tuple[str, str] | None = None,
    period1_days: tuple[int, int] | None = None,
    period2_days: tuple[int, int] | None = None,
    config: Config | None = None,
) -> ChangeResult: ...


def change_detection(
    loc_or_lat: Location | float,
    lon: float | None = None,
    *,
    period1: tuple[str, str] | None = None,
    period2: tuple[str, str] | None = None,
    period1_days: tuple[int, int] | None = None,
    period2_days: tuple[int, int] | None = None,
    config: Config | None = None,
) -> ChangeResult:
    """Detect vegetation changes between two time periods.

    Compares NDVI values from two time periods to identify areas of
    vegetation change (growth, decline, or stability).

    You can specify periods either as ISO date strings or as days-ago tuples.
    If neither is specified, defaults to comparing last 30 days vs previous 30 days.

    Args:
        loc_or_lat: A Location object, or latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees (required if first arg is latitude).
        period1: Baseline period as (start_date, end_date) ISO strings.
        period2: Comparison period as (start_date, end_date) ISO strings.
        period1_days: Baseline as (start_days_ago, end_days_ago).
            E.g., (60, 30) means 60-30 days ago.
        period2_days: Comparison as (start_days_ago, end_days_ago).
            E.g., (30, 0) means last 30 days.
        config: Optional configuration override.

    Returns:
        ChangeResult with change magnitude, direction, and confidence.

    Example:
        >>> import satellitehub as sh
        >>> # Compare last month to previous month (using days)
        >>> result = sh.change_detection(
        ...     52.23, 21.01,
        ...     period1_days=(60, 30),
        ...     period2_days=(30, 0),
        ... )
        >>> print(f"Change: {result.delta:+.2f}")
        >>>
        >>> # Or use explicit dates
        >>> result = sh.change_detection(
        ...     52.23, 21.01,
        ...     period1=("2025-06-01", "2025-06-30"),
        ...     period2=("2026-06-01", "2026-06-30"),
        ... )
    """
    location = _resolve_location(loc_or_lat, lon, config)

    # Resolve period1
    if period1 is not None:
        p1 = period1
    elif period1_days is not None:
        p1 = _days_to_date_range(period1_days[0], period1_days[1])
    else:
        # Default: 60-30 days ago
        p1 = _days_to_date_range(60, 30)

    # Resolve period2
    if period2 is not None:
        p2 = period2
    elif period2_days is not None:
        p2 = _days_to_date_range(period2_days[0], period2_days[1])
    else:
        # Default: last 30 days
        p2 = _days_to_date_range(30, 0)

    return location.vegetation_change(period_1=p1, period_2=p2)


# ── Weather ───────────────────────────────────────────────────────────


@overload
def weather(
    loc_or_lat: Location,
    lon: None = None,
    *,
    last_days: int = ...,
    config: None = None,
) -> WeatherResult: ...


@overload
def weather(
    loc_or_lat: float,
    lon: float,
    *,
    last_days: int = ...,
    config: Config | None = None,
) -> WeatherResult: ...


def weather(
    loc_or_lat: Location | float,
    lon: float | None = None,
    *,
    last_days: int = 7,
    config: Config | None = None,
) -> WeatherResult:
    """Get weather summary for a location.

    Retrieves temperature and precipitation data from ERA5 reanalysis
    (global) and IMGW stations (Poland only, when in range).

    Args:
        loc_or_lat: A Location object, or latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees (required if first arg is latitude).
        last_days: Number of days to look back. Defaults to 7.
        config: Optional configuration override.

    Returns:
        WeatherResult with temperature and precipitation statistics.

    Example:
        >>> import satellitehub as sh
        >>> result = sh.weather(52.23, 21.01, last_days=7)
        >>> print(f"Mean temp: {result.mean_temperature:.1f}C")
    """
    location = _resolve_location(loc_or_lat, lon, config)
    return location.weather(last_days=last_days)
