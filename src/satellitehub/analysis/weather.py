"""Weather data aggregation and processing.

This module provides functions for aggregating raw weather data from ERA5
and IMGW providers into daily summaries, and building WeatherResult objects.

Example:
    >>> from satellitehub.analysis.weather import build_weather_result
    >>> result = build_weather_result(
    ...     era5_data=era5_raw,
    ...     imgw_data=imgw_raw,
    ...     location=loc,
    ...     time_range=("2024-01-01", "2024-01-31"),
    ... )
    >>> result.mean_temperature
    4.2
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from satellitehub.results import ResultMetadata, WeatherResult

if TYPE_CHECKING:
    from satellitehub._types import RawData
    from satellitehub.location import Location

logger = logging.getLogger(__name__)


def aggregate_era5_daily(raw_data: RawData) -> pd.DataFrame:
    """Aggregate ERA5 hourly/sub-daily data to daily summaries.

    ERA5 data typically has 4 timestamps per day (00:00, 06:00, 12:00, 18:00)
    or hourly resolution. This function aggregates to daily min/max/mean
    temperature and total precipitation.

    Args:
        raw_data: RawData from CDSProvider.download() with data array
            of shape (time, variables) containing temperature and precipitation.
            Metadata should include 'timestamps' and 'variables' lists.

    Returns:
        DataFrame with columns: timestamp (date string), temperature_min,
        temperature_max, temperature_mean, precipitation, source.
        Returns empty DataFrame if no valid data.

    Example:
        >>> df = aggregate_era5_daily(era5_raw)
        >>> df.columns.tolist()
        ['timestamp', 'temperature_min', 'temperature_max', 'temperature_mean',
         'precipitation', 'source']
    """
    data_array = raw_data.data
    metadata = raw_data.metadata

    # Define empty DataFrame columns for consistent returns
    empty_columns = [
        "timestamp",
        "temperature_min",
        "temperature_max",
        "temperature_mean",
        "precipitation",
        "source",
    ]

    # Check for required data
    if data_array.size == 0:
        logger.warning("No ERA5 data to aggregate")
        return pd.DataFrame(columns=empty_columns)

    # Extract timestamps and variables from metadata
    timestamps: list[str] = metadata.get("timestamps", [])
    variables: list[str] = metadata.get("variables", [])

    if not timestamps:
        logger.warning("No timestamps in ERA5 metadata")
        return pd.DataFrame(columns=empty_columns)

    # Find variable indices
    temp_idx: int | None = None
    precip_idx: int | None = None

    for idx, var_name in enumerate(variables):
        if var_name in ("2m_temperature", "t2m", "2t"):
            temp_idx = idx
        elif var_name in ("total_precipitation", "tp"):
            precip_idx = idx

    # Parse timestamps into datetime objects and extract dates
    parsed_times: list[tuple[str, datetime]] = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            parsed_times.append((date_str, dt))
        except (ValueError, AttributeError):
            continue

    if not parsed_times:
        return pd.DataFrame(columns=empty_columns)

    # Group data by date
    # Data shape: (time, variables)
    date_groups: dict[str, list[int]] = {}
    for idx, (date_str, _) in enumerate(parsed_times):
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(idx)

    # Create daily records
    daily_records: list[dict[str, Any]] = []

    for date_str in sorted(date_groups.keys()):
        indices = date_groups[date_str]

        # Extract temperature values for this date
        if temp_idx is not None and data_array.ndim >= 2:
            temp_values = data_array[indices, temp_idx]
            valid_temps = temp_values[np.isfinite(temp_values)]
            if len(valid_temps) > 0:
                temp_min = float(np.min(valid_temps))
                temp_max = float(np.max(valid_temps))
                temp_mean = float(np.mean(valid_temps))
            else:
                temp_min = temp_max = temp_mean = float("nan")
        elif data_array.ndim == 1:
            # 1D array fallback (single variable)
            temp_values = data_array[indices]
            valid_temps = temp_values[np.isfinite(temp_values)]
            if len(valid_temps) > 0:
                temp_min = float(np.min(valid_temps))
                temp_max = float(np.max(valid_temps))
                temp_mean = float(np.mean(valid_temps))
            else:
                temp_min = temp_max = temp_mean = float("nan")
        else:
            temp_min = temp_max = temp_mean = float("nan")

        # Extract precipitation values for this date (sum for daily total)
        if precip_idx is not None and data_array.ndim >= 2:
            precip_values = data_array[indices, precip_idx]
            valid_precip = precip_values[np.isfinite(precip_values)]
            if len(valid_precip) > 0:
                # ERA5 precipitation is in meters; convert to mm
                precip_total = float(np.sum(valid_precip)) * 1000.0
            else:
                precip_total = 0.0
        else:
            precip_total = 0.0

        daily_records.append(
            {
                "timestamp": date_str,
                "temperature_min": temp_min,
                "temperature_max": temp_max,
                "temperature_mean": temp_mean,
                "precipitation": precip_total,
                "source": "era5",
            }
        )

    return pd.DataFrame(daily_records)


def aggregate_imgw_daily(raw_data: RawData) -> pd.DataFrame:
    """Convert IMGW point measurements to daily format.

    IMGW provides current synoptic observations. This function formats
    them into the same daily structure as ERA5 data for merging.

    Args:
        raw_data: RawData from IMGWProvider.download() with station
            measurements in metadata.

    Returns:
        DataFrame with columns: timestamp (date string), temperature_min,
        temperature_max, temperature_mean, precipitation, source.
        Returns empty DataFrame if no valid data.

    Example:
        >>> df = aggregate_imgw_daily(imgw_raw)
        >>> df["source"].iloc[0]
        'imgw'
    """
    metadata = raw_data.metadata

    # Check for IMGW measurements in metadata
    measurements = metadata.get("measurements", {})
    if not measurements:
        logger.warning("No IMGW measurements to aggregate")
        return pd.DataFrame(
            columns=[
                "timestamp",
                "temperature_min",
                "temperature_max",
                "temperature_mean",
                "precipitation",
                "source",
            ]
        )

    # Extract values from measurements
    temperature = measurements.get("temperature")
    precipitation = measurements.get("precipitation", 0.0)
    observation_time = metadata.get("observation_time", "")

    # Parse date from observation time
    date_str = ""
    if observation_time:
        try:
            dt = datetime.fromisoformat(observation_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now().strftime("%Y-%m-%d")

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # IMGW provides single point measurement - use as daily value
    # MVP: Single observation per day
    if temperature is not None:
        try:
            temp_val = float(temperature)
        except (ValueError, TypeError):
            temp_val = float("nan")
    else:
        temp_val = float("nan")

    try:
        precip_val = float(precipitation) if precipitation is not None else 0.0
    except (ValueError, TypeError):
        precip_val = 0.0

    record = {
        "timestamp": date_str,
        "temperature_min": temp_val,  # Single obs = all same
        "temperature_max": temp_val,
        "temperature_mean": temp_val,
        "precipitation": precip_val,
        "source": "imgw",
    }

    return pd.DataFrame([record])


def merge_weather_sources(
    era5_df: pd.DataFrame,
    imgw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ERA5 and IMGW DataFrames into unified daily weather data.

    Combines data from both sources, preferring IMGW for overlapping dates
    when available (ground truth). Adds source column to identify origin.

    Args:
        era5_df: DataFrame from aggregate_era5_daily().
        imgw_df: DataFrame from aggregate_imgw_daily().

    Returns:
        Combined DataFrame sorted by timestamp with source column
        indicating data origin for each row.

    Example:
        >>> merged = merge_weather_sources(era5_df, imgw_df)
        >>> merged["source"].unique().tolist()
        ['era5', 'imgw']
    """
    # Handle empty cases
    if era5_df.empty and imgw_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "temperature_min",
                "temperature_max",
                "temperature_mean",
                "precipitation",
                "source",
            ]
        )

    if era5_df.empty:
        return imgw_df.copy()

    if imgw_df.empty:
        return era5_df.copy()

    # For overlapping dates, prefer IMGW (ground truth)
    imgw_dates = set(imgw_df["timestamp"].tolist())
    era5_unique = era5_df[~era5_df["timestamp"].isin(imgw_dates)]

    # Combine and sort
    merged = pd.concat([era5_unique, imgw_df], ignore_index=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged


def _calculate_confidence(
    era5_days: int,
    imgw_days: int,
    requested_days: int,
) -> float:
    """Calculate confidence based on data completeness.

    ERA5 is the primary source (global coverage), IMGW provides
    supplementary ground truth for Polish locations.

    Args:
        era5_days: Number of ERA5 daily observations.
        imgw_days: Number of IMGW daily observations.
        requested_days: Number of days in requested time range.

    Returns:
        Confidence score 0.0-1.0.

    Example:
        >>> _calculate_confidence(30, 5, 31)
        0.806...
    """
    if requested_days == 0:
        return 0.0

    # ERA5 is primary (global coverage), IMGW is supplementary
    era5_coverage = min(era5_days / requested_days, 1.0)

    # Base confidence on ERA5 coverage, boost if IMGW confirms
    confidence = era5_coverage * 0.8
    if imgw_days > 0:
        imgw_coverage = min(imgw_days / requested_days, 1.0)
        confidence += 0.2 * imgw_coverage

    return min(confidence, 1.0)


def build_weather_result(
    era5_data: RawData | None,
    imgw_data: RawData | None,
    location: Location,
    time_range: tuple[str, str],
) -> WeatherResult:
    """Build a WeatherResult from ERA5 and/or IMGW raw data.

    Aggregates raw provider data into daily summaries, calculates
    statistics, and constructs a WeatherResult with confidence scoring.

    Args:
        era5_data: Optional RawData from CDSProvider.download().
        imgw_data: Optional RawData from IMGWProvider.download().
        location: Location object for the query area.
        time_range: Tuple of (start_date, end_date) ISO strings.

    Returns:
        WeatherResult with aggregated weather data and statistics.

    Example:
        >>> result = build_weather_result(
        ...     era5_data=era5_raw,
        ...     imgw_data=None,
        ...     location=loc,
        ...     time_range=("2024-01-01", "2024-01-31"),
        ... )
        >>> result.data_source
        'era5'
    """
    # Aggregate each source
    era5_df = pd.DataFrame()
    imgw_df = pd.DataFrame()

    if era5_data is not None:
        era5_df = aggregate_era5_daily(era5_data)

    if imgw_data is not None:
        imgw_df = aggregate_imgw_daily(imgw_data)

    # Merge sources
    merged_df = merge_weather_sources(era5_df, imgw_df)

    # Calculate requested days
    try:
        start_dt = datetime.fromisoformat(time_range[0])
        end_dt = datetime.fromisoformat(time_range[1])
        requested_days = (end_dt - start_dt).days + 1
    except (ValueError, IndexError):
        requested_days = 0

    # Determine data source string
    has_era5 = not era5_df.empty
    has_imgw = not imgw_df.empty

    if has_era5 and has_imgw:
        data_source = "era5+imgw"
    elif has_era5:
        data_source = "era5"
    elif has_imgw:
        data_source = "imgw"
    else:
        data_source = ""

    # Calculate statistics from merged data
    if merged_df.empty:
        mean_temp = float("nan")
        total_precip = float("nan")
        temp_min = float("nan")
        temp_max = float("nan")
        observation_count = 0
    else:
        mean_temp = float(merged_df["temperature_mean"].mean())
        total_precip = float(merged_df["precipitation"].sum())
        temp_min = float(merged_df["temperature_min"].min())
        temp_max = float(merged_df["temperature_max"].max())
        observation_count = len(merged_df)

    # Calculate confidence
    era5_days = len(era5_df)
    imgw_days = len(imgw_df)
    confidence = _calculate_confidence(era5_days, imgw_days, requested_days)

    # Build metadata
    timestamps: list[str] = []
    if not merged_df.empty:
        timestamps = merged_df["timestamp"].tolist()

    # Get bounds from location
    bounds: dict[str, float] = {}
    if hasattr(location, "bounds") and location.bounds:
        bounds = {
            "minx": location.bounds[0],
            "miny": location.bounds[1],
            "maxx": location.bounds[2],
            "maxy": location.bounds[3],
        }

    metadata = ResultMetadata(
        source=data_source,
        timestamps=timestamps,
        observation_count=observation_count,
        bounds=bounds,
    )

    # Build warnings
    warnings: list[str] = []
    if confidence < 0.5:
        warnings.append("Low confidence due to limited data coverage")
    if not has_era5 and not has_imgw:
        warnings.append("No weather data available for this location/period")
    if has_imgw and not has_era5:
        warnings.append("IMGW data only (current observation, no historical)")

    # Use empty array for type compatibility (actual data is DataFrame)
    data_array = np.array([], dtype=np.float32)

    return WeatherResult(
        data=data_array,
        confidence=confidence,
        metadata=metadata,
        warnings=warnings,
        mean_temperature=mean_temp,
        total_precipitation=total_precip,
        data_source=data_source,
        temperature_min=temp_min,
        temperature_max=temp_max,
        observation_count=observation_count,
    )
