"""Result object model for analysis outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd

# ── NDVI interpretation thresholds ────────────────────────────────
_NDVI_HEALTHY_THRESHOLD: float = 0.6
_NDVI_MODERATE_THRESHOLD: float = 0.3
_NDVI_SPARSE_THRESHOLD: float = 0.1

# ── Change detection thresholds ────────────────────────────────────
_STABLE_THRESHOLD: float = 0.02  # Delta below this is "stable"
_SIGNIFICANT_CHANGE_THRESHOLD: float = 0.15  # Delta above this is "significant"

# ── Temperature interpretation thresholds (°C) ─────────────────────
_TEMP_COLD_THRESHOLD: float = 5.0  # Below: "cold"
_TEMP_MILD_THRESHOLD: float = 15.0  # 5-15: "mild"
_TEMP_WARM_THRESHOLD: float = 25.0  # 15-25: "warm", above: "hot"

# ── Precipitation interpretation thresholds (mm total) ─────────────
_PRECIP_DRY_THRESHOLD: float = 1.0  # Below: "dry"
_PRECIP_LIGHT_THRESHOLD: float = 10.0  # 1-10: "light rainfall"
_PRECIP_MODERATE_THRESHOLD: float = 30.0  # 10-30: "moderate", above: "heavy"

# ── Land Surface Temperature interpretation thresholds (°C) ─────────
# LST differs from air temperature: surfaces can be much hotter
_LST_COLD_THRESHOLD: float = 10.0  # Below: "cold surface"
_LST_COOL_THRESHOLD: float = 20.0  # 10-20: "cool surface"
_LST_WARM_THRESHOLD: float = 35.0  # 20-35: "warm surface"
_LST_HOT_THRESHOLD: float = 45.0  # 35-45: "hot surface", above: "extreme"


def _interpret_ndvi(value: float) -> str:
    """Return plain-language interpretation of an NDVI value.

    Args:
        value: NDVI value (typically in [-1, 1]).

    Returns:
        Human-readable interpretation string.
    """
    if math.isnan(value):
        return "no data"
    if value >= _NDVI_HEALTHY_THRESHOLD:
        return "healthy vegetation"
    if value >= _NDVI_MODERATE_THRESHOLD:
        return "moderate vegetation"
    if value >= _NDVI_SPARSE_THRESHOLD:
        return "sparse/stressed vegetation"
    return "bare soil/water"


def _interpret_temperature(mean_temp: float) -> str:
    """Return plain-language interpretation of a temperature value.

    Args:
        mean_temp: Mean temperature in degrees Celsius.

    Returns:
        Human-readable interpretation string.

    Example:
        >>> _interpret_temperature(12.5)
        'mild'
        >>> _interpret_temperature(28.0)
        'hot'
    """
    if math.isnan(mean_temp):
        return "no data"
    if mean_temp < _TEMP_COLD_THRESHOLD:
        return "cold"
    if mean_temp < _TEMP_MILD_THRESHOLD:
        return "mild"
    if mean_temp < _TEMP_WARM_THRESHOLD:
        return "warm"
    return "hot"


def _interpret_precipitation(total_precip: float) -> str:
    """Return plain-language interpretation of precipitation totals.

    Args:
        total_precip: Total precipitation in millimeters.

    Returns:
        Human-readable interpretation string.

    Example:
        >>> _interpret_precipitation(0.5)
        'dry conditions'
        >>> _interpret_precipitation(25.0)
        'moderate rainfall'
    """
    if math.isnan(total_precip):
        return "no data"
    if total_precip < _PRECIP_DRY_THRESHOLD:
        return "dry conditions"
    if total_precip < _PRECIP_LIGHT_THRESHOLD:
        return "light rainfall"
    if total_precip < _PRECIP_MODERATE_THRESHOLD:
        return "moderate rainfall"
    return "heavy rainfall"


def _interpret_surface_temperature(mean_temp: float) -> str:
    """Return plain-language interpretation of land surface temperature.

    Land surface temperature (LST) differs from air temperature - surfaces
    like asphalt, bare soil, or rooftops can be significantly hotter than
    the surrounding air, especially in summer.

    Args:
        mean_temp: Mean land surface temperature in degrees Celsius.

    Returns:
        Human-readable interpretation string.

    Example:
        >>> _interpret_surface_temperature(25.5)
        'warm surface'
        >>> _interpret_surface_temperature(48.0)
        'extreme heat'
    """
    if math.isnan(mean_temp):
        return "no data"
    if mean_temp < _LST_COLD_THRESHOLD:
        return "cold surface"
    if mean_temp < _LST_COOL_THRESHOLD:
        return "cool surface"
    if mean_temp < _LST_WARM_THRESHOLD:
        return "warm surface"
    if mean_temp < _LST_HOT_THRESHOLD:
        return "hot surface"
    return "extreme heat"


def _interpret_change(delta: float) -> str:
    """Return plain-language interpretation of vegetation change.

    Args:
        delta: NDVI change (period_2 - period_1). Positive = improvement.

    Returns:
        Human-readable interpretation of the vegetation change.

    Example:
        >>> _interpret_change(-0.20)
        'significant vegetation decline detected'
        >>> _interpret_change(0.01)
        'vegetation stable (no significant change)'
    """
    if math.isnan(delta):
        return "change could not be determined"
    if abs(delta) < _STABLE_THRESHOLD:
        return "vegetation stable (no significant change)"
    if delta > _SIGNIFICANT_CHANGE_THRESHOLD:
        return "significant vegetation improvement detected"
    if delta > 0:
        return "vegetation improvement detected"
    if delta < -_SIGNIFICANT_CHANGE_THRESHOLD:
        return "significant vegetation decline detected"
    return "vegetation decline detected"


class ResultMetadata(BaseModel):
    """Metadata for analysis results.

    Uses Pydantic (not dataclass) for JSON serialization in export methods
    and API responses. See AD-5: "Pydantic for config/serialization boundaries."

    Attributes:
        source: Data provider name (e.g., ``"cdse"``).
        timestamps: ISO-8601 acquisition timestamps of observations used.
        observation_count: Number of satellite passes or data points.
        cloud_cover_pct: Average cloud cover percentage across observations.
        crs: Coordinate reference system (e.g., ``"EPSG:32634"``).
        bounds: Spatial bounding box ``{"minx", "miny", "maxx", "maxy"}``.
        resolution_m: Spatial resolution in metres, if applicable.
        bands: Band identifiers present in the result data.

    Example:
        >>> meta = ResultMetadata(source="cdse", observation_count=3)
        >>> meta.source
        'cdse'
    """

    source: str = ""
    timestamps: list[str] = Field(default_factory=list)
    observation_count: int = 0
    cloud_cover_pct: float = 0.0
    crs: str = ""
    bounds: dict[str, float] = Field(default_factory=dict)
    resolution_m: float | None = None
    bands: list[str] = Field(default_factory=list)


@dataclass
class BaseResult:
    """Base result for all analysis outputs.

    Dataclass (not Pydantic) because numpy arrays are the primary payload
    and Pydantic fights arbitrary types. See AD-5.

    Attributes:
        data: Raster array with shape ``(bands, height, width)`` or empty.
        confidence: Overall confidence score (0.0--1.0).
        metadata: Pydantic model with source, timestamps, and quality fields.
        warnings: Human-readable quality or availability warnings.

    Example:
        >>> result = BaseResult(data=np.array([]))
        >>> result.confidence
        0.0
        >>> len(result.warnings)
        0
    """

    data: npt.NDArray[np.floating[Any]]
    confidence: float = 0.0
    metadata: ResultMetadata = field(default_factory=ResultMetadata)
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return summary representation following progressive disclosure.

        Shows type name, confidence, observation count, and warning count.
        Does NOT show raw arrays or full metadata dictionaries.
        """
        cls_name = type(self).__name__
        obs = self.metadata.observation_count
        warn_count = len(self.warnings)
        parts = [
            f"confidence={self.confidence:.2f}",
            f"observations={obs}",
        ]
        if warn_count:
            parts.append(f"warnings={warn_count}")
        return f"{cls_name}({', '.join(parts)})"

    def narrative(self) -> str:
        """Return a human-readable narrative summary of the analysis result.

        Provides a plain-language description of the result suitable for
        display to end users. Subclasses with domain-specific fields
        (VegetationResult, WeatherResult, etc.) return more detailed
        narratives including interpretation of the data.

        Returns:
            Multi-line string with the result summary.

        Example:
            >>> result = VegetationResult(data=np.array([]), mean_ndvi=0.42)
            >>> print(result.narrative())  # doctest: +SKIP
            VegetationResult(
              location: 51.50°N, 22.50°E
              mean_ndvi: 0.42 ± 0.08 — moderate vegetation
              ...
            )
        """
        return repr(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Export result data to a pandas DataFrame.

        Returns a DataFrame with descriptive column names containing the result
        data, location, timestamps, and key metrics.

        Returns:
            pandas DataFrame with the result data.

        Example:
            >>> result = BaseResult(data=np.array([[1, 2], [3, 4]]))
            >>> df = result.to_dataframe()
            >>> len(df) > 0
            True
        """
        import pandas as pd

        # Base implementation returns metadata and data summary
        rows: list[dict[str, Any]] = []
        bounds = self.metadata.bounds
        lat = lon = None
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2

        row: dict[str, Any] = {
            "confidence": self.confidence,
            "source": self.metadata.source,
            "observation_count": self.metadata.observation_count,
            "crs": self.metadata.crs,
            "latitude": lat,
            "longitude": lon,
        }

        # Add timestamps if available
        if self.metadata.timestamps:
            row["period_start"] = self.metadata.timestamps[0]
            row["period_end"] = self.metadata.timestamps[-1]

        # Add data summary if array has content
        if self.data.size > 0:
            row["data_min"] = float(np.nanmin(self.data))
            row["data_max"] = float(np.nanmax(self.data))
            row["data_mean"] = float(np.nanmean(self.data))

        rows.append(row)
        return pd.DataFrame(rows)

    def to_geotiff(self, path: str | Path) -> Path:
        """Export raster data to a GeoTIFF file.

        Writes the result data array as a GeoTIFF with correct CRS metadata.

        Args:
            path: Output file path (will be created/overwritten).

        Returns:
            Path object pointing to the written file.

        Raises:
            ValueError: If result has no raster data or is not raster-based.

        Example:
            >>> result = VegetationResult(data=np.random.rand(1, 100, 100))
            >>> output_path = result.to_geotiff("output.tif")
        """
        import rasterio
        from rasterio.transform import from_bounds

        path = Path(path)

        # Check for valid raster data
        if self.data.size == 0:
            msg = "Cannot export empty result to GeoTIFF"
            raise ValueError(msg)

        if self.data.ndim < 2:
            msg = "Data must be at least 2D for GeoTIFF export"
            raise ValueError(msg)

        # Ensure 3D array (bands, height, width)
        data = self.data
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        bands, height, width = data.shape

        # Get CRS and bounds from metadata
        crs = self.metadata.crs or "EPSG:4326"
        bounds = self.metadata.bounds

        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            transform = from_bounds(
                bounds["minx"],
                bounds["miny"],
                bounds["maxx"],
                bounds["maxy"],
                width,
                height,
            )
        else:
            # Default transform if no bounds available
            transform = from_bounds(0, 0, width, height, width, height)

        # Write GeoTIFF
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(bands):
                dst.write(data[i], i + 1)

        return path

    def to_png(self, path: str | Path) -> Path:
        """Export visualization to a PNG image.

        Writes a PNG image with the primary data visualization including
        a title with location, period, and key metric.

        Args:
            path: Output file path (will be created/overwritten).

        Returns:
            Path object pointing to the written file.

        Example:
            >>> result = BaseResult(data=np.random.rand(100, 100))
            >>> output_path = result.to_png("output.png")
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for file output
        import matplotlib.pyplot as plt

        path = Path(path)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Build title
        title_parts = [type(self).__name__]
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            title_parts.append(f"({abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew})")

        ts = self.metadata.timestamps
        if ts:
            title_parts.append(f"\n{ts[0][:10]} \u2192 {ts[-1][:10]}")

        title_parts.append(f"\nConfidence: {self.confidence:.2f}")
        ax.set_title(" ".join(title_parts[:2]) + "".join(title_parts[2:]))

        # Plot data
        if self.data.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        elif self.data.ndim >= 2:
            # For 3D data, show first band or mean
            plot_data = self.data
            if plot_data.ndim == 3:
                plot_data = plot_data[0]  # First band
            im = ax.imshow(plot_data, cmap="viridis")
            plt.colorbar(im, ax=ax, label="Value")
        else:
            # 1D data as line plot
            ax.plot(self.data)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def to_html(self, path: str | Path | None = None) -> str | Path:
        """Export result as a standalone HTML report.

        Generates a self-contained HTML document with embedded visualization
        (as base64 PNG), key metrics, and metadata. Can either write to a file
        or return the HTML string.

        Args:
            path: Output file path. If None, returns HTML as string.
                If provided, writes to file and returns the Path.

        Returns:
            HTML string if path is None, otherwise Path to written file.

        Example:
            >>> result = BaseResult(data=np.random.rand(100, 100))
            >>> html = result.to_html()  # Returns HTML string
            >>> result.to_html("report.html")  # Writes to file
        """
        import base64
        import tempfile
        from datetime import datetime

        # Generate PNG to temp file and encode as base64
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        self.to_png(tmp_path)
        with open(tmp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        tmp_path.unlink()

        # Build location string
        loc_str = ""
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            loc_str = f"{abs(lat):.4f}°{ns}, {abs(lon):.4f}°{ew}"

        # Build period string
        ts = self.metadata.timestamps
        period_str = ""
        if ts:
            period_str = f"{ts[0][:10]} to {ts[-1][:10]}"

        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cls_name = type(self).__name__

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{cls_name} Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6; color: #333; max-width: 900px;
            margin: 0 auto; padding: 20px; background: #f5f5f5;
        }}
        .report {{
            background: white; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c5530, #4a7c59);
            color: white; padding: 25px; text-align: center;
        }}
        .header h1 {{ margin: 0 0 5px; font-size: 24px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 14px; }}
        .section {{ padding: 20px 25px; border-bottom: 1px solid #eee; }}
        .metrics {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin: 15px 0;
        }}
        .metric {{
            background: #f8f9fa; padding: 12px; border-radius: 6px; text-align: center;
        }}
        .metric .value {{ font-size: 22px; font-weight: bold; color: #2c5530; }}
        .metric .label {{ font-size: 11px; color: #666; text-transform: uppercase; }}
        .chart {{ width: 100%; max-width: 750px; margin: 15px auto; display: block; }}
        .warning {{ background: #fff3cd; padding: 10px 15px; margin: 10px 0;
            border-left: 4px solid #ffc107; border-radius: 0 4px 4px 0; }}
        .footer {{ text-align: center; padding: 15px; background: #f8f9fa;
            color: #666; font-size: 11px; }}
    </style>
</head>
<body>
<div class="report">
    <div class="header">
        <h1>{cls_name}</h1>
        <div class="subtitle">{loc_str or 'Location not specified'}</div>
    </div>
    <div class="section">
        <div class="metrics">
            <div class="metric">
                <div class="value">{self.confidence:.0%}</div>
                <div class="label">Confidence</div>
            </div>
            <div class="metric">
                <div class="value">{self.metadata.observation_count}</div>
                <div class="label">Observations</div>
            </div>
            <div class="metric">
                <div class="value">{self.metadata.source or 'N/A'}</div>
                <div class="label">Source</div>
            </div>
            <div class="metric">
                <div class="value">{period_str or 'N/A'}</div>
                <div class="label">Period</div>
            </div>
        </div>
        <img src="data:image/png;base64,{img_b64}" alt="Chart" class="chart">
    </div>
"""

        # Add warnings if present
        if self.warnings:
            html += '    <div class="section">\n'
            for w in self.warnings:
                html += f'        <div class="warning">{w}</div>\n'
            html += "    </div>\n"

        html += f"""    <div class="footer">
        Generated: {report_time}
    </div>
</div>
</body>
</html>
"""

        if path is None:
            return html

        path = Path(path)
        path.write_text(html, encoding="utf-8")
        return path


@dataclass
class VegetationResult(BaseResult):
    """Vegetation health analysis result with NDVI statistics.

    Extends ``BaseResult`` with vegetation-specific fields. The ``__repr__``
    renders a narrative summary following the progressive disclosure principle
    (AD-5): shows location, period, observation count, confidence, mean NDVI
    with interpretation, trend, and warnings — but NOT raw arrays.

    Attributes:
        mean_ndvi: Spatial mean NDVI across cloud-free pixels (NaN if no data).
        ndvi_std: Standard deviation of NDVI values (NaN if no data).
        trend: Monthly NDVI trend (None for single-period analysis).
        observation_count: Total satellite passes in the query period.
        cloud_free_count: Passes with usable cloud-free data.

    Example:
        >>> import numpy as np
        >>> result = VegetationResult(
        ...     data=np.array([[0.42, 0.38]], dtype=np.float32),
        ...     confidence=0.78,
        ...     mean_ndvi=0.42,
        ...     ndvi_std=0.08,
        ...     observation_count=6,
        ...     cloud_free_count=4,
        ... )
        >>> result.mean_ndvi
        0.42
    """

    mean_ndvi: float = float("nan")
    ndvi_std: float = float("nan")
    trend: float | None = None
    observation_count: int = 0
    cloud_free_count: int = 0

    def __repr__(self) -> str:
        """Return narrative summary for interactive display.

        Shows location, period, observations, confidence, mean NDVI
        with interpretation, trend, and warnings. Does NOT show raw
        arrays or full metadata dictionaries. Works in Jupyter, REPL,
        debuggers, and any context that calls repr().
        """
        lines: list[str] = [f"{type(self).__name__}("]

        # Location from bounds (requires all 4 keys to be valid)
        bounds = self.metadata.bounds
        required_keys = {"minx", "miny", "maxx", "maxy"}
        if bounds and required_keys.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            lines.append(
                f"  location: {abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew}"
            )

        # Period from timestamps
        ts = self.metadata.timestamps
        if ts:
            lines.append(f"  period: {ts[0][:10]} \u2192 {ts[-1][:10]}")

        # Observations
        lines.append(
            f"  observations: {self.cloud_free_count} of "
            f"{self.observation_count} passes (cloud-free)"
        )

        # Confidence
        lines.append(f"  confidence: {self.confidence:.2f}")

        # Mean NDVI with interpretation
        if math.isnan(self.mean_ndvi):
            lines.append("  mean_ndvi: N/A (no valid data)")
        else:
            interp = _interpret_ndvi(self.mean_ndvi)
            lines.append(
                f"  mean_ndvi: {self.mean_ndvi:.2f} \u00b1 {self.ndvi_std:.2f}"
                f" \u2014 {interp}"
            )

        # Trend
        if self.trend is not None:
            if self.trend > 0:
                direction = "improving"
            elif self.trend < 0:
                direction = "declining"
            else:
                direction = "stable"
            lines.append(f"  trend: {direction} ({self.trend:+.2f}/month)")
        else:
            lines.append("  trend: N/A (single period)")

        # Warnings
        for w in self.warnings:
            lines.append(f"  \u26a0 {w}")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export vegetation result to pandas DataFrame.

        Returns a DataFrame with vegetation-specific columns including NDVI
        statistics, observation counts, and metadata.

        Returns:
            pandas DataFrame with vegetation analysis data.
        """
        import pandas as pd

        bounds = self.metadata.bounds
        lat = lon = None
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2

        row: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "mean_ndvi": self.mean_ndvi,
            "ndvi_std": self.ndvi_std,
            "ndvi_interpretation": _interpret_ndvi(self.mean_ndvi),
            "trend": self.trend,
            "observation_count": self.observation_count,
            "cloud_free_count": self.cloud_free_count,
            "confidence": self.confidence,
            "source": self.metadata.source,
            "crs": self.metadata.crs,
        }

        if self.metadata.timestamps:
            row["period_start"] = self.metadata.timestamps[0]
            row["period_end"] = self.metadata.timestamps[-1]

        return pd.DataFrame([row])

    def to_png(self, path: str | Path) -> Path:
        """Export NDVI visualization to PNG image.

        Creates a visualization showing the NDVI raster with appropriate
        colormap and annotations for vegetation health.

        Args:
            path: Output file path.

        Returns:
            Path object pointing to the written file.
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for file output
        import matplotlib.pyplot as plt

        path = Path(path)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Build title
        title_parts = ["Vegetation Health (NDVI)"]
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            title_parts.append(f"({abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew})")

        ts = self.metadata.timestamps
        if ts:
            title_parts.append(f"\n{ts[0][:10]} \u2192 {ts[-1][:10]}")

        # Add key metric
        if not math.isnan(self.mean_ndvi):
            interp = _interpret_ndvi(self.mean_ndvi)
            title_parts.append(f"\nMean NDVI: {self.mean_ndvi:.2f} ({interp})")
        else:
            title_parts.append("\nMean NDVI: N/A")

        ax.set_title(" ".join(title_parts[:2]) + "".join(title_parts[2:]))

        # Plot NDVI data
        if self.data.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        elif self.data.ndim >= 2:
            plot_data = self.data
            if plot_data.ndim == 3:
                plot_data = plot_data[0]
            # NDVI typically ranges -1 to 1, use RdYlGn colormap
            im = ax.imshow(plot_data, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
            plt.colorbar(im, ax=ax, label="NDVI")
        else:
            ax.plot(self.data)
            ax.set_ylabel("NDVI")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def to_html(self, path: str | Path | None = None) -> str | Path:
        """Export vegetation result as a standalone HTML report.

        Generates a self-contained HTML document with NDVI visualization,
        vegetation health metrics, and interpretation.

        Args:
            path: Output file path. If None, returns HTML as string.

        Returns:
            HTML string if path is None, otherwise Path to written file.
        """
        import base64
        import tempfile
        from datetime import datetime

        # Generate PNG and encode
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        self.to_png(tmp_path)
        with open(tmp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        tmp_path.unlink()

        # Location and period
        loc_str = ""
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            loc_str = f"{abs(lat):.4f}°{ns}, {abs(lon):.4f}°{ew}"

        ts = self.metadata.timestamps
        period_str = f"{ts[0][:10]} to {ts[-1][:10]}" if ts else "N/A"

        # NDVI interpretation
        interp = _interpret_ndvi(self.mean_ndvi)
        ndvi_val = f"{self.mean_ndvi:.2f}" if not math.isnan(self.mean_ndvi) else "N/A"
        std_val = f"±{self.ndvi_std:.2f}" if not math.isnan(self.ndvi_std) else "N/A"
        passes_str = f"{self.cloud_free_count}/{self.observation_count}"

        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vegetation Health Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6; color: #333; max-width: 900px;
            margin: 0 auto; padding: 20px; background: #f5f5f5;
        }}
        .report {{
            background: white; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c5530, #4a7c59);
            color: white; padding: 25px; text-align: center;
        }}
        .header h1 {{ margin: 0 0 5px; font-size: 24px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 14px; }}
        .section {{ padding: 20px 25px; border-bottom: 1px solid #eee; }}
        .metrics {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin: 15px 0;
        }}
        .metric {{
            background: #f8f9fa; padding: 12px; border-radius: 6px; text-align: center;
        }}
        .metric .value {{ font-size: 22px; font-weight: bold; color: #2c5530; }}
        .metric .label {{ font-size: 11px; color: #666; text-transform: uppercase; }}
        .status {{
            display: inline-block; padding: 8px 16px; border-radius: 20px;
            font-weight: 500; background: #d4edda; color: #155724; margin: 10px 0;
        }}
        .chart {{ width: 100%; max-width: 750px; margin: 15px auto; display: block; }}
        .warning {{ background: #fff3cd; padding: 10px 15px; margin: 10px 0;
            border-left: 4px solid #ffc107; border-radius: 0 4px 4px 0; }}
        .footer {{ text-align: center; padding: 15px; background: #f8f9fa;
            color: #666; font-size: 11px; }}
    </style>
</head>
<body>
<div class="report">
    <div class="header">
        <h1>Vegetation Health Analysis</h1>
        <div class="subtitle">{loc_str or 'Location not specified'} | {period_str}</div>
    </div>
    <div class="section">
        <div class="status">{interp.title()}</div>
        <div class="metrics">
            <div class="metric">
                <div class="value">{ndvi_val}</div>
                <div class="label">Mean NDVI</div>
            </div>
            <div class="metric">
                <div class="value">{std_val}</div>
                <div class="label">Std Deviation</div>
            </div>
            <div class="metric">
                <div class="value">{self.confidence:.0%}</div>
                <div class="label">Confidence</div>
            </div>
            <div class="metric">
                <div class="value">{passes_str}</div>
                <div class="label">Cloud-Free</div>
            </div>
        </div>
        <img src="data:image/png;base64,{img_b64}" alt="NDVI Chart" class="chart">
    </div>
"""
        if self.warnings:
            html += '    <div class="section">\n'
            for w in self.warnings:
                html += f'        <div class="warning">{w}</div>\n'
            html += "    </div>\n"

        html += f"""    <div class="footer">
        Generated: {report_time}
    </div>
</div>
</body>
</html>
"""

        if path is None:
            return html
        path = Path(path)
        path.write_text(html, encoding="utf-8")
        return path


@dataclass
class ChangeResult(BaseResult):
    """Vegetation change detection result comparing two time periods.

    Extends ``BaseResult`` with change-specific fields for comparing NDVI
    between a baseline period and a comparison period. The ``__repr__``
    renders a narrative summary showing both periods, the delta, and
    plain-language interpretation.

    Attributes:
        period_1_ndvi: Mean NDVI for the first (baseline) period (NaN if no data).
        period_2_ndvi: Mean NDVI for the second (comparison) period (NaN if no data).
        delta: Change in NDVI (period_2 - period_1). Positive = improvement.
        direction: Plain-language direction ("improving", "declining", "stable",
            or "unknown" if either period lacks data).
        period_1_confidence: Confidence score for period 1 data (0.0--1.0).
        period_2_confidence: Confidence score for period 2 data (0.0--1.0).
        period_1_range: Time range tuple (start, end) ISO strings for period 1.
        period_2_range: Time range tuple (start, end) ISO strings for period 2.

    Example:
        >>> import numpy as np
        >>> result = ChangeResult(
        ...     data=np.array([], dtype=np.float32),
        ...     confidence=0.65,
        ...     period_1_ndvi=0.42,
        ...     period_2_ndvi=0.27,
        ...     delta=-0.15,
        ...     direction="declining",
        ...     period_1_confidence=0.78,
        ...     period_2_confidence=0.65,
        ...     period_1_range=("2025-01-01", "2025-01-31"),
        ...     period_2_range=("2026-01-01", "2026-01-31"),
        ... )
        >>> result.delta
        -0.15
        >>> result.direction
        'declining'
    """

    period_1_ndvi: float = float("nan")
    period_2_ndvi: float = float("nan")
    delta: float = float("nan")
    direction: str = "unknown"
    period_1_confidence: float = 0.0
    period_2_confidence: float = 0.0
    period_1_range: tuple[str, str] = field(default_factory=lambda: ("", ""))
    period_2_range: tuple[str, str] = field(default_factory=lambda: ("", ""))

    def __repr__(self) -> str:
        """Return narrative summary for interactive display.

        Shows both periods with NDVI and confidence, the change delta
        with interpretation, combined confidence, and warnings. Does NOT
        show raw arrays or full metadata dictionaries.
        """
        lines: list[str] = [f"{type(self).__name__}("]

        # Period 1
        p1_start, p1_end = self.period_1_range
        if p1_start and p1_end:
            if math.isnan(self.period_1_ndvi):
                ndvi_str = "N/A"
            else:
                ndvi_str = f"{self.period_1_ndvi:.2f}"
            lines.append(
                f"  period_1: {p1_start} \u2192 {p1_end} "
                f"(NDVI: {ndvi_str}, confidence: {self.period_1_confidence:.2f})"
            )

        # Period 2
        p2_start, p2_end = self.period_2_range
        if p2_start and p2_end:
            if math.isnan(self.period_2_ndvi):
                ndvi_str = "N/A"
            else:
                ndvi_str = f"{self.period_2_ndvi:.2f}"
            lines.append(
                f"  period_2: {p2_start} \u2192 {p2_end} "
                f"(NDVI: {ndvi_str}, confidence: {self.period_2_confidence:.2f})"
            )

        # Change delta with interpretation
        if math.isnan(self.delta):
            lines.append("  change: N/A (insufficient data)")
        else:
            interp = _interpret_change(self.delta)
            lines.append(f"  change: {self.delta:+.2f} ({interp})")

        # Combined confidence
        lines.append(f"  combined_confidence: {self.confidence:.2f}")

        # Warnings
        for w in self.warnings:
            lines.append(f"  \u26a0 {w}")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export change result to pandas DataFrame.

        Returns a DataFrame with change-specific columns including both period
        NDVI values, delta, direction, and confidence scores.

        Returns:
            pandas DataFrame with change detection data.
        """
        import pandas as pd

        bounds = self.metadata.bounds
        lat = lon = None
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2

        p1_start, p1_end = self.period_1_range
        p2_start, p2_end = self.period_2_range

        row: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "period_1_start": p1_start,
            "period_1_end": p1_end,
            "period_1_ndvi": self.period_1_ndvi,
            "period_1_confidence": self.period_1_confidence,
            "period_2_start": p2_start,
            "period_2_end": p2_end,
            "period_2_ndvi": self.period_2_ndvi,
            "period_2_confidence": self.period_2_confidence,
            "delta": self.delta,
            "direction": self.direction,
            "change_interpretation": _interpret_change(self.delta),
            "combined_confidence": self.confidence,
            "source": self.metadata.source,
            "crs": self.metadata.crs,
        }

        return pd.DataFrame([row])

    def to_png(self, path: str | Path) -> Path:
        """Export change detection visualization to PNG image.

        Creates a visualization showing the change between periods with
        appropriate colormap and annotations.

        Args:
            path: Output file path.

        Returns:
            Path object pointing to the written file.
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for file output
        import matplotlib.pyplot as plt

        path = Path(path)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Build title
        title_parts = ["Vegetation Change Detection"]
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            title_parts.append(f"({abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew})")

        p1_start, p1_end = self.period_1_range
        p2_start, p2_end = self.period_2_range
        if p1_start and p2_end:
            title_parts.append(f"\n{p1_start} \u2192 {p2_end}")

        # Add key metric
        if not math.isnan(self.delta):
            interp = _interpret_change(self.delta)
            title_parts.append(f"\nChange: {self.delta:+.2f} ({interp})")
        else:
            title_parts.append("\nChange: N/A")

        ax.set_title(" ".join(title_parts[:2]) + "".join(title_parts[2:]))

        # Plot change data
        if self.data.size == 0:
            # No raster data - show summary bar chart
            values = [self.period_1_ndvi, self.period_2_ndvi]
            labels = [f"Period 1\n{p1_start[:10]}", f"Period 2\n{p2_start[:10]}"]
            colors = ["#3498db", "#e74c3c" if self.delta < 0 else "#2ecc71"]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Mean NDVI")
            ax.set_ylim(0, 1)

            # Add delta annotation
            if not math.isnan(self.delta):
                ax.annotate(
                    f"\u0394 = {self.delta:+.2f}",
                    xy=(0.5, 0.95),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
        elif self.data.ndim >= 2:
            plot_data = self.data
            if plot_data.ndim == 3:
                plot_data = plot_data[0]
            # Change map: red = decline, green = improvement
            im = ax.imshow(plot_data, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
            plt.colorbar(im, ax=ax, label="NDVI Change")
        else:
            ax.plot(self.data)
            ax.set_ylabel("NDVI Change")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path


@dataclass
class WeatherResult(BaseResult):
    """Weather analysis result with temperature and precipitation.

    Extends ``BaseResult`` with weather-specific fields. The ``__repr__``
    renders a narrative summary following the progressive disclosure principle
    (AD-5): shows location, period, observation count, confidence, mean
    temperature with interpretation, precipitation totals, and data source.

    Attributes:
        data: DataFrame with daily weather summary (columns: timestamp,
            temperature_min, temperature_max, temperature_mean, precipitation,
            source). For type checking purposes, declared as NDArray but
            actually holds a pandas DataFrame at runtime.
        mean_temperature: Average temperature over period in °C (NaN if no data).
        total_precipitation: Total precipitation over period in mm (NaN if no data).
        data_source: Provider(s) used ("era5", "imgw", or "era5+imgw").
        temperature_min: Minimum temperature observed in °C (NaN if no data).
        temperature_max: Maximum temperature observed in °C (NaN if no data).
        observation_count: Number of daily observations.

    Example:
        >>> import numpy as np
        >>> result = WeatherResult(
        ...     data=np.array([], dtype=np.float32),
        ...     confidence=0.92,
        ...     mean_temperature=4.2,
        ...     total_precipitation=45.2,
        ...     data_source="era5+imgw",
        ...     temperature_min=-2.5,
        ...     temperature_max=12.3,
        ...     observation_count=31,
        ... )
        >>> result.mean_temperature
        4.2
    """

    mean_temperature: float = float("nan")
    total_precipitation: float = float("nan")
    data_source: str = ""
    temperature_min: float = float("nan")
    temperature_max: float = float("nan")
    observation_count: int = 0

    def __repr__(self) -> str:
        """Return narrative summary for interactive display.

        Shows location, period, observations, confidence, mean temperature
        with interpretation, temperature range, precipitation totals with
        interpretation, and data source. Does NOT show raw arrays or full
        metadata dictionaries.
        """
        lines: list[str] = [f"{type(self).__name__}("]

        # Location from bounds (requires all 4 keys to be valid)
        bounds = self.metadata.bounds
        required_keys = {"minx", "miny", "maxx", "maxy"}
        if bounds and required_keys.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            lines.append(
                f"  location: {abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew}"
            )

        # Period from timestamps
        ts = self.metadata.timestamps
        if ts:
            lines.append(f"  period: {ts[0][:10]} \u2192 {ts[-1][:10]}")

        # Observations
        lines.append(f"  observations: {self.observation_count} days")

        # Confidence
        lines.append(f"  confidence: {self.confidence:.2f}")

        # Mean temperature with interpretation
        if math.isnan(self.mean_temperature):
            lines.append("  mean_temperature: N/A (no valid data)")
        else:
            interp = _interpret_temperature(self.mean_temperature)
            temp_str = f"{self.mean_temperature:.1f}\u00b0C ({interp})"
            lines.append(f"  mean_temperature: {temp_str}")

        # Temperature range
        has_min = not math.isnan(self.temperature_min)
        has_max = not math.isnan(self.temperature_max)
        if has_min and has_max:
            lines.append(
                f"  temperature_range: {self.temperature_min:.1f}\u00b0C to "
                f"{self.temperature_max:.1f}\u00b0C"
            )

        # Total precipitation with interpretation
        if math.isnan(self.total_precipitation):
            lines.append("  total_precipitation: N/A (no valid data)")
        else:
            interp = _interpret_precipitation(self.total_precipitation)
            precip_str = f"{self.total_precipitation:.1f}mm ({interp})"
            lines.append(f"  total_precipitation: {precip_str}")

        # Data source
        if self.data_source:
            lines.append(f"  data_source: {self.data_source}")

        # Warnings
        for w in self.warnings:
            lines.append(f"  \u26a0 {w}")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export weather result to pandas DataFrame.

        Returns the daily weather time series data. If the internal data
        is already a DataFrame, returns it directly. Otherwise constructs
        a summary DataFrame from the aggregated statistics.

        Returns:
            pandas DataFrame with weather time series or summary data.
        """
        import pandas as pd

        # WeatherResult.data may be a DataFrame (from build_weather_result)
        # Check if it has DataFrame-like attributes
        if hasattr(self.data, "columns") and hasattr(self.data, "iloc"):
            # It's a DataFrame, return a copy (cast to Any for type safety)
            df_data: Any = self.data
            return df_data.copy()

        # Otherwise build summary from aggregated stats
        bounds = self.metadata.bounds
        lat = lon = None
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2

        row: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "mean_temperature": self.mean_temperature,
            "temperature_interpretation": _interpret_temperature(self.mean_temperature),
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "total_precipitation": self.total_precipitation,
            "precipitation_interpretation": _interpret_precipitation(
                self.total_precipitation
            ),
            "observation_count": self.observation_count,
            "data_source": self.data_source,
            "confidence": self.confidence,
        }

        if self.metadata.timestamps:
            row["period_start"] = self.metadata.timestamps[0]
            row["period_end"] = self.metadata.timestamps[-1]

        return pd.DataFrame([row])

    def to_geotiff(self, path: str | Path) -> Path:
        """Export to GeoTIFF (not supported for weather time series).

        Weather data is time-series based and cannot be meaningfully
        exported as a raster GeoTIFF format.

        Args:
            path: Output file path (unused).

        Raises:
            ValueError: Always raised because weather data is not raster-based.
        """
        msg = (
            "WeatherResult contains time-series data which cannot be exported "
            "as GeoTIFF. Use to_dataframe() to export the daily weather data "
            "or to_png() for visualization."
        )
        raise ValueError(msg)

    def to_png(self, path: str | Path) -> Path:
        """Export weather visualization to PNG image.

        Creates a dual-axis chart showing temperature and precipitation
        over time with appropriate annotations.

        Args:
            path: Output file path.

        Returns:
            Path object pointing to the written file.
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for file output
        import matplotlib.pyplot as plt

        path = Path(path)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Build title
        title_parts = ["Weather Summary"]
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            title_parts.append(f"({abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew})")

        ts = self.metadata.timestamps
        if ts:
            title_parts.append(f"\n{ts[0][:10]} \u2192 {ts[-1][:10]}")

        # Add key metrics
        metrics = []
        if not math.isnan(self.mean_temperature):
            interp = _interpret_temperature(self.mean_temperature)
            metrics.append(f"Mean: {self.mean_temperature:.1f}\u00b0C ({interp})")
        if not math.isnan(self.total_precipitation):
            interp = _interpret_precipitation(self.total_precipitation)
            metrics.append(f"Precip: {self.total_precipitation:.1f}mm ({interp})")
        if metrics:
            title_parts.append(f"\n{', '.join(metrics)}")

        ax1.set_title(" ".join(title_parts[:2]) + "".join(title_parts[2:]))

        # Check if we have DataFrame data for time series
        # Note: WeatherResult.data may be a DataFrame at runtime despite the type hint
        has_df = hasattr(self.data, "columns") and hasattr(self.data, "iloc")

        if has_df:
            # Cast to Any for DataFrame operations (type is NDArray in signature)
            df: Any = self.data
            if len(df) > 0:
                # Temperature on primary axis
                if "temperature_mean" in df.columns:
                    ax1.plot(
                        df["temperature_mean"],
                        color="tab:red",
                        linewidth=2,
                        label="Mean Temperature",
                    )
                if "temperature_min" in df.columns:
                    ax1.fill_between(
                        range(len(df)),
                        df["temperature_min"],
                        df["temperature_max"],
                        alpha=0.3,
                        color="tab:red",
                        label="Temp Range",
                    )
                ax1.set_xlabel("Day")
                ax1.set_ylabel("Temperature (\u00b0C)", color="tab:red")
                ax1.tick_params(axis="y", labelcolor="tab:red")

                # Precipitation on secondary axis
                ax2 = ax1.twinx()
                if "precipitation" in df.columns:
                    ax2.bar(
                        range(len(df)),
                        df["precipitation"],
                        alpha=0.6,
                        color="tab:blue",
                        label="Precipitation",
                    )
                ax2.set_ylabel("Precipitation (mm)", color="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:blue")

                # Combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        if not has_df or (has_df and len(self.data) == 0):
            # No time series - show summary bar chart
            categories = [
                "Mean Temp\n(\u00b0C)",
                "Min Temp\n(\u00b0C)",
                "Max Temp\n(\u00b0C)",
                "Precip\n(mm)",
            ]
            values = [
                self.mean_temperature,
                self.temperature_min,
                self.temperature_max,
                self.total_precipitation,
            ]
            colors = ["#e74c3c", "#3498db", "#e67e22", "#2980b9"]

            # Filter out NaN values
            valid = [
                (c, v, col)
                for c, v, col in zip(categories, values, colors, strict=True)
                if not math.isnan(v)
            ]
            if valid:
                cats, vals, cols = zip(*valid, strict=True)
                ax1.bar(cats, vals, color=cols)
                ax1.set_ylabel("Value")
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No weather data available",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax1.transAxes,
                )

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path


@dataclass
class ThermalResult(BaseResult):
    """Land surface temperature analysis result from thermal satellite bands.

    Extends ``BaseResult`` with thermal-specific fields for Landsat thermal
    band analysis (B10, B11, ST_B10). The ``__repr__`` renders a narrative
    summary showing temperature statistics with land surface interpretation.

    Land surface temperature (LST) differs from air temperature: surfaces like
    asphalt, bare soil, or rooftops can be significantly hotter than the
    surrounding air, especially in summer.

    Attributes:
        mean_temperature: Spatial mean land surface temperature in °C.
        temperature_min: Minimum observed temperature in °C.
        temperature_max: Maximum observed temperature in °C.
        temperature_std: Standard deviation of temperature values in °C.
        thermal_unit: Temperature unit, always "celsius" after conversion.
        thermal_band: Band identifier used (e.g., "B10", "ST_B10").

    Example:
        >>> import numpy as np
        >>> result = ThermalResult(
        ...     data=np.array([[25.5, 26.0], [24.8, 25.2]], dtype=np.float32),
        ...     confidence=0.85,
        ...     mean_temperature=25.4,
        ...     temperature_min=24.8,
        ...     temperature_max=26.0,
        ...     temperature_std=0.4,
        ...     thermal_unit="celsius",
        ...     thermal_band="ST_B10",
        ... )
        >>> result.mean_temperature
        25.4
    """

    mean_temperature: float = float("nan")
    temperature_min: float = float("nan")
    temperature_max: float = float("nan")
    temperature_std: float = float("nan")
    thermal_unit: str = "celsius"
    thermal_band: str = ""

    def __repr__(self) -> str:
        """Return narrative summary for interactive display.

        Shows location, acquisition time, confidence, temperature statistics
        with land surface interpretation, and thermal band used. Does NOT
        show raw arrays or full metadata dictionaries.
        """
        lines: list[str] = [f"{type(self).__name__}("]

        # Location from bounds (requires all 4 keys to be valid)
        bounds = self.metadata.bounds
        required_keys = {"minx", "miny", "maxx", "maxy"}
        if bounds and required_keys.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            lines.append(
                f"  location: {abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew}"
            )

        # Acquisition time from timestamps
        ts = self.metadata.timestamps
        if ts:
            lines.append(f"  acquisition: {ts[0][:10]}")

        # Confidence
        lines.append(f"  confidence: {self.confidence:.2f}")

        # Mean temperature with interpretation
        if math.isnan(self.mean_temperature):
            lines.append("  mean_temperature: N/A (no valid data)")
        else:
            interp = _interpret_surface_temperature(self.mean_temperature)
            temp_str = f"{self.mean_temperature:.1f}\u00b0C ({interp})"
            lines.append(f"  mean_temperature: {temp_str}")

        # Temperature range
        has_min = not math.isnan(self.temperature_min)
        has_max = not math.isnan(self.temperature_max)
        if has_min and has_max:
            lines.append(
                f"  temperature_range: {self.temperature_min:.1f}\u00b0C to "
                f"{self.temperature_max:.1f}\u00b0C"
            )

        # Standard deviation
        if not math.isnan(self.temperature_std):
            lines.append(f"  temperature_std: \u00b1{self.temperature_std:.1f}\u00b0C")

        # Thermal band
        if self.thermal_band:
            lines.append(f"  thermal_band: {self.thermal_band}")

        # Warnings
        for w in self.warnings:
            lines.append(f"  \u26a0 {w}")

        lines.append(")")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export thermal result to pandas DataFrame.

        Returns a DataFrame with thermal-specific columns including temperature
        statistics, interpretation, and metadata.

        Returns:
            pandas DataFrame with thermal analysis data.
        """
        import pandas as pd

        bounds = self.metadata.bounds
        lat = lon = None
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2

        row: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "mean_temperature": self.mean_temperature,
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "temperature_std": self.temperature_std,
            "temperature_interpretation": _interpret_surface_temperature(
                self.mean_temperature
            ),
            "thermal_unit": self.thermal_unit,
            "thermal_band": self.thermal_band,
            "confidence": self.confidence,
            "source": self.metadata.source,
            "crs": self.metadata.crs,
        }

        if self.metadata.timestamps:
            row["acquisition_date"] = self.metadata.timestamps[0]

        return pd.DataFrame([row])

    def to_png(self, path: str | Path) -> Path:
        """Export thermal visualization to PNG image.

        Creates a visualization showing the land surface temperature raster
        with a heat-appropriate colormap and annotations.

        Args:
            path: Output file path.

        Returns:
            Path object pointing to the written file.
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for file output
        import matplotlib.pyplot as plt

        path = Path(path)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Build title
        title_parts = ["Land Surface Temperature"]
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            title_parts.append(f"({abs(lat):.2f}\u00b0{ns}, {abs(lon):.2f}\u00b0{ew})")

        ts = self.metadata.timestamps
        if ts:
            title_parts.append(f"\n{ts[0][:10]}")

        # Add key metric
        if not math.isnan(self.mean_temperature):
            interp = _interpret_surface_temperature(self.mean_temperature)
            title_parts.append(f"\nMean: {self.mean_temperature:.1f}\u00b0C ({interp})")
        else:
            title_parts.append("\nMean: N/A")

        ax.set_title(" ".join(title_parts[:2]) + "".join(title_parts[2:]))

        # Plot thermal data
        if self.data.size == 0:
            ax.text(
                0.5,
                0.5,
                "No thermal data available",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        elif self.data.ndim >= 2:
            plot_data = self.data
            if plot_data.ndim == 3:
                plot_data = plot_data[0]
            # Use 'hot' colormap for thermal data, auto-scale to data range
            has_min = not math.isnan(self.temperature_min)
            has_max = not math.isnan(self.temperature_max)
            vmin = self.temperature_min if has_min else None
            vmax = self.temperature_max if has_max else None
            im = ax.imshow(plot_data, cmap="hot", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label="Temperature (\u00b0C)")
        else:
            ax.plot(self.data)
            ax.set_ylabel("Temperature (\u00b0C)")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def to_html(self, path: str | Path | None = None) -> str | Path:
        """Export thermal result as a standalone HTML report.

        Generates a self-contained HTML document with thermal visualization,
        temperature statistics, and land surface interpretation.

        Args:
            path: Output file path. If None, returns HTML as string.

        Returns:
            HTML string if path is None, otherwise Path to written file.
        """
        import base64
        import tempfile
        from datetime import datetime

        # Generate PNG and encode
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        self.to_png(tmp_path)
        with open(tmp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        tmp_path.unlink()

        # Location and acquisition time
        loc_str = ""
        bounds = self.metadata.bounds
        if bounds and {"minx", "miny", "maxx", "maxy"}.issubset(bounds.keys()):
            lat = (bounds["miny"] + bounds["maxy"]) / 2
            lon = (bounds["minx"] + bounds["maxx"]) / 2
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            loc_str = f"{abs(lat):.4f}°{ns}, {abs(lon):.4f}°{ew}"

        ts = self.metadata.timestamps
        acq_str = ts[0][:10] if ts else "N/A"

        # Temperature values
        interp = _interpret_surface_temperature(self.mean_temperature)
        mean_val = f"{self.mean_temperature:.1f}°C" if not math.isnan(
            self.mean_temperature
        ) else "N/A"
        min_val = f"{self.temperature_min:.1f}°C" if not math.isnan(
            self.temperature_min
        ) else "N/A"
        max_val = f"{self.temperature_max:.1f}°C" if not math.isnan(
            self.temperature_max
        ) else "N/A"

        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Land Surface Temperature Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6; color: #333; max-width: 900px;
            margin: 0 auto; padding: 20px; background: #f5f5f5;
        }}
        .report {{
            background: white; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #8b4513, #cd853f);
            color: white; padding: 25px; text-align: center;
        }}
        .header h1 {{ margin: 0 0 5px; font-size: 24px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 14px; }}
        .section {{ padding: 20px 25px; border-bottom: 1px solid #eee; }}
        .metrics {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin: 15px 0;
        }}
        .metric {{
            background: #f8f9fa; padding: 12px; border-radius: 6px; text-align: center;
        }}
        .metric .value {{ font-size: 22px; font-weight: bold; color: #8b4513; }}
        .metric .label {{ font-size: 11px; color: #666; text-transform: uppercase; }}
        .status {{
            display: inline-block; padding: 8px 16px; border-radius: 20px;
            font-weight: 500; background: #ffe4c4; color: #8b4513; margin: 10px 0;
        }}
        .chart {{ width: 100%; max-width: 750px; margin: 15px auto; display: block; }}
        .warning {{ background: #fff3cd; padding: 10px 15px; margin: 10px 0;
            border-left: 4px solid #ffc107; border-radius: 0 4px 4px 0; }}
        .footer {{ text-align: center; padding: 15px; background: #f8f9fa;
            color: #666; font-size: 11px; }}
    </style>
</head>
<body>
<div class="report">
    <div class="header">
        <h1>Land Surface Temperature</h1>
        <div class="subtitle">{loc_str or 'Location not specified'} | {acq_str}</div>
    </div>
    <div class="section">
        <div class="status">{interp.title()}</div>
        <div class="metrics">
            <div class="metric">
                <div class="value">{mean_val}</div>
                <div class="label">Mean Temperature</div>
            </div>
            <div class="metric">
                <div class="value">{min_val}</div>
                <div class="label">Minimum</div>
            </div>
            <div class="metric">
                <div class="value">{max_val}</div>
                <div class="label">Maximum</div>
            </div>
            <div class="metric">
                <div class="value">{self.thermal_band or 'N/A'}</div>
                <div class="label">Thermal Band</div>
            </div>
        </div>
        <img src="data:image/png;base64,{img_b64}" alt="Thermal Chart" class="chart">
    </div>
"""
        if self.warnings:
            html += '    <div class="section">\n'
            for w in self.warnings:
                html += f'        <div class="warning">{w}</div>\n'
            html += "    </div>\n"

        html += f"""    <div class="footer">
        Generated: {report_time} | Confidence: {self.confidence:.0%}
    </div>
</div>
</body>
</html>
"""

        if path is None:
            return html
        path = Path(path)
        path.write_text(html, encoding="utf-8")
        return path
