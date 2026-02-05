#!/usr/bin/env python3
"""Generate HTML analysis report from satellite data.

This script produces a self-contained HTML report with vegetation health
analysis, change detection, and weather context for a given location.

Usage:
    python run_analysis.py --lat 51.41 --lon 21.97 --days 30 --output report.html

Example:
    python run_analysis.py --lat 52.23 --lon 21.01 --days 90 --output warsaw_report.html
"""

from __future__ import annotations

import argparse
import base64
import math
import sys
from datetime import datetime
from pathlib import Path

# Check imports before running
try:
    import satellitehub as sh
except ImportError:
    print("Error: satellitehub not installed. Run: pip install satellitehub")
    sys.exit(1)


def encode_image_base64(path: Path) -> str:
    """Read image file and return base64 encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html_report(
    lat: float,
    lon: float,
    days: int,
    output_path: Path,
    location_name: str | None = None,
) -> None:
    """Generate HTML analysis report for given location.

    Args:
        lat: Latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees.
        days: Number of days to analyze.
        output_path: Path for output HTML file.
        location_name: Optional human-readable location name.
    """
    print(f"Generating analysis report for ({lat}, {lon})...")

    # Create location
    location = sh.location(lat=lat, lon=lon)
    loc_name = location_name or f"{lat:.2f}°N, {lon:.2f}°E"

    print(f"  Location: {loc_name}")
    print(f"  UTM Zone: {location.utm_zone}")
    print(f"  Analysis period: {days} days")

    # Run analyses
    print("  Running vegetation health analysis...")
    veg_result = location.vegetation_health(last_days=days)

    # Skip change detection for quick reports (requires additional downloads)
    change_result = None

    print("  Fetching weather data...")
    weather_result = location.weather(last_days=days)

    # Generate chart images to temporary files
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())

    veg_png = temp_dir / "vegetation.png"
    veg_result.to_png(veg_png)
    veg_b64 = encode_image_base64(veg_png)

    change_b64 = ""
    if change_result is not None:
        change_png = temp_dir / "change.png"
        change_result.to_png(change_png)
        change_b64 = encode_image_base64(change_png)

    weather_b64 = ""
    if weather_result.confidence > 0:
        weather_png = temp_dir / "weather.png"
        weather_result.to_png(weather_png)
        weather_b64 = encode_image_base64(weather_png)

    # Generate timestamp
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Interpret NDVI
    if math.isnan(veg_result.mean_ndvi):
        ndvi_status = "No data available"
        ndvi_class = "warning"
    elif veg_result.mean_ndvi >= 0.6:
        ndvi_status = "Healthy - Dense, vigorous vegetation"
        ndvi_class = "success"
    elif veg_result.mean_ndvi >= 0.3:
        ndvi_status = "Moderate - Normal vegetation cover"
        ndvi_class = "info"
    elif veg_result.mean_ndvi >= 0.1:
        ndvi_status = "Sparse - Stressed or sparse vegetation"
        ndvi_class = "warning"
    else:
        ndvi_status = "Minimal - Bare soil or water"
        ndvi_class = "danger"

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Satellite Analysis Report - {loc_name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c5530 0%, #4a7c59 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 16px; }}
        .section {{
            padding: 25px 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{
            color: #2c5530;
            font-size: 20px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e8f5e9;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric .value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c5530;
        }}
        .metric .label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 500;
            margin: 10px 0;
        }}
        .status.success {{ background: #d4edda; color: #155724; }}
        .status.info {{ background: #cce5ff; color: #004085; }}
        .status.warning {{ background: #fff3cd; color: #856404; }}
        .status.danger {{ background: #f8d7da; color: #721c24; }}
        .chart {{
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
            display: block;
            border-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}
        .quality-list {{
            list-style: none;
            padding: 0;
        }}
        .quality-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .quality-list li:last-child {{ border-bottom: none; }}
        .quality-list .label {{ color: #666; }}
        .quality-list .value {{ float: right; font-weight: 500; }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 12px;
        }}
        @media print {{
            body {{ background: white; }}
            .report {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="report">
        <div class="header">
            <h1>Satellite Analysis Report</h1>
            <div class="subtitle">{loc_name} | {days}-Day Analysis</div>
        </div>

        <div class="section">
            <h2>Location Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{lat:.4f}°</div>
                    <div class="label">Latitude</div>
                </div>
                <div class="metric">
                    <div class="value">{lon:.4f}°</div>
                    <div class="label">Longitude</div>
                </div>
                <div class="metric">
                    <div class="value">{location.utm_zone}</div>
                    <div class="label">UTM Zone</div>
                </div>
                <div class="metric">
                    <div class="value">{days}</div>
                    <div class="label">Analysis Days</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Vegetation Health Analysis</h2>
            <div class="status {ndvi_class}">{ndvi_status}</div>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{veg_result.mean_ndvi:.2f}</div>
                    <div class="label">Mean NDVI</div>
                </div>
                <div class="metric">
                    <div class="value">±{veg_result.ndvi_std:.2f}</div>
                    <div class="label">Std Deviation</div>
                </div>
                <div class="metric">
                    <div class="value">{veg_result.confidence:.0%}</div>
                    <div class="label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="value">{veg_result.cloud_free_count}/{veg_result.observation_count}</div>
                    <div class="label">Cloud-Free Passes</div>
                </div>
            </div>
            <img src="data:image/png;base64,{veg_b64}" alt="Vegetation Health Chart" class="chart">
        </div>
"""

    # Change detection section (if available)
    if change_result is not None:
        html += f"""
        <div class="section">
            <h2>Vegetation Change Detection</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{change_result.period_1_ndvi:.2f}</div>
                    <div class="label">Baseline NDVI</div>
                </div>
                <div class="metric">
                    <div class="value">{change_result.period_2_ndvi:.2f}</div>
                    <div class="label">Current NDVI</div>
                </div>
                <div class="metric">
                    <div class="value">{change_result.delta:+.2f}</div>
                    <div class="label">Change (Δ)</div>
                </div>
                <div class="metric">
                    <div class="value">{change_result.direction.upper()}</div>
                    <div class="label">Direction</div>
                </div>
            </div>
            <img src="data:image/png;base64,{change_b64}" alt="Change Detection Chart" class="chart">
        </div>
"""

    # Weather section (if available)
    if weather_result.confidence > 0:
        html += f"""
        <div class="section">
            <h2>Weather Context</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{weather_result.mean_temperature:.1f}°C</div>
                    <div class="label">Mean Temperature</div>
                </div>
                <div class="metric">
                    <div class="value">{weather_result.temperature_min:.1f}° - {weather_result.temperature_max:.1f}°</div>
                    <div class="label">Temperature Range</div>
                </div>
                <div class="metric">
                    <div class="value">{weather_result.total_precipitation:.1f} mm</div>
                    <div class="label">Total Precipitation</div>
                </div>
                <div class="metric">
                    <div class="value">{weather_result.observation_count}</div>
                    <div class="label">Days of Data</div>
                </div>
            </div>
            <img src="data:image/png;base64,{weather_b64}" alt="Weather Chart" class="chart">
        </div>
"""
    else:
        html += """
        <div class="section">
            <h2>Weather Context</h2>
            <div class="warning-box">
                Weather data not available. CDS credentials may not be configured.
            </div>
        </div>
"""

    # Data quality section
    warnings = []
    if veg_result.warnings:
        warnings.extend([f"Vegetation: {w}" for w in veg_result.warnings])
    if change_result is not None and change_result.warnings:
        warnings.extend([f"Change: {w}" for w in change_result.warnings])
    if weather_result.warnings:
        warnings.extend([f"Weather: {w}" for w in weather_result.warnings])

    change_conf_html = ""
    if change_result is not None:
        change_conf_html = f"""
                <li>
                    <span class="label">Change Detection Confidence</span>
                    <span class="value">{change_result.confidence:.0%}</span>
                </li>"""

    html += f"""
        <div class="section">
            <h2>Data Quality</h2>
            <ul class="quality-list">
                <li>
                    <span class="label">Vegetation Confidence</span>
                    <span class="value">{veg_result.confidence:.0%}</span>
                </li>{change_conf_html}
                <li>
                    <span class="label">Weather Confidence</span>
                    <span class="value">{weather_result.confidence:.0%}</span>
                </li>
                <li>
                    <span class="label">Satellite Passes (Total)</span>
                    <span class="value">{veg_result.observation_count}</span>
                </li>
                <li>
                    <span class="label">Cloud-Free Passes</span>
                    <span class="value">{veg_result.cloud_free_count}</span>
                </li>
            </ul>
"""

    if warnings:
        html += """
            <h3 style="margin-top: 20px; color: #856404;">Warnings</h3>
"""
        for w in warnings:
            html += f"""
            <div class="warning-box">{w}</div>
"""

    html += f"""
        </div>

        <div class="footer">
            <p>Generated with SatelliteHub v{sh.__version__}</p>
            <p>Report Time: {report_time}</p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    output_path.write_text(html, encoding="utf-8")
    print(f"\n[OK] Report saved to: {output_path.absolute()}")

    # Cleanup temp files
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Generate HTML analysis report from satellite data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --lat 51.41 --lon 21.97 --days 30 --output report.html
  python run_analysis.py --lat 52.23 --lon 21.01 --days 90 --name "Warsaw" -o warsaw.html
        """,
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude in WGS84 degrees (e.g., 51.41)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude in WGS84 degrees (e.g., 21.97)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="report.html",
        help="Output HTML file path (default: report.html)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Human-readable location name (optional)",
    )

    args = parser.parse_args()

    # Validate coordinates
    if not -90 <= args.lat <= 90:
        print(f"Error: Invalid latitude {args.lat}. Must be between -90 and 90.")
        sys.exit(1)
    if not -180 <= args.lon <= 180:
        print(f"Error: Invalid longitude {args.lon}. Must be between -180 and 180.")
        sys.exit(1)
    if args.days < 1:
        print(f"Error: Days must be at least 1, got {args.days}.")
        sys.exit(1)

    output_path = Path(args.output)

    try:
        generate_html_report(
            lat=args.lat,
            lon=args.lon,
            days=args.days,
            output_path=output_path,
            location_name=args.name,
        )
    except Exception as e:
        print(f"\nError generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
