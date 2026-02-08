# SatelliteHub

Unified Python SDK for satellite data access and vegetation analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version: 0.2.0](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/kacperkonopka/satellitehub)

## Features

- **Vegetation Health Analysis** - NDVI-based health assessment with cloud masking
- **Change Detection** - Compare vegetation between time periods
- **Weather Integration** - ERA5 reanalysis (NetCDF parsing) and IMGW Polish stations
- **Landsat Thermal Bands** - Land surface temperature analysis (B10/B11/ST_B10)
- **Parallel Downloads** - Concurrent band downloads for improved performance
- **Geo Metadata** - CRS, transform, and bounds in download results
- **Export Options** - DataFrame, GeoTIFF, PNG, HTML reports
- **Two-Tier API** - High-level semantic methods + low-level data access

## Installation

```bash
pip install satellitehub
```

## Quick Start

```python
import satellitehub as sh

# Create a location (Warsaw)
loc = sh.location(lat=52.23, lon=21.01)

# Analyze vegetation health
result = loc.vegetation_health(last_days=30)

print(f"NDVI: {result.mean_ndvi:.2f}")
print(f"Confidence: {result.confidence:.0%}")
print(result.narrative())

# Export results
result.to_png("vegetation.png")
result.to_dataframe().to_csv("data.csv")
```

## Configuration

Create `~/.satellitehub/credentials.json`:

```json
{
  "cdse": {
    "username": "your-copernicus-email",
    "password": "your-copernicus-password"
  }
}
```

Register free at [Copernicus Data Space](https://dataspace.copernicus.eu/).

## CLI Report Generator

```bash
python -m satellitehub.scripts.run_analysis \
  --lat 52.23 --lon 21.01 \
  --days 30 \
  --name "Warsaw" \
  --output report.html
```

## API Overview

### Semantic Methods (High-Level)

```python
loc = sh.location(lat=52.23, lon=21.01)

# Vegetation analysis
loc.vegetation_health(last_days=30)
loc.vegetation_change(period_1=("2024-01-01", "2024-01-31"),
                      period_2=("2024-06-01", "2024-06-30"))

# Weather data
loc.weather(last_days=30)

# Check available methods
loc.available_methods()
```

### Data Tier (Low-Level)

```python
# Direct provider access
provider = loc.get_provider("cdse")
entries = provider.search(location=loc, time_range=("2024-01-01", "2024-01-31"))
raw_data = provider.download(entries[0], bands=["B04", "B08"])
```

### Landsat Thermal Analysis

```python
# Access Landsat thermal bands for land surface temperature
provider = loc.get_provider("landsat")
entries = provider.search(location=loc, time_range=("2024-06-01", "2024-06-30"))

# Download thermal bands (automatically converted to Celsius)
raw_data = provider.download(entries[0], bands=["B10", "ST_B10"])

print(f"CRS: {raw_data.metadata['crs']}")
print(f"Thermal unit: {raw_data.metadata['thermal_unit']}")  # 'celsius'
print(f"Temperature range: {raw_data.data.min():.1f}C to {raw_data.data.max():.1f}C")
```

## Data Sources

| Provider | Data | Features | Registration |
|----------|------|----------|--------------|
| CDSE | Sentinel-2 L2A | 10-60m optical, cloud masking | [Copernicus](https://dataspace.copernicus.eu/) |
| Landsat | Landsat 8/9 L2 | 30m optical + thermal (B10/B11) | No registration needed |
| CDS | ERA5 reanalysis | Temperature, precipitation (NetCDF) | [CDS](https://cds.climate.copernicus.eu/) |
| IMGW | Polish stations | Real-time synoptic data | No registration needed |

## Development

```bash
git clone https://github.com/kacperkonopka/satellitehub
cd satellitehub
pip install -e ".[all]"
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
