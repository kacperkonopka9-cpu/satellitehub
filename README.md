# SatelliteHub

Unified Python SDK for satellite data access and vegetation analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Vegetation Health Analysis** - NDVI-based health assessment with cloud masking
- **Change Detection** - Compare vegetation between time periods
- **Weather Integration** - ERA5 reanalysis and IMGW Polish stations
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

## Data Sources

| Provider | Data | Registration |
|----------|------|--------------|
| CDSE | Sentinel-2 L2A imagery | [Copernicus](https://dataspace.copernicus.eu/) |
| Landsat | Landsat 8/9 L2 imagery | No registration needed |
| CDS | ERA5 weather reanalysis | [CDS](https://cds.climate.copernicus.eu/) |
| IMGW | Polish weather stations | No registration needed |

## Development

```bash
git clone https://github.com/kacperkonopka/satellitehub
cd satellitehub
pip install -e ".[all]"
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
