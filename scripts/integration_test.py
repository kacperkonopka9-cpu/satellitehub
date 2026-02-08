#!/usr/bin/env python3
"""Comprehensive integration test for SatelliteHub MVP.

Tests all major functionality with real API credentials:
- Location creation
- Provider status checks
- Weather data (ERA5 + IMGW)
- Vegetation health analysis
- Result exports (DataFrame, PNG)
- Method discovery

Usage:
    python scripts/integration_test.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

print("=" * 60)
print("SatelliteHub MVP Integration Test")
print("=" * 60)
print(f"Started: {datetime.now().isoformat()}")
print()

# Step 1: Import and basic setup
print("[1/8] Importing satellitehub...")
try:
    import satellitehub as sh
    print(f"  [OK] Version: {sh.__version__}")
except ImportError as e:
    print(f"  [FAIL] Import failed: {e}")
    sys.exit(1)

# Step 2: Create location (Warsaw area)
print("\n[2/8] Creating location (Warsaw: 52.23N, 21.01E)...")
try:
    loc = sh.location(lat=52.23, lon=21.01)
    print("  [OK] Location created")
    print(f"  [OK] Bounds: {loc.bounds}")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    sys.exit(1)

# Step 3: Check provider status
print("\n[3/8] Checking provider status...")
try:
    methods = loc.available_methods()
    print(f"  [OK] Found {len(methods)} methods")
    for m in methods:
        status = "[OK]" if m.available else "[--]"
        desc = m.description[:50] if m.description else "No description"
        print(f"    {status} {m.name}: {desc}...")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")

# Step 4: Weather data test (ERA5)
print("\n[4/8] Testing weather data (last 7 days)...")
weather = None
try:
    weather = loc.weather(last_days=7)
    print("  [OK] WeatherResult received")
    print(f"    - Confidence: {weather.confidence:.2f}")
    print(f"    - Data source: {weather.data_source}")
    if weather.confidence > 0:
        print(f"    - Mean temperature: {weather.mean_temperature:.1f}C")
        print(f"    - Total precipitation: {weather.total_precipitation:.1f}mm")
        print(f"    - Observations: {weather.observation_count}")
    else:
        print(f"    - Warnings: {weather.warnings}")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Raw ERA5 data tier test
print("\n[5/8] Testing raw ERA5 data tier...")
era5_raw = None
try:
    era5_raw = loc.data.era5(variables=["2m_temperature"], last_days=3)
    print("  [OK] ERA5 raw data received")
    print(f"    - Confidence: {era5_raw.confidence:.2f}")
    print(f"    - Data shape: {era5_raw.data.shape}")
    if era5_raw.confidence > 0:
        print(f"    - Timestamps: {len(era5_raw.metadata.timestamps)}")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()

# Step 6: Vegetation health test
print("\n[6/8] Testing vegetation health (last 30 days)...")
veg = None
try:
    veg = loc.vegetation_health(last_days=30)
    print("  [OK] VegetationResult received")
    print(f"    - Confidence: {veg.confidence:.2f}")
    if veg.confidence > 0:
        print(f"    - Mean NDVI: {veg.mean_ndvi:.3f}")
        print(f"    - Cloud-free observations: {veg.cloud_free_count}/{veg.observation_count}")
    else:
        warnings_preview = veg.warnings[:2] if veg.warnings else ["No warnings"]
        print(f"    - Warnings: {warnings_preview}...")
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Export tests
print("\n[7/8] Testing result exports...")
with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)

    # DataFrame export
    try:
        if weather is not None:
            df = weather.to_dataframe()
            print(f"  [OK] to_dataframe(): {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"  [FAIL] to_dataframe() failed: {e}")

    # PNG export
    try:
        if weather is not None:
            png_path = weather.to_png(tmppath / "weather.png")
            print(f"  [OK] to_png(): {png_path.name} ({png_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"  [FAIL] to_png() failed: {e}")

    # Vegetation PNG
    try:
        if veg is not None:
            veg_png = veg.to_png(tmppath / "vegetation.png")
            print(f"  [OK] Vegetation to_png(): {veg_png.name}")
    except Exception as e:
        print(f"  [FAIL] Vegetation to_png() failed: {e}")

# Step 8: Summary
print("\n[8/8] Integration Test Summary")
print("=" * 60)

results = {
    "Import": True,
    "Location": loc is not None,
    "Methods": 'methods' in dir() and len(methods) > 0,
    "Weather": weather is not None,
    "ERA5 Raw": era5_raw is not None,
    "Vegetation": veg is not None,
}

passed = sum(results.values())
total = len(results)

for test, result in results.items():
    status = "[PASS]" if result else "[FAIL]"
    print(f"  {status}: {test}")

print()
print(f"Result: {passed}/{total} tests passed")
print(f"Completed: {datetime.now().isoformat()}")
print("=" * 60)

if passed < total:
    sys.exit(1)
