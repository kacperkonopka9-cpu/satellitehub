"""SatelliteHub — Unified Python SDK for satellite data access and analysis.

Example:
    >>> import satellitehub as sh
    >>>
    >>> # Quick analysis with just coordinates
    >>> result = sh.vegetation_health(52.23, 21.01, last_days=30)
    >>> print(f"NDVI: {result.mean_ndvi:.2f}")
    >>>
    >>> # Or use a Location for multiple analyses
    >>> loc = sh.location(52.23, 21.01)
    >>> veg = sh.vegetation_health(loc)
    >>> weather = sh.weather(loc, last_days=7)
"""

from satellitehub.__about__ import __version__
from satellitehub.api import change_detection, vegetation_health, weather
from satellitehub.config import Config, configure
from satellitehub.exceptions import (
    CacheError,
    ConfigurationError,
    ProviderError,
    SatelliteHubError,
)
from satellitehub.location import Location, location
from satellitehub.results import (
    BaseResult,
    ChangeResult,
    ResultMetadata,
    VegetationResult,
    WeatherResult,
)

__all__ = [
    # Version
    "__version__",
    # Semantic API (top-level functions)
    "change_detection",
    "vegetation_health",
    "weather",
    # Location
    "Location",
    "location",
    # Configuration
    "Config",
    "configure",
    # Results
    "BaseResult",
    "ChangeResult",
    "ResultMetadata",
    "VegetationResult",
    "WeatherResult",
    # Exceptions
    "CacheError",
    "ConfigurationError",
    "ProviderError",
    "SatelliteHubError",
]
