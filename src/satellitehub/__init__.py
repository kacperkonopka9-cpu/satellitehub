"""SatelliteHub — Unified Python SDK for satellite data access and analysis."""

from satellitehub.__about__ import __version__
from satellitehub.config import Config, configure
from satellitehub.exceptions import (
    CacheError,
    ConfigurationError,
    ProviderError,
    SatelliteHubError,
)
from satellitehub.location import location
from satellitehub.results import (
    BaseResult,
    ChangeResult,
    ResultMetadata,
    VegetationResult,
    WeatherResult,
)

__all__ = [
    "__version__",
    "BaseResult",
    "CacheError",
    "ChangeResult",
    "Config",
    "ConfigurationError",
    "ProviderError",
    "ResultMetadata",
    "SatelliteHubError",
    "VegetationResult",
    "WeatherResult",
    "configure",
    "location",
]
