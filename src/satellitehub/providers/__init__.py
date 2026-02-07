"""Provider registry for data source access.

Provides ``get_provider()`` to instantiate configured provider
instances by name. Supports CDSE (Sentinel-2), CDS (ERA5),
IMGW (Polish weather), and Landsat (Landsat 8/9 via Planetary Computer).
"""

from __future__ import annotations

from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError
from satellitehub.providers.base import DataProvider

_PROVIDER_REGISTRY: dict[str, type[DataProvider]] = {}
_REGISTRY_INITIALIZED = False


def _init_registry() -> None:
    """Populate the provider registry on first use (lazy import)."""
    global _REGISTRY_INITIALIZED  # noqa: PLW0603
    if _REGISTRY_INITIALIZED:
        return

    from satellitehub.providers.cds import CDSProvider
    from satellitehub.providers.cdse import CDSEProvider
    from satellitehub.providers.imgw import IMGWProvider
    from satellitehub.providers.landsat import LandsatProvider

    _PROVIDER_REGISTRY.update(
        {
            "cdse": CDSEProvider,
            "cds": CDSProvider,
            "imgw": IMGWProvider,
            "landsat": LandsatProvider,
        }
    )
    _REGISTRY_INITIALIZED = True


def get_registered_names() -> list[str]:
    """Return sorted list of registered provider names.

    Initializes the registry if not already done.

    Returns:
        Sorted list of provider identifier strings.
    """
    _init_registry()
    return sorted(_PROVIDER_REGISTRY)


def get_provider(name: str, config: Config) -> DataProvider:
    """Return a configured provider instance by name.

    Instantiates the provider class with the given configuration.
    Provider names are case-insensitive.

    Args:
        name: Provider identifier (``"cdse"``, ``"cds"``, ``"imgw"``, or ``"landsat"``).
        config: Frozen configuration snapshot from the Location.

    Returns:
        A configured ``DataProvider`` instance ready for use.

    Raises:
        ConfigurationError: If *name* does not match a registered provider.

    Example:
        >>> from satellitehub.config import Config
        >>> provider = get_provider("cdse", Config())
        >>> provider.name
        'cdse'
    """
    _init_registry()
    key = name.lower()
    if key not in _PROVIDER_REGISTRY:
        valid = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ConfigurationError(
            what=f"Unknown provider: {name!r}",
            cause=f"Valid providers are: {valid}",
            fix=f"Use one of: {valid}",
        )
    return _PROVIDER_REGISTRY[key](config=config)
