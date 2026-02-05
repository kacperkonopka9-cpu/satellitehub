"""Configuration and credential management for SatelliteHub.

Implements AD-1 (configuration injection) and AD-8 (credential storage)
from the architecture specification.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from satellitehub.exceptions import ConfigurationError

logger = logging.getLogger("satellitehub")

_CREDENTIALS_ENV_VAR = "SATELLITEHUB_CREDENTIALS"
_DEFAULT_CREDENTIALS_PATH = Path("~/.satellitehub/credentials.json")


class Config(BaseModel):
    """SDK configuration model (AD-1).

    Immutable pydantic model storing SDK settings. Each ``Location``
    captures a snapshot of the active ``Config`` at creation time so
    later ``configure()`` calls never affect existing locations.

    Args:
        copernicus_credentials: Path to Copernicus credentials file.
        cds_credentials: Path to CDS API credentials file.
        cache_dir: Local directory for cached data.
        cache_size_mb: Maximum cache size in megabytes.
        default_crs: Default coordinate reference system for outputs.

    Example:
        >>> cfg = Config(cache_dir="~/my-cache", cache_size_mb=2000)
        >>> cfg.default_crs
        'EPSG:4326'
    """

    model_config = ConfigDict(frozen=True, validate_default=True, extra="forbid")

    copernicus_credentials: Path | None = None
    cds_credentials: Path | None = None
    cache_dir: Path = Path("~/.satellitehub/cache")
    cache_size_mb: int = 5000
    default_crs: str = "EPSG:4326"

    @field_validator("copernicus_credentials", "cds_credentials", mode="before")
    @classmethod
    def _expand_credential_paths(
        cls,
        v: str | Path | None,
    ) -> Path | None:
        """Expand ``~`` in credential paths."""
        if v is None:
            return None
        return Path(v).expanduser()

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _expand_cache_dir(cls, v: str | Path) -> Path:
        """Expand ``~`` in cache directory path."""
        return Path(v).expanduser()

    @field_validator("cache_size_mb")
    @classmethod
    def _validate_cache_size(cls, v: int) -> int:
        """Ensure cache size is positive."""
        if v <= 0:
            msg = "cache_size_mb must be greater than 0"
            raise ValueError(msg)
        return v

    @field_validator("default_crs")
    @classmethod
    def _validate_crs(cls, v: str) -> str:
        """Ensure CRS matches EPSG format."""
        if not re.match(r"^EPSG:\d+$", v):
            msg = "default_crs must match 'EPSG:<number>' format"
            raise ValueError(msg)
        return v


_default_config = Config()


def configure(**kwargs: Any) -> None:
    """Set module-level default configuration.

    Creates a new ``Config`` from the current defaults merged with
    the provided keyword arguments.

    Args:
        **kwargs: Any ``Config`` field (e.g. ``cache_dir``,
            ``copernicus_credentials``, ``cache_size_mb``).

    Raises:
        ValidationError: If a provided value fails pydantic validation.

    Example:
        >>> configure(cache_size_mb=2000, default_crs="EPSG:32634")
    """
    global _default_config  # noqa: PLW0603
    current = _default_config.model_dump()
    current.update(kwargs)
    _default_config = Config(**current)


def get_default_config() -> Config:
    """Return the current module-level default configuration.

    Returns:
        The active ``Config`` instance.
    """
    return _default_config


def resolve_credentials_path(
    explicit: Path | None = None,
) -> Path | None:
    """Resolve the credentials file path using the AD-8 priority chain.

    Resolution order:
        1. *explicit* argument (highest priority)
        2. ``SATELLITEHUB_CREDENTIALS`` environment variable
        3. Default ``~/.satellitehub/credentials.json``

    After resolution, emits a warning if the file exists and has
    world-readable permissions on POSIX systems (NFR5).

    Args:
        explicit: An explicit path passed via ``Config``.

    Returns:
        Resolved ``Path``, or ``None`` if no credentials file exists
        at any of the candidate locations.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
    elif os.environ.get(_CREDENTIALS_ENV_VAR):
        path = Path(os.environ[_CREDENTIALS_ENV_VAR]).expanduser()
    else:
        path = _DEFAULT_CREDENTIALS_PATH.expanduser()

    if not path.exists():
        return None

    _check_file_permissions(path)
    return path


def _check_file_permissions(path: Path) -> None:
    """Warn if *path* is readable by group or others (NFR5).

    Skipped on Windows where POSIX permission bits are not meaningful.
    """
    if sys.platform == "win32":
        return
    try:
        mode = path.stat().st_mode
        if mode & 0o077:
            logger.warning(
                "Credentials file %s has overly permissive "
                "permissions (%o). Consider running: "
                "chmod 600 %s",
                path,
                mode & 0o777,
                path,
            )
    except OSError:
        pass


def load_credentials(path: Path) -> dict[str, Any]:
    """Load and parse a JSON credentials file.

    Args:
        path: Absolute or ``~``-expanded path to the JSON file.

    Returns:
        Parsed credentials dictionary.

    Raises:
        ConfigurationError: If the file is missing or contains
            invalid JSON.
    """
    resolved = Path(path).expanduser()
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigurationError(
            what="Cannot read credentials file",
            cause=f"File not found: {resolved}",
            fix=(
                f"Create {resolved} with provider credentials, "
                f"or set the {_CREDENTIALS_ENV_VAR} environment variable"
            ),
        ) from None
    except PermissionError:
        raise ConfigurationError(
            what="Cannot read credentials file",
            cause=f"Permission denied: {resolved}",
            fix=f"Check file permissions on {resolved}",
        ) from None

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            what="Invalid credentials file format",
            cause=f"JSON parse error in {resolved}: {exc}",
            fix=(
                "Ensure the file contains valid JSON with structure: "
                '{"copernicus": {"username": "...", "password": "..."}, '
                '"cds": {"api_key": "..."}}'
            ),
        ) from None

    if not isinstance(parsed, dict):
        raise ConfigurationError(
            what="Invalid credentials file format",
            cause=f"Expected a JSON object in {resolved}, got {type(parsed).__name__}",
            fix=(
                "Ensure the file contains a JSON object with structure: "
                '{"copernicus": {"username": "...", "password": "..."}, '
                '"cds": {"api_key": "..."}}'
            ),
        )

    return parsed
