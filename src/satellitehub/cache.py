"""Cache subsystem with SQLite metadata index and file-based storage.

Implements AD-3 (cache architecture) from the architecture specification.
The cache stores and retrieves opaque bytes keyed by a composite cache key.
It sits between the pipeline and providers at Boundary 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from satellitehub._types import TimeRange
    from satellitehub.config import Config

logger = logging.getLogger("satellitehub")


@dataclass
class CacheEntry:
    """Metadata for a single cached data file.

    Each entry maps a composite cache key to a file on disk, tracking
    timestamps for TTL expiration and LRU eviction.

    Args:
        cache_key: Composite key
            ``{provider}:{product}:{location_hash}:{start}:{end}:{params_hash}``.
        provider: Data provider name (e.g. ``"cdse"``).
        product: Data product identifier (e.g. ``"S2"``).
        location_hash: Hex digest identifying the geographic location.
        created_at: ISO-8601 UTC timestamp when the entry was created.
        last_accessed_at: ISO-8601 UTC timestamp of the most recent read.
        expires_at: ISO-8601 UTC timestamp when the entry becomes stale.
        file_path: Absolute path to the cached data file on disk.
        size_bytes: Size of the cached file in bytes.
        params_json: JSON-serialized additional parameters used to build the key.

    Example:
        >>> entry = CacheEntry(
        ...     cache_key="cdse:S2:abc123:2024-01-01:2024-01-31:def456",
        ...     provider="cdse", product="S2", location_hash="abc123",
        ...     created_at="2024-01-15T10:00:00+00:00",
        ...     last_accessed_at="2024-01-15T10:00:00+00:00",
        ...     expires_at="2024-01-16T10:00:00+00:00",
        ...     file_path="/cache/cdse/sha256.dat",
        ...     size_bytes=1024, params_json="{}",
        ... )
    """

    __slots__ = (
        "cache_key",
        "provider",
        "product",
        "location_hash",
        "created_at",
        "last_accessed_at",
        "expires_at",
        "file_path",
        "size_bytes",
        "params_json",
    )

    cache_key: str
    provider: str
    product: str
    location_hash: str
    created_at: str
    last_accessed_at: str
    expires_at: str
    file_path: str
    size_bytes: int
    params_json: str


@dataclass
class CacheStatus:
    """Summary statistics for the cache.

    Args:
        entry_count: Number of entries currently in the cache.
        total_size_bytes: Combined size of all cached files in bytes.
        oldest_entry: ISO-8601 UTC timestamp of the oldest entry, or
            empty string if the cache is empty.

    Example:
        >>> status = CacheStatus(entry_count=0, total_size_bytes=0, oldest_entry="")
        >>> status.entry_count
        0
    """

    __slots__ = ("entry_count", "total_size_bytes", "oldest_entry")

    entry_count: int
    total_size_bytes: int
    oldest_entry: str


_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    product TEXT NOT NULL,
    location_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    file_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    params_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider ON cache_entries(provider);
CREATE INDEX IF NOT EXISTS idx_expires ON cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_lru ON cache_entries(last_accessed_at);
"""


class CacheManager:
    """Manages local caching of downloaded satellite and weather data.

    Uses an SQLite metadata index for fast lookups and stores data as
    opaque files on the local filesystem. Supports TTL-based expiration
    and LRU eviction when the cache exceeds its configured size limit.

    Cache methods **never raise exceptions** to callers. All errors are
    caught internally and logged as warnings (NFR15).

    Args:
        config: SDK configuration providing ``cache_dir`` and
            ``cache_size_mb`` settings.

    Example:
        >>> from satellitehub.config import Config
        >>> mgr = CacheManager(config=Config(cache_dir="/tmp/test-cache"))
        >>> mgr.status()
        CacheStatus(entry_count=0, total_size_bytes=0, oldest_entry='')
    """

    def __init__(self, config: Config) -> None:
        """Initialize cache manager with configuration.

        Creates the cache directory and SQLite database on first use.

        Args:
            config: SDK configuration captured at Location creation.
        """
        self._config = config
        self._cache_dir = Path(config.cache_dir)
        self._db_path = self._cache_dir / "cache.db"
        self._init_db()

    def _init_db(self) -> None:
        """Create the cache directory and database schema if needed."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.executescript(_CREATE_TABLE_SQL)
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cache database initialization failed: %s", exc)

    def _get_connection(self) -> sqlite3.Connection:
        """Open a new SQLite connection to the cache database.

        Returns:
            A fresh ``sqlite3.Connection``.
        """
        return sqlite3.connect(str(self._db_path))

    def _build_cache_key(
        self,
        provider: str,
        product: str,
        location_hash: str,
        time_range: TimeRange,
        params: dict[str, str],
    ) -> str:
        """Build a deterministic composite cache key.

        Args:
            provider: Data provider name.
            product: Data product identifier.
            location_hash: Hex digest for the geographic location.
            time_range: ISO-8601 date pair ``(start, end)``.
            params: Additional parameters to incorporate in the key.

        Returns:
            Composite key string
            ``{provider}:{product}:{location_hash}:{start}:{end}:{params_hash}``.

        Example:
            >>> mgr._build_cache_key(
            ...     "cdse", "S2", "abc", ("2024-01-01", "2024-01-31"), {}
            ... )  # doctest: +SKIP
            'cdse:S2:abc:2024-01-01:2024-01-31:44136fa...'
        """
        params_json = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()
        return (
            f"{provider}:{product}:{location_hash}"
            f":{time_range[0]}:{time_range[1]}:{params_hash}"
        )

    def _key_to_path(self, cache_key: str, provider: str) -> Path:
        """Compute the filesystem path for a cache key.

        Args:
            cache_key: The full composite cache key.
            provider: Data provider name (used as subdirectory).

        Returns:
            Path ``{cache_dir}/{provider}/{sha256_of_cache_key}.dat``.
        """
        key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        return self._cache_dir / provider / f"{key_hash}.dat"

    def get(self, cache_key: str) -> bytes | None:
        """Look up cached data by key.

        Returns the cached bytes if the entry exists, has not expired,
        and the backing file is present. Otherwise returns ``None``.

        This method **never raises** — all errors are logged internally.

        Args:
            cache_key: The composite cache key to look up.

        Returns:
            Cached data bytes, or ``None`` if not found / expired / corrupt.
        """
        try:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT expires_at, file_path FROM cache_entries "
                    "WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    return None

                expires_at_str: str = row[0]
                file_path_str: str = row[1]

                # Check TTL expiration
                now = datetime.now(timezone.utc)
                expires_at = datetime.fromisoformat(expires_at_str)
                if now >= expires_at:
                    self._delete_entry(conn, cache_key, file_path_str)
                    conn.commit()
                    return None

                # Check file exists
                file_path = Path(file_path_str)
                if not file_path.exists():
                    logger.warning(
                        "Cache file missing for key %s at %s — removing stale entry",
                        cache_key,
                        file_path_str,
                    )
                    self._delete_entry(conn, cache_key, file_path_str)
                    conn.commit()
                    return None

                # Read file and update last_accessed_at
                data = file_path.read_bytes()
                now_iso = now.isoformat()
                conn.execute(
                    "UPDATE cache_entries SET last_accessed_at = ? WHERE cache_key = ?",
                    (now_iso, cache_key),
                )
                conn.commit()
                return data
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cache lookup failed for key %s: %s", cache_key, exc)
            return None

    def store(
        self,
        cache_key: str,
        provider: str,
        product: str,
        location_hash: str,
        data: bytes,
        ttl_hours: float = 24.0,
        params: dict[str, str] | None = None,
    ) -> None:
        """Store data in the cache.

        Writes the data file atomically and creates an SQLite metadata
        entry. Triggers LRU eviction if total cache size exceeds the
        configured limit.

        This method **never raises** — all errors are logged internally.

        Args:
            cache_key: The composite cache key.
            provider: Data provider name.
            product: Data product identifier.
            location_hash: Hex digest for the geographic location.
            data: Raw bytes to cache.
            ttl_hours: Time-to-live in hours (default 24.0 for satellite,
                use 6.0 for weather).
            params: Original query parameters for audit metadata. Stored
                as JSON in the SQLite entry. Defaults to empty dict.
        """
        try:
            file_path = self._key_to_path(cache_key, provider)

            # Create provider subdirectory
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to .tmp, then rename
            tmp_path = file_path.with_suffix(".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(file_path)

            # Insert/replace SQLite entry
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            expires_iso = (now + timedelta(hours=ttl_hours)).isoformat()
            params_json = json.dumps(
                params if params is not None else {}, sort_keys=True
            )

            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "(cache_key, provider, product, location_hash, "
                    "created_at, last_accessed_at, expires_at, "
                    "file_path, size_bytes, params_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        cache_key,
                        provider,
                        product,
                        location_hash,
                        now_iso,
                        now_iso,
                        expires_iso,
                        str(file_path),
                        len(data),
                        params_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            # Check eviction after successful store
            self._maybe_evict()

        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cache store failed for key %s: %s", cache_key, exc)

    def _delete_entry(
        self, conn: sqlite3.Connection, cache_key: str, file_path_str: str
    ) -> None:
        """Delete a cache entry and its backing file.

        Args:
            conn: Active SQLite connection (caller manages commit).
            cache_key: The cache key to remove.
            file_path_str: Path to the data file on disk.
        """
        conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
        try:
            file_path = Path(file_path_str)
            if file_path.exists():
                file_path.unlink()
        except OSError as exc:
            logger.warning("Failed to delete cache file %s: %s", file_path_str, exc)

    def _maybe_evict(self) -> None:
        """Evict least-recently-accessed entries if cache exceeds size limit.

        Queries total cache size and removes entries ordered by
        ``last_accessed_at`` ascending (LRU) until total size is within
        the configured ``cache_size_mb`` limit.
        """
        try:
            max_bytes = self._config.cache_size_mb * 1024 * 1024
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries"
                ).fetchone()
                total_size: int = row[0] if row else 0

                if total_size <= max_bytes:
                    return

                # Fetch all LRU-ordered entries before mutating rows
                entries = conn.execute(
                    "SELECT cache_key, file_path, size_bytes "
                    "FROM cache_entries ORDER BY last_accessed_at ASC"
                ).fetchall()
                evicted = 0
                for entry_row in entries:
                    if total_size <= max_bytes:
                        break
                    ck: str = entry_row[0]
                    fp: str = entry_row[1]
                    sz: int = entry_row[2]
                    self._delete_entry(conn, ck, fp)
                    total_size -= sz
                    evicted += 1

                conn.commit()
                if evicted > 0:
                    logger.info(
                        "Cache eviction: removed %d entries to stay within %d MB limit",
                        evicted,
                        self._config.cache_size_mb,
                    )
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cache eviction failed: %s", exc)

    def status(self) -> CacheStatus:
        """Return summary statistics for the cache.

        This method **never raises** — returns zeros on any error.

        Returns:
            ``CacheStatus`` with entry count, total size, and oldest entry
            timestamp.

        Example:
            >>> mgr.status()
            CacheStatus(entry_count=0, total_size_bytes=0, oldest_entry='')
        """
        try:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), "
                    "COALESCE(MIN(created_at), '') FROM cache_entries"
                ).fetchone()
                if row is None:
                    return CacheStatus(
                        entry_count=0, total_size_bytes=0, oldest_entry=""
                    )
                logger.debug(
                    "Cache status: %d entries, %d bytes",
                    row[0],
                    row[1],
                )
                return CacheStatus(
                    entry_count=row[0],
                    total_size_bytes=row[1],
                    oldest_entry=row[2],
                )
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cache status query failed: %s", exc)
            return CacheStatus(entry_count=0, total_size_bytes=0, oldest_entry="")

    def clear(self) -> None:
        """Remove all cache entries and their backing files.

        Deletes every data file in provider subdirectories, clears the
        SQLite table, and runs ``VACUUM`` to reclaim space.

        This method **never raises** — all errors are logged internally.
        """
        try:
            conn = self._get_connection()
            try:
                # Collect all file paths before deleting rows
                cursor = conn.execute("SELECT file_path FROM cache_entries")
                file_paths = [row[0] for row in cursor]

                for fp_str in file_paths:
                    fp = Path(fp_str)
                    try:
                        if fp.exists():
                            fp.unlink()
                    except OSError as exc:
                        logger.warning("Failed to delete cache file %s: %s", fp, exc)

                conn.execute("DELETE FROM cache_entries")
                conn.commit()
            finally:
                conn.close()

            # VACUUM must run outside a transaction
            conn = self._get_connection()
            try:
                conn.execute("VACUUM")
            finally:
                conn.close()

            logger.debug("Cache cleared")
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cache clear failed: %s", exc)
