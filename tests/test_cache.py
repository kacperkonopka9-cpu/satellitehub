"""Tests for the cache subsystem (Story 2.3)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from satellitehub.cache import CacheEntry, CacheManager, CacheStatus
from satellitehub.config import Config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_mgr(tmp_path: Path) -> CacheManager:
    """Create a CacheManager with an isolated temporary directory."""
    cfg = Config(cache_dir=tmp_path, cache_size_mb=10)
    return CacheManager(config=cfg)


@pytest.fixture
def small_cache(tmp_path: Path) -> CacheManager:
    """Create a CacheManager with a very small size limit for eviction tests."""
    cfg = Config(cache_dir=tmp_path, cache_size_mb=1)
    return CacheManager(config=cfg)


# ---------------------------------------------------------------------------
# Task 8: Cache key building and path generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCacheKeyBuilding:
    """Tests for _build_cache_key and _key_to_path."""

    def test_build_cache_key_produces_composite_key(
        self, cache_mgr: CacheManager
    ) -> None:
        """_build_cache_key returns a deterministic composite key."""
        key = cache_mgr._build_cache_key(
            "cdse", "S2", "abc123", ("2024-01-01", "2024-01-31"), {}
        )
        parts = key.split(":")
        assert len(parts) == 6
        assert parts[0] == "cdse"
        assert parts[1] == "S2"
        assert parts[2] == "abc123"
        assert parts[3] == "2024-01-01"
        assert parts[4] == "2024-01-31"
        # params_hash is SHA-256 of '{}'
        expected_hash = hashlib.sha256(b"{}").hexdigest()
        assert parts[5] == expected_hash

    def test_same_inputs_produce_same_key(self, cache_mgr: CacheManager) -> None:
        """Identical inputs always produce the same cache key (NFR20)."""
        args: tuple[str, str, str, tuple[str, str], dict[str, str]] = (
            "cdse",
            "S2",
            "abc123",
            ("2024-01-01", "2024-01-31"),
            {"resolution": "10m"},
        )
        key1 = cache_mgr._build_cache_key(*args)
        key2 = cache_mgr._build_cache_key(*args)
        assert key1 == key2

    def test_different_params_produce_different_key(
        self, cache_mgr: CacheManager
    ) -> None:
        """Different parameters produce different cache keys."""
        key1 = cache_mgr._build_cache_key(
            "cdse",
            "S2",
            "abc123",
            ("2024-01-01", "2024-01-31"),
            {"resolution": "10m"},
        )
        key2 = cache_mgr._build_cache_key(
            "cdse",
            "S2",
            "abc123",
            ("2024-01-01", "2024-01-31"),
            {"resolution": "20m"},
        )
        assert key1 != key2

    def test_params_hash_uses_sorted_json(self, cache_mgr: CacheManager) -> None:
        """Key ordering in params dict does not affect the cache key."""
        key1 = cache_mgr._build_cache_key(
            "cdse",
            "S2",
            "abc",
            ("2024-01-01", "2024-01-31"),
            {"a": "1", "b": "2"},
        )
        key2 = cache_mgr._build_cache_key(
            "cdse",
            "S2",
            "abc",
            ("2024-01-01", "2024-01-31"),
            {"b": "2", "a": "1"},
        )
        assert key1 == key2

    def test_key_to_path_returns_correct_structure(
        self, cache_mgr: CacheManager
    ) -> None:
        """_key_to_path returns {cache_dir}/{provider}/{sha256}.dat."""
        cache_key = "cdse:S2:abc:2024-01-01:2024-01-31:hash123"
        path = cache_mgr._key_to_path(cache_key, "cdse")

        expected_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        assert path.parent.name == "cdse"
        assert path.name == f"{expected_hash}.dat"
        assert path.parent.parent == cache_mgr._cache_dir

    def test_key_to_path_sha256_is_of_full_key(self, cache_mgr: CacheManager) -> None:
        """Path SHA-256 is computed from the full composite key string."""
        cache_key = "cdse:S2:abc:2024-01-01:2024-01-31:params"
        path = cache_mgr._key_to_path(cache_key, "cdse")

        key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        assert path.stem == key_hash


# ---------------------------------------------------------------------------
# Task 9: Get/store round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetStoreRoundTrip:
    """Tests for store() and get() round-trip."""

    def test_store_then_get_returns_same_bytes(self, cache_mgr: CacheManager) -> None:
        """store() followed by get() returns the exact same data."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        data = b"satellite image bytes"

        cache_mgr.store(key, "cdse", "S2", "abc", data)
        result = cache_mgr.get(key)

        assert result == data

    def test_get_nonexistent_key_returns_none(self, cache_mgr: CacheManager) -> None:
        """get() for a key that was never stored returns None."""
        result = cache_mgr.get("nonexistent:key")
        assert result is None

    def test_get_updates_last_accessed_at(self, cache_mgr: CacheManager) -> None:
        """get() updates the last_accessed_at timestamp in SQLite."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"data")

        # Read initial timestamp
        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT last_accessed_at FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        initial_ts = row[0]
        conn.close()

        # Small delay then access
        time.sleep(0.05)
        cache_mgr.get(key)

        # Read updated timestamp
        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT last_accessed_at FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        updated_ts = row[0]
        conn.close()

        assert updated_ts > initial_ts

    def test_file_and_sqlite_entry_created(self, cache_mgr: CacheManager) -> None:
        """store() creates both a file on disk and an SQLite entry."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"data")

        # Check SQLite entry
        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT cache_key, provider, product, size_bytes "
            "FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == key
        assert row[1] == "cdse"
        assert row[2] == "S2"
        assert row[3] == 4  # len(b"data")

        # Check file exists
        file_path = cache_mgr._key_to_path(key, "cdse")
        assert file_path.exists()
        assert file_path.read_bytes() == b"data"

    def test_store_persists_params_json(self, cache_mgr: CacheManager) -> None:
        """store() writes params as JSON in the SQLite params_json column."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        params = {"resolution": "10m", "cloud_cover": "20"}
        cache_mgr.store(key, "cdse", "S2", "abc", b"data", params=params)

        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT params_json FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()

        assert row is not None
        stored_params = json.loads(row[0])
        assert stored_params == {"cloud_cover": "20", "resolution": "10m"}

    def test_store_without_params_stores_empty_dict(
        self, cache_mgr: CacheManager
    ) -> None:
        """store() with no params argument stores '{}' as params_json."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"data")

        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT params_json FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "{}"


# ---------------------------------------------------------------------------
# Task 10: TTL expiration
# ---------------------------------------------------------------------------


class _MockDatetime:
    """Mock datetime module that returns a fixed time from now()."""

    def __init__(self, fixed_now: datetime) -> None:
        self._fixed_now = fixed_now

    def now(self, tz: Any = None) -> datetime:  # noqa: ARG002
        """Return the fixed timestamp."""
        return self._fixed_now

    def fromisoformat(self, s: str) -> datetime:
        """Delegate to real datetime.fromisoformat."""
        return datetime.fromisoformat(s)


@pytest.mark.unit
class TestTTLExpiration:
    """Tests for TTL-based entry expiration."""

    def test_entry_within_ttl_is_returned(self, cache_mgr: CacheManager) -> None:
        """A freshly stored entry (within TTL) is returned by get()."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"data", ttl_hours=24.0)

        result = cache_mgr.get(key)
        assert result == b"data"

    def test_expired_entry_returns_none(
        self, cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired entry returns None and is cleaned up."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"data", ttl_hours=1.0)

        # Advance time by 2 hours
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        mock_dt = _MockDatetime(future)
        monkeypatch.setattr("satellitehub.cache.datetime", mock_dt)

        result = cache_mgr.get(key)
        assert result is None

        # Verify SQLite entry was cleaned up (restore real datetime for query)
        monkeypatch.undo()
        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT cache_key FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()
        assert row is None

    def test_custom_ttl_value(
        self, cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom TTL (e.g. 6 hours for weather) is respected."""
        key = "cds:ERA5:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cds", "ERA5", "abc", b"weather", ttl_hours=6.0)

        # 5 hours later: still valid
        future_5h = datetime.now(timezone.utc) + timedelta(hours=5)
        mock_dt = _MockDatetime(future_5h)
        monkeypatch.setattr("satellitehub.cache.datetime", mock_dt)
        assert cache_mgr.get(key) == b"weather"

        # 7 hours later: expired
        future_7h = datetime.now(timezone.utc) + timedelta(hours=7)
        mock_dt_7h = _MockDatetime(future_7h)
        monkeypatch.setattr("satellitehub.cache.datetime", mock_dt_7h)
        assert cache_mgr.get(key) is None


# ---------------------------------------------------------------------------
# Task 11: LRU eviction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLRUEviction:
    """Tests for LRU eviction when cache exceeds size limit."""

    def test_oldest_entries_evicted_when_over_limit(
        self, small_cache: CacheManager
    ) -> None:
        """When total exceeds cache_size_mb, LRU entries are evicted first."""
        # 1 MB limit. Store 3 entries of ~400 KB each (total ~1.2 MB > 1 MB)
        data_400k = b"x" * (400 * 1024)

        small_cache.store("key1", "cdse", "S2", "h1", data_400k)
        time.sleep(0.05)

        small_cache.store("key2", "cdse", "S2", "h2", data_400k)
        time.sleep(0.05)

        # Access key1 AFTER key2 is stored to make key1 more recently used
        small_cache.get("key1")
        time.sleep(0.05)

        small_cache.store("key3", "cdse", "S2", "h3", data_400k)

        # After storing key3, total ~1.2 MB > 1 MB limit.
        # LRU order: key2 (oldest access), key1 (accessed after key2), key3 (newest).
        # key2 should be evicted first.
        status = small_cache.status()
        max_bytes = 1 * 1024 * 1024
        assert status.total_size_bytes <= max_bytes

        # key2 should be evicted (oldest last_accessed_at)
        assert small_cache.get("key2") is None
        # key1 (accessed after key2) and key3 (newest) should survive
        assert small_cache.get("key1") is not None
        assert small_cache.get("key3") is not None

    def test_recently_accessed_entries_survive(self, small_cache: CacheManager) -> None:
        """Most recently accessed entries survive eviction."""
        data_600k = b"x" * (600 * 1024)

        # Store entry A
        small_cache.store("keyA", "cdse", "S2", "hA", data_600k)
        time.sleep(0.05)

        # Store entry B
        small_cache.store("keyB", "cdse", "S2", "hB", data_600k)
        # At this point total ~1.2 MB > 1 MB, eviction should remove keyA
        # since keyA was accessed least recently

        # keyB should survive (most recently stored/accessed)
        assert small_cache.get("keyB") is not None

    def test_eviction_deletes_file_and_sqlite_entry(
        self, small_cache: CacheManager
    ) -> None:
        """Eviction removes both the file and the SQLite entry."""
        data_600k = b"x" * (600 * 1024)

        small_cache.store("keyA", "cdse", "S2", "hA", data_600k)
        file_a = small_cache._key_to_path("keyA", "cdse")
        assert file_a.exists()
        time.sleep(0.05)

        # Store a second entry that triggers eviction
        small_cache.store("keyB", "cdse", "S2", "hB", data_600k)

        # keyA should be evicted — check file gone
        assert not file_a.exists()

        # Check SQLite entry gone
        conn = sqlite3.connect(str(small_cache._db_path))
        row = conn.execute(
            "SELECT cache_key FROM cache_entries WHERE cache_key = ?",
            ("keyA",),
        ).fetchone()
        conn.close()
        assert row is None


# ---------------------------------------------------------------------------
# Task 12: Corruption recovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorruptionRecovery:
    """Tests for cache corruption recovery (NFR17)."""

    def test_missing_file_returns_none_and_cleans_entry(
        self, cache_mgr: CacheManager
    ) -> None:
        """get() with missing file returns None and cleans the SQLite entry."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"data")

        # Delete the file manually
        file_path = cache_mgr._key_to_path(key, "cdse")
        file_path.unlink()

        result = cache_mgr.get(key)
        assert result is None

        # SQLite entry should be cleaned up
        conn = sqlite3.connect(str(cache_mgr._db_path))
        row = conn.execute(
            "SELECT cache_key FROM cache_entries WHERE cache_key = ?",
            (key,),
        ).fetchone()
        conn.close()
        assert row is None

    def test_corrupted_file_does_not_crash(self, cache_mgr: CacheManager) -> None:
        """A truncated/corrupted file still returns data (bytes are bytes)."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"original data")

        # Corrupt the file by truncating
        file_path = cache_mgr._key_to_path(key, "cdse")
        file_path.write_bytes(b"corrupt")

        # get() should return the corrupted bytes without crashing
        result = cache_mgr.get(key)
        assert result == b"corrupt"

    def test_after_recovery_restore_works(self, cache_mgr: CacheManager) -> None:
        """After corruption recovery, re-storing data works normally."""
        key = "cdse:S2:abc:2024-01-01:2024-01-31:hash"
        cache_mgr.store(key, "cdse", "S2", "abc", b"original")

        # Delete file to trigger recovery
        file_path = cache_mgr._key_to_path(key, "cdse")
        file_path.unlink()

        # get() triggers cleanup
        assert cache_mgr.get(key) is None

        # Re-store and verify
        cache_mgr.store(key, "cdse", "S2", "abc", b"new data")
        assert cache_mgr.get(key) == b"new data"


# ---------------------------------------------------------------------------
# Task 13: Status and clear
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStatusAndClear:
    """Tests for status() and clear()."""

    def test_status_returns_correct_counts(self, cache_mgr: CacheManager) -> None:
        """status() returns correct counts after multiple stores."""
        cache_mgr.store("key1", "cdse", "S2", "h1", b"aaaa")
        cache_mgr.store("key2", "cdse", "S2", "h2", b"bbbbbb")

        st = cache_mgr.status()
        assert st.entry_count == 2
        assert st.total_size_bytes == 10  # 4 + 6
        assert st.oldest_entry != ""

    def test_status_on_empty_cache_returns_zeros(self, cache_mgr: CacheManager) -> None:
        """status() on an empty cache returns zero counts."""
        st = cache_mgr.status()
        assert st.entry_count == 0
        assert st.total_size_bytes == 0
        assert st.oldest_entry == ""

    def test_clear_removes_all_entries_and_files(self, cache_mgr: CacheManager) -> None:
        """clear() removes all entries and files."""
        cache_mgr.store("key1", "cdse", "S2", "h1", b"data1")
        cache_mgr.store("key2", "cds", "ERA5", "h2", b"data2")

        file1 = cache_mgr._key_to_path("key1", "cdse")
        file2 = cache_mgr._key_to_path("key2", "cds")
        assert file1.exists()
        assert file2.exists()

        cache_mgr.clear()

        # Files should be gone
        assert not file1.exists()
        assert not file2.exists()

        # SQLite should be empty
        conn = sqlite3.connect(str(cache_mgr._db_path))
        count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        conn.close()
        assert count == 0

    def test_after_clear_status_returns_zeros(self, cache_mgr: CacheManager) -> None:
        """After clear(), status() returns zeros."""
        cache_mgr.store("key1", "cdse", "S2", "h1", b"data")
        cache_mgr.clear()

        st = cache_mgr.status()
        assert st.entry_count == 0
        assert st.total_size_bytes == 0
        assert st.oldest_entry == ""


# ---------------------------------------------------------------------------
# Edge case: CacheEntry and CacheStatus dataclass construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataclasses:
    """Tests for CacheEntry and CacheStatus dataclass construction."""

    def test_cache_entry_fields(self) -> None:
        """CacheEntry stores all expected fields."""
        entry = CacheEntry(
            cache_key="key",
            provider="cdse",
            product="S2",
            location_hash="abc",
            created_at="2024-01-01T00:00:00+00:00",
            last_accessed_at="2024-01-01T00:00:00+00:00",
            expires_at="2024-01-02T00:00:00+00:00",
            file_path="/cache/cdse/hash.dat",
            size_bytes=1024,
            params_json="{}",
        )
        assert entry.cache_key == "key"
        assert entry.size_bytes == 1024

    def test_cache_status_fields(self) -> None:
        """CacheStatus stores count, size, and oldest entry."""
        status = CacheStatus(
            entry_count=5,
            total_size_bytes=10240,
            oldest_entry="2024-01-01T00:00:00+00:00",
        )
        assert status.entry_count == 5
        assert status.total_size_bytes == 10240
        assert status.oldest_entry == "2024-01-01T00:00:00+00:00"
