"""Local caching for API responses to avoid rate limiting."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class DataCache:
    """File-based cache for DataFrames and other data."""

    def __init__(
        self,
        cache_dir: Union[Path, str] = DEFAULT_CACHE_DIR,
        default_ttl_days: int = 7,
    ):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached files.
            default_ttl_days: Default time-to-live for cached data in days.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_days = default_ttl_days
        self._metadata_file = self.cache_dir / "_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _make_key(self, name: str, **params: Any) -> str:
        """Generate a unique cache key from name and parameters."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        hash_input = f"{name}:{param_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _get_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.parquet"

    def is_valid(self, key: str, ttl_days: Optional[int] = None) -> bool:
        """Check if a cached entry is still valid.

        Args:
            key: Cache key to check.
            ttl_days: Time-to-live in days. Uses default if not specified.

        Returns:
            True if the cache entry exists and is not expired.
        """
        if key not in self._metadata:
            return False

        ttl = ttl_days if ttl_days is not None else self.default_ttl_days
        cached_at = datetime.fromisoformat(self._metadata[key]["cached_at"])
        expires_at = cached_at + timedelta(days=ttl)
        return datetime.now() < expires_at

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame from cache.

        Args:
            key: Cache key to retrieve.

        Returns:
            Cached DataFrame or None if not found.
        """
        path = self._get_path(key)
        if path.exists():
            logger.debug(f"Cache hit: {key}")
            return pd.read_parquet(path)
        return None

    def set(self, key: str, data: pd.DataFrame, **extra_metadata: Any) -> None:
        """Store a DataFrame in cache.

        Args:
            key: Cache key to store under.
            data: DataFrame to cache.
            **extra_metadata: Additional metadata to store.
        """
        path = self._get_path(key)
        data.to_parquet(path)

        self._metadata[key] = {
            "cached_at": datetime.now().isoformat(),
            "rows": len(data),
            "columns": list(data.columns),
            **extra_metadata,
        }
        self._save_metadata()
        logger.debug(f"Cached {len(data)} rows under key: {key}")

    def get_or_fetch(
        self,
        name: str,
        fetch_fn: Callable[[], pd.DataFrame],
        ttl_days: Optional[int] = None,
        **params: Any,
    ) -> pd.DataFrame:
        """Get data from cache or fetch if not available/expired.

        Args:
            name: Descriptive name for this data.
            fetch_fn: Function to call to fetch fresh data.
            ttl_days: Time-to-live in days.
            **params: Parameters that affect the data (used in cache key).

        Returns:
            DataFrame from cache or freshly fetched.
        """
        key = self._make_key(name, **params)

        if self.is_valid(key, ttl_days):
            cached = self.get(key)
            if cached is not None:
                logger.info(f"Using cached data for {name}")
                return cached

        logger.info(f"Fetching fresh data for {name}")
        data = fetch_fn()
        self.set(key, data, name=name, params=params)
        return data

    def clear(self, older_than_days: Optional[int] = None) -> int:
        """Clear cached data.

        Args:
            older_than_days: Only clear entries older than this. If None, clear all.

        Returns:
            Number of entries cleared.
        """
        cleared = 0
        keys_to_remove = []

        for key, meta in self._metadata.items():
            if older_than_days is not None:
                cached_at = datetime.fromisoformat(meta["cached_at"])
                if datetime.now() - cached_at < timedelta(days=older_than_days):
                    continue

            path = self._get_path(key)
            if path.exists():
                path.unlink()
            keys_to_remove.append(key)
            cleared += 1

        for key in keys_to_remove:
            del self._metadata[key]

        self._save_metadata()
        logger.info(f"Cleared {cleared} cache entries")
        return cleared

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total_size = sum(
            self._get_path(key).stat().st_size
            for key in self._metadata
            if self._get_path(key).exists()
        )

        return {
            "entries": len(self._metadata),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
