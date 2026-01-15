"""Caching layer for embeddings data."""

import os
from pathlib import Path

import joblib

from color_perception_bench.registry import sanitize_model_name

CACHE_DIR = Path("cache")


def _get_cache_path(model_name: str) -> Path:
    """Get the cache file path for a model."""
    safe_name = sanitize_model_name(model_name)
    return CACHE_DIR / f"embeddings_{safe_name}.joblib"


def ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(exist_ok=True)


def load_embeddings(model_name: str) -> dict | None:
    """
    Load cached embeddings for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Dictionary of embeddings data, or None if not cached.
    """
    cache_path = _get_cache_path(model_name)

    if not cache_path.exists():
        return None

    try:
        return joblib.load(cache_path)
    except Exception as e:
        print(f"Warning: Failed to load cache for {model_name}: {e}")
        return None


def save_embeddings(model_name: str, data: dict) -> None:
    """
    Save embeddings to cache for a model.

    Args:
        model_name: Name of the model.
        data: Embeddings data to cache.
    """
    ensure_cache_dir()
    cache_path = _get_cache_path(model_name)

    try:
        joblib.dump(data, cache_path)
    except Exception as e:
        print(f"Warning: Failed to save cache for {model_name}: {e}")


def invalidate_cache(model_name: str) -> bool:
    """
    Invalidate (delete) cache for a model.

    Args:
        model_name: Name of the model.

    Returns:
        True if cache was deleted, False if it didn't exist.
    """
    cache_path = _get_cache_path(model_name)

    if not cache_path.exists():
        return False

    try:
        cache_path.unlink()
        return True
    except Exception as e:
        print(f"Warning: Failed to delete cache for {model_name}: {e}")
        return False


def list_cached_models() -> list[str]:
    """
    List all models with cached embeddings.

    Returns:
        List of model names (sanitized) that have cache files.
    """
    if not CACHE_DIR.exists():
        return []

    cached = []
    for path in CACHE_DIR.glob("embeddings_*.joblib"):
        # Extract model name from filename
        name = path.stem.replace("embeddings_", "")
        cached.append(name)

    return cached


def get_cache_info(model_name: str) -> dict | None:
    """
    Get information about a model's cache file.

    Args:
        model_name: Name of the model.

    Returns:
        Dictionary with cache info, or None if not cached.
    """
    cache_path = _get_cache_path(model_name)

    if not cache_path.exists():
        return None

    stat = cache_path.stat()
    return {
        "path": str(cache_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": stat.st_mtime,
    }


def clear_all_caches() -> int:
    """
    Clear all cache files.

    Returns:
        Number of cache files deleted.
    """
    if not CACHE_DIR.exists():
        return 0

    count = 0
    for path in CACHE_DIR.glob("embeddings_*.joblib"):
        try:
            path.unlink()
            count += 1
        except Exception as e:
            print(f"Warning: Failed to delete {path}: {e}")

    return count
