import os
import pickle
from typing import Any, Dict


class Cache:
    """
    A key-value store that persists data to disk in a .cache file using pickle.
    Supports storing and retrieving complex data structures.
    """
    
    def __init__(self, cache_file: str = ".cache"):
        """
        Initialize the cache.
        
        Args:
            cache_file: Path to the cache file (default: ".cache")
        """
        self.cache_file = cache_file
        self.cache_data: Dict[str, Any] = {}
        self._load()

    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return len(self.cache_data) == 0
    
    def _load(self) -> None:
        """Load the cache from disk if it exists."""
        if not os.path.exists(self.cache_file):
            return        
        with open(self.cache_file, 'rb') as f:
            self.cache_data = pickle.load(f)
    
    def _save(self) -> None:
        """Save the cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache_data, f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The key to look up
            default: Value to return if key is not found
            
        Returns:
            The cached value or default if not found
        """
        return self.cache_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The key to store the value under
            value: The value to store (can be any pickle-serializable object)
        """
        self.cache_data[key] = value
        self._save()

    def keys(self) -> list[str]:
        """Get all keys in the cache."""
        return list(self.cache_data.keys())
