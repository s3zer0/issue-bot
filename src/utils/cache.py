"""
Performance optimization: Simple in-memory cache with TTL support.

This module provides caching functionality to reduce redundant API calls
and improve response times for frequently requested data.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    value: Any
    expires_at: float
    created_at: float
    access_count: int = 0


class PerformanceCache:
    """
    High-performance in-memory cache with TTL, LRU eviction, and statistics.
    
    Features:
    - TTL-based expiration
    - LRU eviction when max_size is reached
    - Access statistics for monitoring
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: Dict[str, float] = {}  # For LRU tracking
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"Performance cache initialized (max_size={max_size}, default_ttl={default_ttl}s)")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(key_parts)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._access_order.pop(key, None)
                self._misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            self._access_order[key] = time.time()
            self._hits += 1
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            ttl = ttl or self.default_ttl
            now = time.time()
            
            # Evict expired entries first
            await self._evict_expired()
            
            # Check if we need to evict LRU entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            entry = CacheEntry(
                value=value,
                expires_at=now + ttl,
                created_at=now
            )
            
            self._cache[key] = entry
            self._access_order[key] = now
    
    async def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._access_order.pop(key, None)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._access_order:
            return
        
        # Find oldest access time
        lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
        
        del self._cache[lru_key]
        del self._access_order[lru_key]
        self._evictions += 1
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions
        }
    
    def cache_key(self, *args, **kwargs):
        """Decorator for automatic cache key generation."""
        def decorator(func):
            async def wrapper(*func_args, **func_kwargs):
                # Generate cache key
                cache_key = self._generate_key(func.__name__, *args, *func_args, **kwargs, **func_kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                result = await func(*func_args, **func_kwargs)
                await self.set(cache_key, result)
                logger.debug(f"Cached result for {func.__name__}")
                
                return result
            return wrapper
        return decorator


# Global cache instance
performance_cache = PerformanceCache(max_size=1000, default_ttl=300)


def cached(ttl: int = 300):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = performance_cache._generate_key(func.__name__, *args, **kwargs)
            
            # Try cache first
            cached_result = await performance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute and cache
            result = await func(*args, **kwargs)
            await performance_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator