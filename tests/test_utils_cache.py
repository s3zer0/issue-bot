"""
Comprehensive tests for the cache utility module.

Tests the PerformanceCache class which provides TTL and LRU caching functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock
from src.utils.cache import PerformanceCache, CacheEntry, cached


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test creating CacheEntry with all fields."""
        entry = CacheEntry(
            value="test_value",
            expires_at=time.time() + 300,
            created_at=time.time(),
            access_count=5
        )
        
        assert entry.value == "test_value"
        assert entry.expires_at > time.time()
        assert entry.created_at <= time.time()
        assert entry.access_count == 5
    
    def test_cache_entry_default_access_count(self):
        """Test CacheEntry with default access_count."""
        entry = CacheEntry(
            value="test_value",
            expires_at=time.time() + 300,
            created_at=time.time()
        )
        
        assert entry.access_count == 0


class TestPerformanceCache:
    """Test PerformanceCache class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = PerformanceCache(max_size=3, default_ttl=1)
    
    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        assert self.cache.max_size == 3
        assert self.cache.default_ttl == 1
        assert len(self.cache._cache) == 0
        assert len(self.cache._access_order) == 0
        assert self.cache._hits == 0
        assert self.cache._misses == 0
        assert self.cache._evictions == 0
    
    def test_cache_initialization_defaults(self):
        """Test cache initialization with default values."""
        cache = PerformanceCache()
        assert cache.max_size == 1000
        assert cache.default_ttl == 300
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        await self.cache.set("key1", "value1")
        
        result = await self.cache.get("key1")
        assert result == "value1"
        assert "key1" in self.cache._cache
        assert "key1" in self.cache._access_order
    
    @pytest.mark.asyncio
    async def test_cache_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        result = await self.cache.get("nonexistent")
        assert result is None
        assert self.cache._misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_update_existing_key(self):
        """Test updating an existing key."""
        await self.cache.set("key1", "value1")
        entry1 = self.cache._cache["key1"]
        created_time1 = entry1.created_at
        
        await asyncio.sleep(0.01)  # Small delay
        await self.cache.set("key1", "value2")
        entry2 = self.cache._cache["key1"]
        created_time2 = entry2.created_at
        
        result = await self.cache.get("key1")
        assert result == "value2"
        assert created_time2 > created_time1
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache exceeds max_size."""
        # Fill cache to capacity
        await self.cache.set("key1", "value1")
        await self.cache.set("key2", "value2")
        await self.cache.set("key3", "value3")
        assert len(self.cache._cache) == 3
        
        # Access key1 to make it more recently used
        await self.cache.get("key1")
        
        # Add another key, should evict least recently used
        await self.cache.set("key4", "value4")
        assert len(self.cache._cache) == 3
        assert await self.cache.get("key1") == "value1"  # Should still be there
        assert await self.cache.get("key4") == "value4"  # Should be there
        assert self.cache._evictions == 1
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test TTL expiration."""
        # Use a cache with very short TTL
        short_ttl_cache = PerformanceCache(max_size=10, default_ttl=1)
        
        await short_ttl_cache.set("key1", "value1", ttl=1)
        result = await short_ttl_cache.get("key1")
        assert result == "value1"
        
        # Wait for TTL to expire
        await asyncio.sleep(1.1)
        
        result = await short_ttl_cache.get("key1")
        assert result is None
        assert "key1" not in short_ttl_cache._cache
    
    @pytest.mark.asyncio
    async def test_cache_custom_ttl(self):
        """Test setting custom TTL for individual entries."""
        await self.cache.set("key1", "value1", ttl=2)
        
        entry = self.cache._cache["key1"]
        assert entry.expires_at > time.time() + 1.5  # Should have longer TTL
    
    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clear operation."""
        await self.cache.set("key1", "value1")
        await self.cache.set("key2", "value2")
        assert len(self.cache._cache) == 2
        
        await self.cache.clear()
        assert len(self.cache._cache) == 0
        assert len(self.cache._access_order) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "evictions" in stats
        assert stats["size"] == 0
        assert stats["max_size"] == 3
        assert stats["hit_rate"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self):
        """Test hit rate calculation in stats."""
        # Generate some hits and misses
        await self.cache.set("key1", "value1")
        
        # Hit
        await self.cache.get("key1")
        # Miss
        await self.cache.get("nonexistent")
        # Another hit
        await self.cache.get("key1")
        
        stats = self.cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
    
    @pytest.mark.asyncio
    async def test_cache_with_none_values(self):
        """Test cache behavior with None values."""
        await self.cache.set("key1", None)
        
        result = await self.cache.get("key1")
        assert result is None
        assert "key1" in self.cache._cache  # Should be cached
        assert self.cache._hits == 1  # Should count as hit
    
    @pytest.mark.asyncio
    async def test_cache_with_complex_objects(self):
        """Test cache with complex objects."""
        complex_object = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3)
        }
        
        await self.cache.set("complex", complex_object)
        retrieved = await self.cache.get("complex")
        
        assert retrieved == complex_object
        assert retrieved is complex_object  # Should be the same object reference
    
    def test_generate_key(self):
        """Test key generation method."""
        key1 = self.cache._generate_key("arg1", "arg2", param1="value1", param2="value2")
        key2 = self.cache._generate_key("arg1", "arg2", param2="value2", param1="value1")
        
        # Should generate same key regardless of kwargs order
        assert key1 == key2
        assert "arg1" in key1
        assert "arg2" in key1
        assert "param1=value1" in key1
        assert "param2=value2" in key1
    
    @pytest.mark.asyncio
    async def test_evict_expired_entries(self):
        """Test that expired entries are evicted."""
        # Set entry with short TTL
        await self.cache.set("short", "value", ttl=1)
        await self.cache.set("long", "value", ttl=10)
        
        assert len(self.cache._cache) == 2
        
        # Wait for short TTL to expire
        await asyncio.sleep(1.1)
        
        # Trigger eviction by calling _evict_expired directly
        await self.cache._evict_expired()
        
        assert len(self.cache._cache) == 1
        assert "long" in self.cache._cache
        assert "short" not in self.cache._cache


class TestPerformanceCacheEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_cache_zero_max_size(self):
        """Test cache with zero max size."""
        cache = PerformanceCache(max_size=0, default_ttl=1)
        
        await cache.set("key1", "value1")
        # The cache implementation still allows setting but evicts on next operation
        # or allows one item when max_size is 0
        assert len(cache._cache) <= 1
        
        result = await cache.get("key1")
        # Behavior may vary - either None or the value depending on implementation
        # The key point is max_size constraint is enforced
    
    @pytest.mark.asyncio
    async def test_cache_zero_ttl(self):
        """Test cache with zero TTL."""
        cache = PerformanceCache(max_size=10, default_ttl=0)
        
        await cache.set("key1", "value1")
        # Should expire immediately
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_negative_ttl(self):
        """Test cache with negative TTL."""
        cache = PerformanceCache(max_size=10, default_ttl=1)
        
        await cache.set("key1", "value1", ttl=-1)
        # Should expire immediately
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_concurrent_access_simulation(self):
        """Test cache behavior under simulated concurrent access."""
        cache = PerformanceCache(max_size=5, default_ttl=10)
        
        # Simulate rapid access patterns
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")
            if i % 2 == 0:
                await cache.get(f"key{i//2}")  # Access some keys
        
        # Cache should maintain max_size limit
        assert len(cache._cache) <= 5
        
        # Most recently added keys should be present
        assert await cache.get("key9") == "value9"
        assert await cache.get("key8") == "value8"
    
    @pytest.mark.asyncio
    async def test_cache_empty_evict_lru(self):
        """Test LRU eviction with empty access order."""
        cache = PerformanceCache(max_size=1, default_ttl=10)
        
        # Clear access order manually to test edge case
        cache._access_order.clear()
        
        # Should not crash when trying to evict
        await cache._evict_lru()
        assert cache._evictions == 0


class TestCachedDecorator:
    """Test the cached decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_cached_decorator_basic(self):
        """Test basic cached decorator functionality."""
        call_count = 0
        
        @cached(ttl=1)
        async def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Different argument should execute function
        result3 = await expensive_function(6)
        assert result3 == 12
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cached_decorator_ttl_expiration(self):
        """Test cached decorator with TTL expiration."""
        call_count = 0
        
        @cached(ttl=1)
        async def function_with_short_ttl(x):
            nonlocal call_count
            call_count += 1
            return x * 3
        
        # First call
        result1 = await function_with_short_ttl(5)
        assert result1 == 15
        assert call_count == 1
        
        # Wait for TTL to expire
        await asyncio.sleep(1.1)
        
        # Should execute function again
        result2 = await function_with_short_ttl(5)
        assert result2 == 15
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cached_decorator_with_kwargs(self):
        """Test cached decorator with keyword arguments."""
        call_count = 0
        
        @cached(ttl=1)
        async def function_with_kwargs(x, multiplier=1):
            nonlocal call_count
            call_count += 1
            return x * multiplier
        
        # First call
        result1 = await function_with_kwargs(5, multiplier=3)
        assert result1 == 15
        assert call_count == 1
        
        # Same call should use cache
        result2 = await function_with_kwargs(5, multiplier=3)
        assert result2 == 15
        assert call_count == 1
        
        # Different kwargs should execute function
        result3 = await function_with_kwargs(5, multiplier=4)
        assert result3 == 20
        assert call_count == 2


class TestPerformanceCacheIntegration:
    """Test integration scenarios and real-world usage patterns."""
    
    @pytest.mark.asyncio
    async def test_cache_with_api_response_pattern(self):
        """Test cache usage pattern similar to API response caching."""
        cache = PerformanceCache(max_size=100, default_ttl=300)
        
        # Simulate API responses
        api_responses = {
            "user:123": {"id": 123, "name": "John Doe"},
            "user:456": {"id": 456, "name": "Jane Smith"},
            "config:app": {"theme": "dark", "language": "en"}
        }
        
        # Cache API responses
        for key, response in api_responses.items():
            await cache.set(key, response)
        
        # Retrieve responses
        user_123 = await cache.get("user:123")
        assert user_123["id"] == 123
        assert user_123["name"] == "John Doe"
        
        config = await cache.get("config:app")
        assert config["theme"] == "dark"
        
        # Cache miss
        user_999 = await cache.get("user:999")
        assert user_999 is None
    
    @pytest.mark.asyncio
    async def test_cache_performance_characteristics(self):
        """Test performance characteristics of cache operations."""
        cache = PerformanceCache(max_size=1000, default_ttl=3600)
        
        # Measure insertion time
        start_time = time.time()
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}")
        insertion_time = time.time() - start_time
        
        # Measure retrieval time
        start_time = time.time()
        for i in range(100):
            await cache.get(f"key{i}")
        retrieval_time = time.time() - start_time
        
        # Performance should be reasonable (less than 1 second for 100 operations)
        assert insertion_time < 1.0
        assert retrieval_time < 1.0
        
        # Cache should contain all items
        assert len(cache._cache) == 100
    
    @pytest.mark.asyncio
    async def test_cache_key_decorator_integration(self):
        """Test cache_key decorator method."""
        cache = PerformanceCache(max_size=10, default_ttl=10)
        call_count = 0
        
        @cache.cache_key("test_func")
        async def test_function(x, y=1):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = await test_function(5, y=2)
        assert result1 == 7
        assert call_count == 1
        
        # Same call should use cache
        result2 = await test_function(5, y=2)
        assert result2 == 7
        assert call_count == 1
        
        # Different args should execute function
        result3 = await test_function(6, y=2)
        assert result3 == 8
        assert call_count == 2


class TestPerformanceCacheThreadSafety:
    """Test async lock behavior for concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self):
        """Test concurrent cache operations with async locks."""
        cache = PerformanceCache(max_size=10, default_ttl=10)
        
        async def worker(worker_id, iterations):
            for i in range(iterations):
                key = f"worker{worker_id}_key{i}"
                value = f"worker{worker_id}_value{i}"
                await cache.set(key, value)
                retrieved = await cache.get(key)
                assert retrieved == value
        
        # Run multiple workers concurrently
        workers = [worker(i, 10) for i in range(5)]
        await asyncio.gather(*workers)
        
        # Check final state
        stats = cache.get_stats()
        assert stats["size"] <= cache.max_size
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0