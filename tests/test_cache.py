"""Tests for generic in-memory caches."""

import unittest

from cache import Cache, CacheFactory, LRUCache


class CacheTest(unittest.TestCase):
    def test_cache_inserts_multiple_pairs_and_removes_data(self) -> None:
        cache = Cache[str, int](3)

        cache.insertPairs((("first", 1), ("second", 2), ("third", 3)))
        cache.remove("second")

        self.assertEqual(cache.memory, {"first": 1, "third": 3})
        self.assertEqual(cache.memory_size, 3)

        cache.purge()

        self.assertEqual(cache.memory, {})

    def test_cache_rejects_a_non_positive_memory_size(self) -> None:
        with self.assertRaises(ValueError):
            Cache(0)

    def test_lru_cache_evicts_the_least_recently_used_pair(self) -> None:
        cache = LRUCache[str, int](2)
        cache.insert("first", 1)
        cache.insert("second", 2)

        self.assertEqual(cache.get("first"), 1)
        self.assertIn("first", cache.memory)
        cache.insert("third", 3)

        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.memory, {"first": 1, "third": 3})

    def test_lru_cache_treats_an_updated_pair_as_most_recent(self) -> None:
        cache = LRUCache[str, int](2)
        cache.insert("first", 1)
        cache.insert("second", 2)

        cache.insert("first", 10)
        cache.insert("third", 3)

        self.assertEqual(cache.get("first"), 10)
        self.assertIsNone(cache.get("second"))

    def test_cache_factory_creates_an_lru_cache(self) -> None:
        cache = CacheFactory.createCache("lru", 4)

        self.assertIsInstance(cache, LRUCache)
        self.assertEqual(cache.memory_size, 4)

    def test_cache_factory_rejects_an_unknown_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown cache type"):
            CacheFactory.createCache("unknown")


if __name__ == "__main__":
    unittest.main()
