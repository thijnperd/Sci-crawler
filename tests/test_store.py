"""Verdict and access caches: content-keyed, expiring, never fatal."""

import tempfile
import unittest
from pathlib import Path

from papersplease.config import Settings
from papersplease.intent import parse_intent
from papersplease.models import Paper, Verdict
from papersplease.store import Caches, JsonCache, fingerprint


def settings(tmp: str, *, cache: bool = True) -> Settings:
    return Settings(cache=cache, cache_dir=Path(tmp), mirrors=["https://sci-hub.se"])


def verdict(**kwargs) -> Verdict:
    return Verdict(
        doi=kwargs.get("doi", "10.1/a"),
        title=kwargs.get("title", "A paper"),
        score=0.8,
        coverage=0.8,
        title_coverage=0.5,
        matched=["muscle"],
        bucket="in_scope",
        in_scope=True,
        reason="matched muscle",
    )


class JsonCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "cache.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_persists_to_disk(self):
        cache = JsonCache(self.path, ttl=3600)
        cache.put("k", {"v": 1})
        cache.flush()
        again = JsonCache(self.path, ttl=3600)
        self.assertEqual(again.get("k"), {"v": 1})
        self.assertEqual(again.hits, 1)

    def test_expired_entries_are_misses(self):
        cache = JsonCache(self.path, ttl=-1)
        cache.put("k", {"v": 1})
        self.assertIsNone(cache.get("k"))
        self.assertEqual(cache.misses, 1)

    def test_disabled_cache_does_nothing(self):
        cache = JsonCache(self.path, ttl=3600, enabled=False)
        cache.put("k", 1)
        self.assertIsNone(cache.get("k"))
        self.assertFalse(self.path.exists())

    def test_corrupt_file_is_ignored(self):
        self.path.write_text("not json", "utf-8")
        cache = JsonCache(self.path, ttl=3600)
        self.assertIsNone(cache.get("k"))
        cache.put("k", 2)
        self.assertEqual(cache.get("k"), 2)


class CachesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = settings(self.tmp.name)
        self.spec = parse_intent("muscle strength in older adults")
        self.paper = Paper(title="Grip strength in older adults", doi="10.1/a")

    def tearDown(self):
        self.tmp.cleanup()

    def test_verdict_survives_a_new_process(self):
        first = Caches(self.settings)
        first.put_verdict(self.spec, self.paper.key(), verdict())
        first.flush()

        second = Caches(self.settings)
        stored = second.get_verdict(self.spec, self.paper.key())
        self.assertIsNotNone(stored)
        self.assertEqual(stored.bucket, "in_scope")
        self.assertEqual(stored.matched, ["muscle"])
        self.assertEqual(second.stats()["verdict_hits"], 1)

    def test_a_different_scope_is_a_different_key(self):
        cache = Caches(self.settings)
        cache.put_verdict(self.spec, self.paper.key(), verdict())
        other = parse_intent("muscle strength in older adults", min_scope=0.4)
        self.assertIsNone(cache.get_verdict(other, self.paper.key()))

    def test_access_round_trip(self):
        cache = Caches(self.settings)
        cache.put_access("10.1/a", {"doi": "10.1/a", "open_access": [], "scihub": [], "best": None, "notes": []})
        self.assertEqual(cache.get_access("10.1/a")["doi"], "10.1/a")
        self.assertIsNone(cache.get_access("10.1/b"))

    def test_fingerprint_ignores_term_order_but_not_threshold(self):
        a = parse_intent("muscle strength", min_scope=0.7)
        b = parse_intent("muscle strength", min_scope=0.7)
        self.assertEqual(fingerprint(a), fingerprint(b))
        c = parse_intent("muscle strength", min_scope=0.5)
        self.assertNotEqual(fingerprint(a), fingerprint(c))

    def test_disabled_cache_does_not_write(self):
        cache = Caches(settings(self.tmp.name, cache=False))
        cache.put_verdict(self.spec, self.paper.key(), verdict())
        cache.flush()
        self.assertFalse((Path(self.tmp.name) / "verdicts").exists())


if __name__ == "__main__":
    unittest.main()
