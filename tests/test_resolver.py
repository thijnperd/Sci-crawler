"""Link building and mirror health, with the network stubbed out."""

import unittest
from unittest.mock import patch

from papersplease import resolver
from papersplease.config import Settings
from papersplease.models import Paper
from papersplease.net import Blocked, FetchError


SETTINGS = Settings(
    cache=False,
    mirrors=["https://sci-hub.se", "https://sci-hub.st"],
    mailto="",
    unpaywall_email="",
)


class LinkTests(unittest.TestCase):
    def test_links_are_built_for_every_mirror(self):
        links = resolver.mirror_links("10.1038/NATURE12373", SETTINGS.mirrors)
        self.assertEqual(len(links), 2)
        self.assertEqual(links[0]["url"], "https://sci-hub.se/10.1038/nature12373")

    def test_doi_url_prefix_is_accepted(self):
        links = resolver.mirror_links("https://doi.org/10.1000/abc", ["https://sci-hub.ru/"])
        self.assertEqual(links[0]["url"], "https://sci-hub.ru/10.1000/abc")

    def test_no_doi_no_link(self):
        self.assertEqual(resolver.mirror_links("", SETTINGS.mirrors), [])
        self.assertEqual(resolver.mirror_links("garbage", SETTINGS.mirrors), [])

    def test_doi_url(self):
        self.assertEqual(resolver.doi_url("10.1000/abc"), "https://doi.org/10.1000/abc")


class HealthTests(unittest.TestCase):
    def setUp(self):
        resolver._health_cache.clear()

    def tearDown(self):
        resolver._health_cache.clear()

    def test_states_are_classified(self):
        def fake_fetch(url, settings, **kwargs):
            if "se" in url:
                return b"<html>ok</html>"
            if "st" in url:
                raise Blocked(url, 403, "blocked")
            raise FetchError(url, None, "connection refused")

        settings = Settings(cache=False, mirrors=["https://sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru"])
        with patch.object(resolver, "fetch", side_effect=fake_fetch):
            statuses = resolver.mirror_health(settings)
        by_mirror = {s.mirror: s.state for s in statuses}
        self.assertEqual(by_mirror["https://sci-hub.se"], "ok")
        self.assertEqual(by_mirror["https://sci-hub.st"], "blocked")
        self.assertEqual(by_mirror["https://sci-hub.ru"], "unreachable")

    def test_health_is_cached(self):
        calls = []

        def fake_fetch(url, settings, **kwargs):
            calls.append(url)
            return b"ok"

        settings = Settings(cache=False, mirrors=["https://m.example"])
        with patch.object(resolver, "fetch", side_effect=fake_fetch):
            resolver.mirror_health(settings)
            resolver.mirror_health(settings)
        self.assertEqual(len(calls), 1)

    def test_force_refreshes(self):
        calls = []

        def fake_fetch(url, settings, **kwargs):
            calls.append(url)
            return b"ok"

        settings = Settings(cache=False, mirrors=["https://m.example"])
        with patch.object(resolver, "fetch", side_effect=fake_fetch):
            resolver.mirror_health(settings)
            resolver.mirror_health(settings, force=True)
        self.assertEqual(len(calls), 2)


class AccessTests(unittest.TestCase):
    def setUp(self):
        resolver._health_cache.clear()

    def _access(self, paper, **kwargs):
        return resolver.access_for(paper, SETTINGS, probe=False, **kwargs)

    def test_scihub_links_are_listed_without_probing(self):
        access = self._access(Paper(title="T", doi="10.1000/abc"))
        self.assertEqual(len(access.scihub), 2)
        self.assertEqual(access.scihub[0]["state"], "unchecked")
        self.assertEqual(access.best["url"], "https://sci-hub.se/10.1000/abc")
        self.assertEqual(access.doi_url, "https://doi.org/10.1000/abc")

    def test_open_access_wins_over_scihub(self):
        paper = Paper(title="T", doi="10.1000/abc", oa_url="https://example.org/paper.pdf")
        access = self._access(paper, include_oa=True)
        self.assertTrue(access.open_access)
        self.assertEqual(access.best["url"], "https://example.org/paper.pdf")

    def test_open_access_can_be_skipped(self):
        paper = Paper(title="T", doi="10.1000/abc", oa_url="https://example.org/paper.pdf")
        access = self._access(paper, include_oa=False)
        self.assertEqual(access.open_access, [])
        self.assertTrue(any("skipped" in note for note in access.notes))

    def test_pmc_pdf_is_preferred_over_the_landing_page(self):
        paper = Paper(title="T", doi="10.1000/abc", pmcid="PMC1")
        access = self._access(paper, include_oa=True)
        self.assertEqual(access.best["kind"], "pmc-pdf")
        self.assertIn("pdf=render", access.best["url"])
        kinds = [item["kind"] for item in access.open_access]
        self.assertLess(kinds.index("pmc-pdf"), kinds.index("pmc"))

    def test_pdf_beats_landing_page(self):
        paper = Paper(title="T", oa_url="https://example.org/landing")
        access = self._access(paper, include_oa=True)
        access.open_access.append({"url": "https://example.org/paper.pdf", "kind": "unpaywall-pdf", "license": None})
        from papersplease.resolver import _download_rank

        ordered = sorted(access.open_access, key=_download_rank)
        self.assertTrue(ordered[0]["url"].endswith(".pdf"))

    def test_pmc_and_arxiv_links_are_collected(self):
        paper = Paper(title="T", doi="10.1000/abc", pmcid="PMC1", arxiv_id="2401.00001")
        access = self._access(paper)
        kinds = {item["kind"] for item in access.open_access}
        self.assertIn("pmc", kinds)
        self.assertIn("arxiv-pdf", kinds)

    def test_no_doi_still_reports_open_access(self):
        access = self._access(Paper(title="T", oa_url="https://example.org/x.pdf"))
        self.assertIsNone(access.doi)
        self.assertEqual(access.scihub, [])
        self.assertEqual(access.best["url"], "https://example.org/x.pdf")


class DownloadAnyTests(unittest.TestCase):
    def test_falls_through_until_a_real_pdf_appears(self):
        access = resolver.Access(
            open_access=[
                {"url": "https://a.example/landing", "kind": "oa"},
                {"url": "https://b.example/paper.pdf", "kind": "unpaywall-pdf"},
            ]
        )
        calls = []

        def fake_download(url, destination, settings):
            calls.append(url)
            return (True, "ok") if url.endswith(".pdf") else (False, "not a PDF")

        with patch.object(resolver, "download", side_effect=fake_download):
            ok, note = resolver.download_any(access, "/tmp/x.pdf", SETTINGS)
        self.assertTrue(ok, note)
        self.assertEqual(calls, ["https://a.example/landing", "https://b.example/paper.pdf"])

    def test_blocked_mirrors_are_not_attempted(self):
        access = resolver.Access(
            scihub=[{"mirror": "m", "url": "https://m.example/10.1/x", "state": "blocked"}]
        )
        with patch.object(resolver, "download", side_effect=AssertionError("must not be called")):
            ok, note = resolver.download_any(access, "/tmp/x.pdf", SETTINGS)
        self.assertFalse(ok)
        self.assertIn("no downloadable source", note)


class DownloadTests(unittest.TestCase):
    def test_non_pdf_is_rejected(self):
        with patch.object(resolver, "fetch", return_value=b"<html>captcha</html>"):
            ok, note = resolver.download("https://m.example/x", "/tmp/should-not-exist.pdf", SETTINGS)
        self.assertFalse(ok)
        self.assertIn("not a PDF", note)

    def test_pdf_is_written(self):
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "x.pdf")
            with patch.object(resolver, "fetch", return_value=b"%PDF-1.4 test"):
                ok, note = resolver.download("https://m.example/x", target, SETTINGS)
            self.assertTrue(ok, note)
            with open(target, "rb") as handle:
                self.assertTrue(handle.read().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
