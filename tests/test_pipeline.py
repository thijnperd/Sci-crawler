"""End-to-end pipeline behaviour with fake sources (no network)."""

import unittest

from papersplease.config import Settings
from papersplease.models import Paper
from papersplease.pipeline import find, merge_papers, screen_only
from papersplease.sources import SourceResult

SETTINGS = Settings(
    cache=False,
    mirrors=["https://sci-hub.se", "https://sci-hub.st"],
    min_scope=0.7,
    mailto="",
    unpaywall_email="",
    sources=["a", "b"],
)

ON_TARGET_ABSTRACT = "Muscle strength was measured in older adults; strength predicted function in adults."


def fake_searchers():
    def searcher_a(spec, settings, limit, mode="and"):
        return SourceResult(
            "a",
            [
                Paper(
                    title="Grip strength in older adults",
                    abstract=ON_TARGET_ABSTRACT,
                    year=2021,
                    doi="10.1/a",
                    cited_by=10,
                    authors=["Smith J"],
                    sources=["a"],
                ),
                Paper(
                    title="Muscle fibre type composition",
                    abstract="Muscle fibres differ in contractile proteins.",
                    year=2019,
                    doi="10.1/b",
                    sources=["a"],
                ),
                Paper(
                    title="Strength training in young athletes",
                    abstract="Strength and muscle size increased.",
                    year=2020,
                    doi="10.1/c",
                    sources=["a"],
                ),
            ],
        )

    def searcher_b(spec, settings, limit, mode="and"):
        return SourceResult(
            "b",
            [
                Paper(
                    title="Grip strength in older adults",
                    abstract="A longer abstract: " + ON_TARGET_ABSTRACT,
                    year=2021,
                    doi="10.1/A",
                    cited_by=31,
                    oa_url="https://example.org/a.pdf",
                    sources=["b"],
                )
            ],
        )

    return {"a": searcher_a, "b": searcher_b}


class MergeTests(unittest.TestCase):
    def test_same_doi_merges_into_one_record(self):
        merged = merge_papers(
            [
                Paper(title="Grip strength in older adults", doi="10.1/a", cited_by=3, sources=["a"]),
                Paper(
                    title="Grip strength in older adults",
                    doi="10.1/A",
                    abstract="long text",
                    cited_by=30,
                    sources=["b"],
                    oa_url="https://example.org/x.pdf",
                ),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].cited_by, 30)
        self.assertEqual(merged[0].abstract, "long text")
        self.assertEqual(sorted(merged[0].sources), ["a", "b"])
        self.assertEqual(merged[0].oa_url, "https://example.org/x.pdf")

    def test_title_match_merges_when_one_lacks_a_doi(self):
        merged = merge_papers(
            [
                Paper(title="Some interesting paper title", doi=None),
                Paper(title="Some interesting paper title.", doi="10.1/z"),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].doi, "10.1/z")

    def test_different_dois_are_kept_apart(self):
        merged = merge_papers(
            [Paper(title="Same title", doi="10.1/a"), Paper(title="Same title", doi="10.1/b")]
        )
        self.assertEqual(len(merged), 2)


class FindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = find(
            "muscle strength in older adults",
            settings=SETTINGS,
            limit=5,
            source_names=["a", "b"],
            searchers=fake_searchers(),
            probe_mirrors=False,
            include_oa=False,
        )

    def test_duplicates_are_collapsed(self):
        self.assertEqual(self.report.candidates, 3)
        self.assertEqual(self.report.source_counts, {"a": 3, "b": 1})

    def test_only_the_specific_paper_is_in_scope(self):
        titles = [finding.paper.title for finding in self.report.in_scope]
        self.assertEqual(titles, ["Grip strength in older adults"])

    def test_broad_paper_is_rejected_with_a_reason(self):
        rejected = {finding.paper.title: finding.verdict for finding in self.report.out_of_scope}
        self.assertIn("Muscle fibre type composition", rejected)
        self.assertTrue(rejected["Muscle fibre type composition"].reason)

    def test_wrong_population_is_adjacent(self):
        buckets = {finding.paper.title: finding.verdict.bucket for finding in self.report.findings}
        self.assertEqual(buckets["Strength training in young athletes"], "adjacent")

    def test_in_scope_paper_has_links(self):
        finding = self.report.in_scope[0]
        self.assertIsNotNone(finding.access)
        self.assertEqual(finding.access.scihub[0]["url"], "https://sci-hub.se/10.1/a")

    def test_rejected_papers_are_not_resolved(self):
        for finding in self.report.out_of_scope:
            self.assertIsNone(finding.access)

    def test_ranking_puts_in_scope_first(self):
        self.assertEqual(self.report.findings[0].verdict.bucket, "in_scope")

    def test_report_dict_is_serialisable(self):
        import json

        payload = self.report.to_dict()
        text = json.dumps(payload)
        self.assertIn("in_scope", text)
        self.assertNotIn("Muscle fibre type composition", text)  # rejected hidden by default
        payload_all = self.report.to_dict(include_rejected=True)
        self.assertEqual(len(payload_all["rejected"]), 1)

    def test_summary_counts(self):
        self.assertIn("1 in scope", self.report.summary())


class PresentationCapTests(unittest.TestCase):
    def test_only_limit_papers_are_listed_but_all_are_counted(self):
        def searcher(spec, settings, limit, mode="and"):
            return SourceResult(
                "a",
                [
                    Paper(title=f"Muscle strength in older adults study {i}",
                          abstract=ON_TARGET_ABSTRACT, year=2020 + i, doi=f"10.1/s{i}", sources=["a"])
                    for i in range(4)
                ],
            )

        report = find(
            "muscle strength in older adults",
            settings=SETTINGS,
            limit=2,
            source_names=["a"],
            searchers={"a": searcher},
            probe_mirrors=False,
            include_oa=False,
        )
        self.assertEqual(report.in_scope_total, 4)
        self.assertEqual(len(report.in_scope), 2)
        self.assertIn("4 in scope (showing 2)", report.summary())
        # Everything listed got links resolved, not just the first ten.
        self.assertTrue(all(f.access is not None for f in report.in_scope))


class ScreenOnlyTests(unittest.TestCase):
    def test_screens_external_refs(self):
        papers = [
            Paper(title="Muscle strength in older adults", abstract=ON_TARGET_ABSTRACT, doi="10.1/x"),
            Paper(title="Muscle anatomy atlas", abstract="Muscle groups."),
        ]
        findings = screen_only("muscle strength in older adults", papers, settings=SETTINGS)
        buckets = {f.paper.title: f.verdict.bucket for f in findings}
        self.assertEqual(buckets["Muscle strength in older adults"], "in_scope")
        self.assertEqual(buckets["Muscle anatomy atlas"], "out_of_scope")
        self.assertIsNone(findings[0].access)


class PipelineRegressionTests(unittest.TestCase):
    def test_merge_keeps_all_distinct_dois_with_identical_title(self):
        p1 = Paper(title="Duplicate Study Title", doi="10.1/a")
        p2 = Paper(title="Duplicate Study Title", doi="10.1/b")
        p3 = Paper(title="Duplicate Study Title", doi="10.1/c")
        merged = merge_papers([p1, p2, p3])
        self.assertEqual({p.doi for p in merged}, {"10.1/a", "10.1/b", "10.1/c"})

    def test_paper_merge_prefers_published_year_over_preprint(self):
        pre = Paper(title="Shared Study", arxiv_id="2301.00001", year=2023, type="preprint")
        pub = Paper(
            title="Shared Study",
            doi="10.1/pub",
            journal="Nature",
            year=2024,
            type="journal-article",
        )
        first = Paper(title=pre.title, arxiv_id=pre.arxiv_id, year=pre.year, type=pre.type)
        first.merge(pub)
        self.assertEqual(first.year, 2024)
        self.assertEqual(first.journal, "Nature")
        self.assertEqual(first.type, "journal-article")

        second = Paper(
            title=pub.title, doi=pub.doi, journal=pub.journal, year=pub.year, type=pub.type
        )
        second.merge(pre)
        self.assertEqual(second.year, 2024)
        self.assertEqual(second.arxiv_id, "2301.00001")

    def test_access_cache_respects_oa_and_probe_flags(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from papersplease import resolver, store
        from papersplease.pipeline import _access

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Settings(cache=True, cache_dir=Path(tmp))
            caches = store.Caches(cfg, enabled=True)
            paper = Paper(title="OA Study", doi="10.1/oa")
            full = resolver.Access(
                doi="10.1/oa",
                doi_url="https://doi.org/10.1/oa",
                open_access=[{"url": "https://oa.example/a.pdf", "kind": "oa"}],
            )
            no_oa = resolver.Access(doi="10.1/oa", doi_url="https://doi.org/10.1/oa", open_access=[])
            with patch.object(resolver, "access_for", side_effect=[full, no_oa]) as mock_af:
                got1 = _access(paper, cfg, caches, probe=True, include_oa=True)
                got2 = _access(paper, cfg, caches, probe=False, include_oa=False)
                self.assertEqual(mock_af.call_count, 2)
                self.assertEqual(len(got1.open_access), 1)
                self.assertEqual(got2.open_access, [])

    def test_read_one_surfaces_error_in_read_note(self):
        from unittest.mock import patch

        from papersplease import reading, sources
        from papersplease.pipeline import read_one

        paper = Paper(title="Deep Read Study", doi="10.1/deep")
        with (
            patch.object(sources, "resolve_identifier", return_value=paper),
            patch.object(reading, "read", side_effect=RuntimeError("boom")),
        ):
            out = read_one("10.1/deep", settings=SETTINGS, deep=True)
        self.assertIsNotNone(out)
        self.assertIn("read failed", out.read_note or "")


if __name__ == "__main__":
    unittest.main()
