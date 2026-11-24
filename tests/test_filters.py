"""Post-search filters behave the same for every source."""

import unittest

from papersplease.filters import apply, keep, normalize_type, paper_type
from papersplease.models import Paper


def paper(**kwargs) -> Paper:
    return Paper(
        title=kwargs.pop("title", "Effects of resistance training volume"),
        authors=kwargs.pop("authors", ["Schoenfeld Brad", "Grgic Jozo"]),
        journal=kwargs.pop("journal", "Journal of Strength Research"),
        year=kwargs.pop("year", 2021),
        cited_by=kwargs.pop("cited_by", 40),
        type=kwargs.pop("type", "journal-article"),
        **kwargs,
    )


class TypeTests(unittest.TestCase):
    def test_provider_types_are_normalised(self):
        self.assertEqual(normalize_type("journal-article"), "journal")
        self.assertEqual(normalize_type("proceedings-article"), "conference")
        self.assertEqual(normalize_type("posted-content"), "preprint")
        self.assertEqual(normalize_type("book-chapter"), "book")
        self.assertEqual(normalize_type("Review"), "other")

    def test_arxiv_id_without_a_type_is_a_preprint(self):
        self.assertEqual(paper_type(paper(type=None, arxiv_id="2401.12345")), "preprint")

    def test_a_journal_record_stays_a_journal_even_with_an_arxiv_id(self):
        self.assertEqual(paper_type(paper(type="journal-article", arxiv_id="2401.12345")), "journal")

    def test_venue_is_the_fallback(self):
        self.assertEqual(paper_type(paper(type=None)), "journal")


class FilterTests(unittest.TestCase):
    def test_author_is_a_case_insensitive_substring(self):
        self.assertTrue(keep(paper(), authors=["schoenfeld"]))
        self.assertFalse(keep(paper(), authors=["Hinton"]))

    def test_venue_is_a_case_insensitive_substring(self):
        self.assertTrue(keep(paper(), venues=["strength"]))
        self.assertFalse(keep(paper(), venues=["Nature"]))

    def test_min_citations_drops_unknown_counts(self):
        self.assertTrue(keep(paper(cited_by=40), min_citations=10))
        self.assertFalse(keep(paper(cited_by=0), min_citations=1))

    def test_peer_reviewed_and_preprints_are_shortcuts(self):
        journal = paper(type="journal-article")
        preprint = paper(type="posted-content")
        self.assertTrue(keep(journal, peer_reviewed=True))
        self.assertFalse(keep(preprint, peer_reviewed=True))
        self.assertTrue(keep(preprint, preprints_only=True))
        self.assertFalse(keep(journal, preprints_only=True))

    def test_explicit_types_accumulate(self):
        journal = paper(type="journal-article")
        preprint = paper(type="posted-content")
        both = {"types": ["journal"], "preprints_only": True}
        self.assertTrue(keep(journal, **both))
        self.assertTrue(keep(preprint, **both))
        self.assertFalse(keep(paper(type="book-chapter"), **both))

    def test_no_filters_keeps_everything(self):
        papers = [paper(), paper(type="posted-content")]
        self.assertEqual(apply(papers), papers)

    def test_apply_preserves_order(self):
        first = paper(title="First")
        second = paper(title="Second")
        filtered = apply([first, second], authors=["Schoenfeld"])
        self.assertEqual([p.title for p in filtered], ["First", "Second"])

    def test_word_boundary_matches_prevent_false_prefix_hits(self):
        self.assertFalse(keep(paper(authors=["Lindqvist A"]), authors=["Li"]))
        self.assertTrue(keep(paper(authors=["Wei Li"]), authors=["Li"]))
        self.assertTrue(keep(paper(authors=["Li J"]), authors=["Li"]))
        self.assertFalse(keep(paper(journal="Signatures"), venues=["Nature"]))
        self.assertTrue(keep(paper(journal="Nature Medicine"), venues=["Nature"]))


if __name__ == "__main__":
    unittest.main()
