"""Source adapters parse real payload shapes; nothing here touches the network."""

import unittest
from unittest.mock import patch

from papersplease.config import Settings
from papersplease.intent import parse_intent
from papersplease import sources


SETTINGS = Settings(cache=False, mailto="test@example.org", unpaywall_email="test@example.org")


EPMC_RECORD = {
    "id": "PMC123456",
    "source": "PMC",
    "pmid": "12345678",
    "pmcid": "PMC123456",
    "doi": "10.1000/ABC.123",
    "title": "Muscle strength in older adults: <i>a cohort study</i>",
    "authorString": "Smith J, Doe A, Roe R.",
    "journalTitle": "Journal of Ageing",
    "pubYear": "2019",
    "abstractText": "<h4>Background</h4> Muscle strength declines in older adults.",
    "citedByCount": 42,
    "isOpenAccess": "Y",
    "keywordList": {"keyword": ["muscle strength", "ageing"]},
    "meshHeadingList": {"meshHeading": [{"descriptorName": "Muscle Strength"}, {"descriptorName": "Aged"}]},
    "pubTypeList": {"pubType": ["research-article"]},
}


class EuropePmcTests(unittest.TestCase):
    def test_record_is_normalised(self):
        paper = sources._epmc_paper(EPMC_RECORD)
        self.assertIsNotNone(paper)
        self.assertEqual(paper.doi, "10.1000/abc.123")
        self.assertEqual(paper.year, 2019)
        self.assertEqual(paper.pmid, "12345678")
        self.assertEqual(paper.pmcid, "PMC123456")
        self.assertTrue(paper.is_oa)
        self.assertNotIn("<i>", paper.title)
        self.assertEqual(paper.authors, ["Smith J", "Doe A", "Roe R"])
        self.assertIn("Muscle Strength", paper.keywords)
        self.assertIn("muscle strength", paper.keywords.lower() if isinstance(paper.keywords, str) else [k.lower() for k in paper.keywords])

    def test_search_parses_payload(self):
        payload = {"resultList": {"result": [EPMC_RECORD, {"title": ""}]}}
        with patch.object(sources, "fetch_json", return_value=payload):
            result = sources.search_europepmc(parse_intent("muscle strength"), SETTINGS, 5)
        self.assertIsNone(result.warning)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].sources, ["europepmc"])

    def test_failure_is_a_warning_not_an_exception(self):
        from papersplease.net import FetchError

        with patch.object(sources, "fetch_json", side_effect=FetchError("u", 500, "boom")):
            result = sources.search_europepmc(parse_intent("muscle strength"), SETTINGS, 5)
        self.assertEqual(result.papers, [])
        self.assertIn("Europe PMC", result.warning)


CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "DOI": "10.1000/xyz",
                "title": ["Resistance training and strength in the elderly"],
                "author": [{"family": "Smith", "given": "Jane"}, {"name": "Consortium X"}],
                "issued": {"date-parts": [[2018, 5, 1]]},
                "container-title": ["Journal of Strength"],
                "abstract": "<jats:p>Strength increased.</jats:p>",
                "is-referenced-by-count": 12,
                "subject": ["Geriatrics"],
                "type": "journal-article",
                "URL": "https://doi.org/10.1000/xyz",
                "link": [{"content-type": "application/pdf", "URL": "https://example.org/a.pdf"}],
            }
        ]
    }
}


class CrossrefTests(unittest.TestCase):
    def test_search_parses_payload(self):
        with patch.object(sources, "fetch_json", return_value=CROSSREF_PAYLOAD):
            result = sources.search_crossref(parse_intent("muscle strength"), SETTINGS, 5)
        self.assertIsNone(result.warning)
        paper = result.papers[0]
        self.assertEqual(paper.doi, "10.1000/xyz")
        self.assertEqual(paper.year, 2018)
        self.assertEqual(paper.abstract, "Strength increased.")
        self.assertEqual(paper.authors, ["Smith Jane", "Consortium X"])
        self.assertTrue(paper.is_oa)
        self.assertEqual(paper.oa_url, "https://example.org/a.pdf")

    def test_search_uses_date_filters(self):
        spec = parse_intent("muscle strength since 2015")
        captured = {}

        def fake(url, settings, **kwargs):
            captured["url"] = url
            return CROSSREF_PAYLOAD

        with patch.object(sources, "fetch_json", side_effect=fake):
            sources.search_crossref(spec, SETTINGS, 5)
        self.assertIn("from-pub-date", captured["url"])
        self.assertIn("2015", captured["url"])


ARXIV_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Learned models of muscle strength</title>
    <summary>We model strength with neural networks.</summary>
    <author><name>A. Researcher</name></author>
    <author><name>B. Researcher</name></author>
    <arxiv:doi>10.1000/arxiv.doi</arxiv:doi>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


class ArxivTests(unittest.TestCase):
    def test_search_parses_atom(self):
        with patch.object(sources, "fetch", return_value=ARXIV_XML):
            result = sources.search_arxiv(parse_intent("muscle strength"), SETTINGS, 5)
        self.assertIsNone(result.warning)
        paper = result.papers[0]
        self.assertEqual(paper.title, "Learned models of muscle strength")
        self.assertEqual(paper.year, 2024)
        self.assertEqual(paper.arxiv_id, "2401.00001v1")
        self.assertEqual(paper.doi, "10.1000/arxiv.doi")
        self.assertEqual(paper.authors, ["A. Researcher", "B. Researcher"])
        self.assertTrue(paper.is_oa)
        self.assertIn("2401.00001v1", paper.oa_url)

    def test_bad_xml_is_a_warning(self):
        with patch.object(sources, "fetch", return_value=b"<not xml"):
            result = sources.search_arxiv(parse_intent("muscle"), SETTINGS, 5)
        self.assertIsNotNone(result.warning)
        self.assertEqual(result.papers, [])


class ArxivIdentifierTests(unittest.TestCase):
    def test_parses_modern_legacy_and_url_forms(self):
        for value in (
            "2401.12345",
            "2401.12345v2",
            "arXiv:2401.12345",
            "https://arxiv.org/abs/2401.12345",
            "https://arxiv.org/pdf/2401.12345.pdf",
            "cs/0002001",
            "alg-geom/9708001v1",
        ):
            self.assertIsNotNone(sources.arxiv_id(value), value)

    def test_does_not_mistake_a_doi_or_title_for_an_arxiv_id(self):
        self.assertIsNone(sources.arxiv_id("10.1000/xyz"))
        self.assertIsNone(sources.arxiv_id("Learned models of muscle strength"))

    def test_fetch_by_arxiv_reuses_the_entry_parser(self):
        with patch.object(sources, "fetch", return_value=ARXIV_XML):
            paper = sources.fetch_by_arxiv("https://arxiv.org/abs/2401.00001v1", SETTINGS)
        self.assertIsNotNone(paper)
        self.assertEqual(paper.arxiv_id, "2401.00001v1")
        self.assertEqual(paper.title, "Learned models of muscle strength")

    def test_fetch_by_arxiv_returns_none_for_a_bad_id(self):
        self.assertIsNone(sources.fetch_by_arxiv("not-an-id", SETTINGS))


class OpenAlexTests(unittest.TestCase):
    def test_inverted_abstract_is_reconstructed(self):
        inverted = {"Muscle": [0], "strength": [1], "declines.": [2]}
        self.assertEqual(sources.reconstruct_abstract(inverted), "Muscle strength declines.")
        self.assertIsNone(sources.reconstruct_abstract(None))

    def test_rate_limit_message_is_actionable(self):
        inverted = None
        from papersplease.net import FetchError

        with patch.object(sources, "fetch_json", side_effect=FetchError("u", 429, "Rate limit")):
            result = sources.search_openalex(parse_intent("muscle"), SETTINGS, 5)
        self.assertIn("PAPERSPLEASE_OPENALEX_KEY", result.warning)


class CleanTextTests(unittest.TestCase):
    def test_markup_and_entities(self):
        self.assertEqual(sources.clean_text("<p>a &amp; b</p>"), "a & b")
        self.assertIsNone(sources.clean_text("   "))

    def test_doi_normalisation(self):
        self.assertEqual(sources.norm_doi("https://doi.org/10.1038/NATURE1"), "10.1038/nature1")
        self.assertEqual(sources.norm_doi("doi:10.1000/x"), "10.1000/x")
        self.assertIsNone(sources.norm_doi("not-a-doi"))


if __name__ == "__main__":
    unittest.main()
