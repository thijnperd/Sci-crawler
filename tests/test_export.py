"""BibTeX and RIS export."""

import json
import unittest

from papersplease.export import to_bibtex, to_csl_json, to_ris
from papersplease.models import Paper, Verdict
from papersplease.pipeline import Finding


def finding(**kwargs) -> Finding:
    paper = Paper(
        title=kwargs.pop("title", "Effects of resistance training volume on hypertrophy"),
        doi=kwargs.pop("doi", "10.1000/abc"),
        year=kwargs.pop("year", 2021),
        authors=kwargs.pop("authors", ["Schoenfeld Brad", "Grgic Jozo"]),
        journal=kwargs.pop("journal", "Journal of Strength Research"),
    )
    verdict = Verdict(
        doi=paper.doi, title=paper.title, score=0.9, coverage=0.9,
        title_coverage=0.8, bucket="in_scope", in_scope=True,
    )
    return Finding(paper=paper, verdict=verdict)


class PlainPaperTests(unittest.TestCase):
    def test_export_accepts_papers_without_a_verdict(self):
        paper = Paper(title="Sets and reps", doi="10.1/x", year=2020, authors=["Doe Jane"])
        self.assertIn("@article{", to_bibtex([paper]))
        self.assertIn("TY  - JOUR", to_ris([paper]))
        self.assertIn("DO  - 10.1/x", to_ris([paper]))


class BibtexTests(unittest.TestCase):
    def test_entry_has_the_fields_a_latex_user_needs(self):
        text = to_bibtex([finding()])
        self.assertTrue(text.startswith("@article{"))
        self.assertIn("title = {Effects of resistance training volume on hypertrophy}", text)
        self.assertIn("author = {Schoenfeld Brad and Grgic Jozo}", text)
        self.assertIn("year = {2021}", text)
        self.assertIn("doi = {10.1000/abc}", text)
        self.assertTrue(text.rstrip().endswith("}"))

    def test_keys_are_unique_for_same_author_and_year(self):
        text = to_bibtex([finding(), finding(title="Another paper on sets")])
        keys = [line.split("{", 1)[1].rstrip(",") for line in text.splitlines() if line.startswith("@article{")]
        self.assertEqual(len(keys), 2)
        self.assertEqual(len(set(keys)), 2)

    def test_ampersands_are_escaped(self):
        self.assertIn(r"\&", to_bibtex([finding(title="Sets & reps")]))

    def test_empty_list_is_empty_output(self):
        self.assertEqual(to_bibtex([]), "")


class CslJsonTests(unittest.TestCase):
    def test_record_has_the_fields_citeproc_reads(self):
        records = json.loads(to_csl_json([finding()]))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["title"], "Effects of resistance training volume on hypertrophy")
        self.assertEqual(record["DOI"], "10.1000/abc")
        self.assertEqual(record["issued"], {"date-parts": [[2021]]})
        self.assertEqual(record["container-title"], "Journal of Strength Research")
        self.assertEqual(record["author"][0], {"family": "Schoenfeld", "given": "Brad"})

    def test_plain_papers_are_accepted_and_ids_are_stable(self):
        paper = Paper(title="Sets and reps", doi="10.1/x", year=2020, authors=["Doe Jane"])
        record = json.loads(to_csl_json([paper]))[0]
        self.assertEqual(record["id"], "10.1/x")
        self.assertEqual(record["author"], [{"family": "Doe", "given": "Jane"}])


class RisTests(unittest.TestCase):
    def test_record_has_type_authors_and_end(self):
        lines = to_ris([finding()]).splitlines()
        self.assertEqual(lines[0], "TY  - JOUR")
        self.assertIn("TI  - Effects of resistance training volume on hypertrophy", lines)
        self.assertIn("AU  - Schoenfeld Brad", lines)
        self.assertIn("AU  - Grgic Jozo", lines)
        self.assertIn("DO  - 10.1000/abc", lines)
        self.assertIn("ER  - ", lines)

    def test_two_records(self):
        text = to_ris([finding(), finding(doi="10.1000/def")])
        self.assertEqual(text.count("TY  - JOUR"), 2)
        self.assertEqual(text.count("ER  - "), 2)


if __name__ == "__main__":
    unittest.main()
