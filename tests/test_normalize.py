"""Intent normalisation: Dutch sentences and gym shorthand become English."""

import unittest

from paperhound.intent import parse_intent
from paperhound.normalize import normalize_intent


DUTCH = (
    "optimale rep en set count en uren en schema en oefeningen in de gym "
    "voor man 18 jaar oud voor spiergroei"
)


class NormalizeTests(unittest.TestCase):
    def test_dutch_training_question_is_translated(self):
        text, notes = normalize_intent(DUTCH)
        self.assertIn("muscle hypertrophy", text)
        self.assertIn("training program", text)
        self.assertIn("number of sets", text)
        self.assertIn("hours per week", text)
        self.assertIn("men", text)
        self.assertNotIn("spiergroei", text)
        self.assertNotIn(" oefeningen", text)
        self.assertTrue(any("translated Dutch" in note for note in notes))
        self.assertIn("parsed as", notes[-1])

    def test_english_intent_is_left_alone(self):
        text, notes = normalize_intent("effect of resistance training on muscle strength")
        self.assertEqual(text, "effect of resistance training on muscle strength")
        self.assertEqual(notes, [])

    def test_shorthand_is_expanded_without_translating_english(self):
        text, notes = normalize_intent("3 sets of 10 reps for hypertrophy")
        self.assertIn("repetitions", text)
        self.assertNotIn("translated Dutch", " ".join(notes))
        self.assertEqual(notes[0], "expanded training shorthand: reps")

    def test_single_dutch_marker_is_not_enough(self):
        # \"van\" alone can be a name; translating on one hit would mangle an
        # English intent that happens to contain it.
        text, _ = normalize_intent("effect of training in van der Waals forces")
        self.assertEqual(text, "effect of training in van der Waals forces")


class DutchIntentParsingTests(unittest.TestCase):
    def setUp(self):
        self.spec = parse_intent(DUTCH)

    def test_no_dutch_function_words_become_facets(self):
        names = {facet.name for facet in self.spec.facets}
        for junk in ("en", "de", "voor", "jaar", "oud", "schema", "oefeningen"):
            self.assertNotIn(junk, names)
        self.assertIn("hypertrophy", names)
        self.assertIn("volume", names)
        self.assertIn("frequency", names)

    def test_facet_count_stays_screenable(self):
        self.assertLessEqual(len(self.spec.facets), 10)
        self.assertGreaterEqual(len(self.spec.facets), 4)

    def test_age_phrase_becomes_a_population(self):
        notes = " ".join(self.spec.extra_notes)
        self.assertIn("young adults", notes)
        self.assertIn("age 18", notes)

    def test_no_head_is_invented_for_a_list_of_variables(self):
        self.assertIsNone(self.spec.head)
        self.assertIn("lists several concepts", " ".join(self.spec.extra_notes))

    def test_anchors_are_the_variables_not_the_outcome_or_population(self):
        # The question is about sets, reps and frequency *for* muscle growth.
        # Hypertrophy is the goal every tangential training study measures, so
        # it must not be an anchor; the variables are what the paper is about.
        self.assertEqual(set(self.spec.anchors), {"volume", "frequency"})
        self.assertNotIn("hypertrophy", self.spec.anchors)
        self.assertNotIn("young adults", self.spec.anchors)
        self.assertNotIn("adult", self.spec.anchors)

    def test_query_uses_english_terms(self):
        query = self.spec.query()
        self.assertIn("hypertrophy", query)
        self.assertNotIn("spiergroei", query)

    def test_explain_payload_carries_the_translation(self):
        from paperhound.pipeline import explain_intent
        from paperhound.config import Settings

        payload = explain_intent(DUTCH, settings=Settings(cache=False))
        self.assertTrue(payload["facets"])
        self.assertTrue(any("parsed as" in note for note in payload["notes"]))

    def test_head_is_still_found_without_a_list(self):
        spec = parse_intent("muscle strength in older adults")
        self.assertEqual(spec.head.name, "strength")


class RequireInTests(unittest.TestCase):
    def test_require_in_populates_the_spec(self):
        spec = parse_intent(
            "muscle strength in older adults",
            require=["strength", "grip"],
            require_in="title",
        )
        self.assertEqual(spec.require_field, "title")
        self.assertEqual([term for term, _ in spec.required], ["strength", "grip"])

    def test_require_in_without_require_is_noted(self):
        spec = parse_intent("muscle strength", require_in="title")
        self.assertEqual(spec.required, [])
        self.assertIn("no effect without --require", " ".join(spec.extra_notes))

    def test_bad_field_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_intent("muscle strength", require=["strength"], require_in="abstract_only")


if __name__ == "__main__":
    unittest.main()
