"""Intent parsing tests: facets, date filters, negations, overrides."""

import unittest

from papersplease.intent import exclusion_terms, parse_intent


class FacetTests(unittest.TestCase):
    def test_specific_intent_becomes_separate_facets(self):
        spec = parse_intent("muscle strength in older adults")
        names = {facet.name for facet in spec.facets}
        self.assertEqual(names, {"muscle", "strength", "older", "adult"})

    def test_synonyms_are_carried_on_each_facet(self):
        spec = parse_intent("muscle strength in older adults")
        older = next(f for f in spec.facets if f.name == "older")
        self.assertIn("elderly", older.terms)
        strength = next(f for f in spec.facets if f.name == "strength")
        self.assertIn("force", strength.terms)

    def test_generic_words_weigh_less(self):
        spec = parse_intent("effect of muscle strength")
        weights = {f.name: f.weight for f in spec.facets}
        self.assertLess(weights["effect"], weights["muscle"])
        self.assertEqual(weights["muscle"], 1.0)

    def test_filler_words_are_dropped(self):
        spec = parse_intent("please find me some papers about muscle strength")
        names = {f.name for f in spec.facets}
        self.assertNotIn("papers", names)
        self.assertNotIn("find", names)
        self.assertIn("strength", names)

    def test_require_forces_a_facet(self):
        spec = parse_intent("exercise in older people", require=["strength"])
        self.assertIn("strength", {f.name for f in spec.facets})
        strength = next(f for f in spec.facets if f.name == "strength")
        self.assertGreaterEqual(strength.weight, 1.2)

    def test_empty_intent_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_intent("   ")


class DateTests(unittest.TestCase):
    def test_since(self):
        spec = parse_intent("muscle strength in older adults since 2015")
        self.assertEqual(spec.year_from, 2015)
        self.assertIsNone(spec.year_to)
        self.assertNotIn("2015", {f.name for f in spec.facets})

    def test_range(self):
        spec = parse_intent("resistance training 2010-2018")
        self.assertEqual((spec.year_from, spec.year_to), (2010, 2018))

    def test_before(self):
        spec = parse_intent("sarcopenia before 2012")
        self.assertEqual(spec.year_to, 2012)

    def test_reversed_range_is_normalised(self):
        spec = parse_intent("sarcopenia 2018 to 2010")
        self.assertEqual((spec.year_from, spec.year_to), (2010, 2018))


class ExclusionTests(unittest.TestCase):
    def test_not_about_is_an_exclusion(self):
        spec = parse_intent("muscle strength not about rodents")
        self.assertEqual(spec.exclude, ["rodents"])
        self.assertNotIn("rodent", {f.name for f in spec.facets})

    def test_minus_shorthand(self):
        spec = parse_intent("muscle strength -mice")
        self.assertIn("mice", spec.exclude)

    def test_cli_exclude(self):
        spec = parse_intent("muscle strength", exclude=["cancer"])
        self.assertIn("cancer", spec.exclude)


class NegationRuleTests(unittest.TestCase):
    """Regressions from the substring-matching negation parser."""

    def test_no_inside_a_word_is_not_a_negation(self):
        # "amino acid" used to match the cue "no " and eat the whole intent.
        spec = parse_intent("amino acid supplementation in older adults")
        self.assertEqual(spec.exclude, [])
        names = {f.name for f in spec.facets}
        self.assertIn("older", names)
        self.assertIn("adult", names)

    def test_but_not_leaves_no_connective_facet(self):
        spec = parse_intent("strength training in older adults but not cancer")
        self.assertEqual(spec.exclude, ["cancer"])
        names = {f.name for f in spec.facets}
        self.assertNotIn("but", names)
        self.assertNotEqual(spec.head.name, "but")

    def test_head_stays_on_the_leading_topic_phrase(self):
        # Blanking phrases before head detection made "adult" the head here.
        spec = parse_intent("strength training in older adults")
        self.assertEqual(spec.head.name, "resistance training")

    def test_generic_word_never_becomes_the_head(self):
        spec = parse_intent("effect of resistance training on muscle strength")
        self.assertEqual(spec.head.name, "resistance training")

    def test_exclusion_stops_at_the_population(self):
        spec = parse_intent("resistance training without equipment in older adults")
        self.assertEqual(spec.exclude, ["equipment"])
        names = {f.name for f in spec.facets}
        self.assertIn("older", names)
        self.assertIn("adult", names)

    def test_exclusion_covers_every_conjunct(self):
        spec = parse_intent("muscle strength not about rodents and cancer")
        self.assertEqual(spec.exclude, ["cancer", "rodents"])

    def test_contrastive_phrasing_is_emphasis_not_exclusion(self):
        # "not just about muscles but about muscle strength" must not exclude
        # "muscles": that would reject every paper on the actual topic.
        spec = parse_intent("not just about muscles but about muscle strength")
        self.assertEqual(spec.exclude, [])
        self.assertEqual(spec.head.name, "strength")
        self.assertTrue(spec.extra_notes)

    def test_but_not_just_about_is_also_contrastive(self):
        spec = parse_intent("muscle strength in older adults but not just about muscles")
        self.assertEqual(spec.exclude, [])
        self.assertEqual(spec.head.name, "strength")

    def test_no_significant_difference_is_not_an_exclusion(self):
        spec = parse_intent("no significant difference in muscle strength in older adults")
        self.assertEqual(spec.exclude, [])

    def test_excluded_words_expand_to_synonyms_and_plurals(self):
        terms = exclusion_terms("rodents")
        for expected in ("rodent", "mice", "murine"):
            self.assertIn(expected, terms)
        self.assertIn("animal study", exclusion_terms("animal studies"))


if __name__ == "__main__":
    unittest.main()
