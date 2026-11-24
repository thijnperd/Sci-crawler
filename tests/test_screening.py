"""The behaviour that matters: a broad paper must not pass as a specific one."""

import unittest

from papersplease.intent import parse_intent
from papersplease.models import Paper
from papersplease.screening import combine, rank, screen_many, screen_one

INTENT = "muscle strength in older adults"


def paper(title, abstract="", year=2020, **kwargs):
    return Paper(title=title, abstract=abstract, year=year, **kwargs)


class SpecificityTests(unittest.TestCase):
    def setUp(self):
        self.spec = parse_intent(INTENT)

    def test_broad_muscle_paper_is_rejected(self):
        verdict = screen_one(
            paper(
                "Muscle fibre type composition in mammalian tissue",
                "Muscle fibres differ in their contractile proteins.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertFalse(verdict.in_scope)
        self.assertLess(verdict.coverage, 0.5)
        self.assertIn("strength", verdict.missing)

    def test_muscle_mass_is_not_strength(self):
        verdict = screen_one(
            paper(
                "Age-related muscle mass loss in older adults",
                "We measured lean mass and muscle thickness in older adults over ten years.",
            ),
            self.spec,
        )
        # Muscle + older + adult but no strength: partial, never in scope.
        self.assertNotEqual(verdict.bucket, "in_scope")
        self.assertIn("strength", verdict.missing)

    def test_on_target_paper_passes(self):
        verdict = screen_one(
            paper(
                "Grip strength and physical function in older adults",
                "Handgrip strength was measured in 300 older adults; strength predicted mobility.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "in_scope")
        self.assertTrue(verdict.in_scope)
        self.assertGreaterEqual(verdict.coverage, 0.7)

    def test_wrong_population_is_adjacent_not_in_scope(self):
        verdict = screen_one(
            paper(
                "Muscle strength adaptations to resistance training in young athletes",
                "Strength and muscle size increased after training.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "adjacent")
        self.assertIn("older", verdict.missing)

    def test_title_only_match_is_penalised(self):
        with_title = screen_one(
            paper("Muscle strength in older adults", "Strength was assessed."), self.spec
        )
        abstract_only = screen_one(
            paper("A cohort study of physical performance", "Muscle strength in older adults was measured."),
            self.spec,
        )
        self.assertGreater(with_title.score, abstract_only.score)

    def test_excluded_concept_disqualifies(self):
        spec = parse_intent(f"{INTENT} not about rodents")
        verdict = screen_one(
            paper(
                "Muscle strength in older adults: a controlled study",
                "Muscle strength improved in older adults, and in rodents the effect differed.",
            ),
            spec,
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertIn("rodents", verdict.excluded_reason)

    def test_excluded_synonym_disqualifies(self):
        # Excluding "rodents" has to catch a paper that only says "murine".
        spec = parse_intent(f"{INTENT} not about rodents")
        verdict = screen_one(
            paper(
                "Muscle strength in older adults: a controlled study",
                "Strength improved in older adults; murine tissue was assayed in parallel.",
            ),
            spec,
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertIn("rodents", verdict.excluded_reason)
        self.assertIn("murine", verdict.excluded_reason)

    def test_excluded_singular_matches_plural(self):
        spec = parse_intent(f"{INTENT} not about rodent")
        verdict = screen_one(
            paper(
                "Muscle strength in older adults: a controlled study",
                "Strength improved in older adults; rodents were also used.",
            ),
            spec,
        )
        self.assertEqual(verdict.bucket, "out_of_scope")

    def test_year_filter(self):
        spec = parse_intent(f"{INTENT} since 2020")
        verdict = screen_one(
            paper("Muscle strength in older adults", "Strength in older adults.", year=2014), spec
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertIn("2014", verdict.excluded_reason)

    def test_keywords_count_as_evidence(self):
        paper_obj = paper("Body composition and function in the elderly", "Lean mass was measured.")
        paper_obj.keywords = ["muscle", "strength"]
        verdict = screen_one(paper_obj, self.spec)
        self.assertIn("muscle", verdict.matched)
        self.assertIn("strength", verdict.matched)

    def test_matched_and_missing_are_reported(self):
        verdict = screen_one(paper("Muscle strength in older adults", "Strength."), self.spec)
        self.assertTrue(verdict.reason)
        self.assertIn("matched", verdict.reason)

    def test_facets_must_be_discussed_together(self):
        # All four concepts appear, but never in one sentence or in the title:
        # this is keyword scatter, not a paper about muscle strength in ageing.
        verdict = screen_one(
            paper(
                "Swallowing function in older adults",
                "Dysphagia is common in this group. Strength of the tongue was tested elsewhere. "
                "Muscle tissue was imaged separately.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.coverage, 1.0)
        self.assertEqual(verdict.bucket, "adjacent")
        self.assertIn("never appears with", verdict.reason)

    def test_excerpt_is_read(self):
        paper_obj = paper("A performance study", "No summary available.")
        verdict_before = screen_one(paper_obj, self.spec)
        paper_obj.excerpt = (
            "Muscle strength in older adults declines with age; strength training helps."
        )
        verdict_after = screen_one(paper_obj, self.spec)
        self.assertGreater(verdict_after.coverage, verdict_before.coverage)

    def test_screening_is_stable_across_a_list(self):
        papers = [
            paper("Muscle strength in older adults", "Strength in older adults."),
            paper("Muscle fibre typing", "Muscle fibres."),
        ]
        verdicts = screen_many(papers, self.spec)
        self.assertEqual([v.bucket for v in verdicts], ["in_scope", "out_of_scope"])


class LlmCombineTests(unittest.TestCase):
    def setUp(self):
        self.spec = parse_intent(INTENT)
        self.verdict = screen_one(
            paper(
                "Muscle strength in older adults",
                "Strength was measured in older adults.",
            ),
            self.spec,
        )

    def test_llm_veto_moves_paper_out_of_scope(self):
        combine(
            self.verdict,
            {"verdict": "out_of_scope", "specificity": "too_broad", "score": 0.2, "reason": "about mass"},
        )
        self.assertEqual(self.verdict.bucket, "out_of_scope")
        self.assertIn("reviewer", self.verdict.reason)
        self.assertLess(self.verdict.score, 0.8)

    def test_llm_promotes_adjacent(self):
        adjacent = screen_one(
            paper("Muscle strength in trained young athletes", "Strength in athletes."), self.spec
        )
        self.assertNotEqual(adjacent.bucket, "in_scope")
        combine(
            adjacent,
            {"verdict": "in_scope", "specificity": "on_target", "score": 0.9, "reason": "population includes adults"},
        )
        self.assertEqual(adjacent.bucket, "in_scope")

    def test_missing_llm_keeps_lexical_verdict(self):
        before = self.verdict.bucket
        combine(self.verdict, None)
        self.assertEqual(self.verdict.bucket, before)


class ListIntentAnchorTests(unittest.TestCase):
    """A list of variables must be screened like a topic, not a checklist."""

    INTENT = (
        "optimale rep en set count en uren en schema en oefeningen in de gym "
        "voor man 18 jaar oud voor spiergroei"
    )

    def setUp(self):
        self.spec = parse_intent(self.INTENT)

    def test_tangential_trial_is_demoted(self):
        # A creatine trial that recorded sets, reps and sessions can reach 70%
        # coverage by mentioning every variable once in its methods. It must not
        # pass as a paper about training variables.
        verdict = screen_one(
            paper(
                "Creatine supplementation and resistance training adaptations in older men",
                "Participants performed resistance training with 3 sets of 10 repetitions, "
                "3 sessions per week for 8 weeks. Muscle mass and strength increased in young and older men.",
            ),
            self.spec,
        )
        self.assertGreaterEqual(verdict.coverage, 0.7)
        self.assertNotEqual(verdict.bucket, "in_scope")
        self.assertIn("title", verdict.reason)

    def test_outcome_only_title_is_demoted(self):
        # One anchor (the outcome) in the title is not enough: supplement
        # studies measure hypertrophy without studying training volume.
        verdict = screen_one(
            paper(
                "Vegan and Omnivorous High Protein Diets Support Comparable Skeletal Muscle Hypertrophy",
                "Participants completed resistance training: 3 sets of 10 repetitions, "
                "3 sessions per week. Muscle mass and strength increased.",
            ),
            self.spec,
        )
        self.assertNotEqual(verdict.bucket, "in_scope")
        self.assertIn("title names", verdict.reason)

    def test_paper_naming_a_key_concept_in_the_title_passes(self):
        verdict = screen_one(
            paper(
                "Weekly resistance training volume and frequency for muscle hypertrophy in young men",
                "Volume progressed from 10 to 20 sets per muscle per week over three sessions per week. "
                "Muscle thickness and lean mass increased in young men.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "in_scope")


class RequireInFieldTests(unittest.TestCase):
    def setUp(self):
        from papersplease.intent import parse_intent

        self.spec = parse_intent(
            "muscle strength in older adults", require=["strength"], require_in="title"
        )

    def test_term_present_in_title_passes(self):
        verdict = screen_one(
            paper("Grip strength in older adults", "Muscle strength in older adults."),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "in_scope")

    def test_term_only_in_the_abstract_is_rejected(self):
        verdict = screen_one(
            paper("Grip in older adults", "Muscle strength in older adults."), self.spec
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertIn("required term", verdict.reason)
        self.assertIn("strength", verdict.reason)
        self.assertIn("title", verdict.reason)

    def test_plural_counts_as_the_same_word(self):
        verdict = screen_one(
            paper("Grip strengths in older adults", "Muscle strengths in older adults."),
            self.spec,
        )
        self.assertNotIn("required term", verdict.reason)


class CausalDirectionTests(unittest.TestCase):
    def setUp(self):
        self.spec = parse_intent("influence of bad diet on diabetes type 1")

    def test_aligned_diet_to_type_1_diabetes_passes(self):
        verdict = screen_one(
            paper(
                "Ultra-processed food and unhealthy diet increase risk of type 1 diabetes",
                "Poor dietary quality and ultra-processed food intake increase type 1 diabetes risk.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "in_scope")
        self.assertEqual(verdict.direction, "aligned")

    def test_reversed_diabetes_to_eating_habits_is_rejected(self):
        verdict = screen_one(
            paper(
                "The influence of type 1 diabetes on eating habits and poor diet quality",
                "Type 1 diabetes alters eating habits and causes disordered eating behaviors.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertEqual(verdict.direction, "reversed")
        self.assertIn("reversed direction", verdict.reason)

    def test_descriptive_diet_in_type_1_diabetes_is_demoted_to_adjacent(self):
        verdict = screen_one(
            paper(
                "Dietary habits in adolescents with type 1 diabetes",
                "We surveyed poor diet quality and meal timing in 120 adolescents with type 1 diabetes.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "adjacent")
        self.assertEqual(verdict.direction, "descriptive")

    def test_type_2_diabetes_title_is_rejected_while_mixed_title_passes_guard(self):
        t2 = screen_one(
            paper(
                "Poor diet and type 2 diabetes risk",
                "Unhealthy diet increases type 2 diabetes incidence.",
            ),
            self.spec,
        )
        self.assertEqual(t2.bucket, "out_of_scope")
        self.assertIn("type 2", t2.excluded_reason)

        mixed = screen_one(
            paper(
                "Poor diet in youth with type 1 or type 2 diabetes",
                "Poor diet worsens glycemic control in type 1 diabetes.",
            ),
            self.spec,
        )
        self.assertIsNone(mixed.excluded_reason)


class SenseDisambiguationAndRankingTests(unittest.TestCase):
    def setUp(self):
        self.spec = parse_intent("optimal training for hypertrophy in younger adults since 2015")

    def test_cardiac_hypertrophy_is_rejected_as_wrong_domain_sense(self):
        verdict = screen_one(
            paper(
                "Left ventricular hypertrophy in young adult athletes after training",
                "Exercise training induced left ventricular hypertrophy in young adults.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "out_of_scope")
        self.assertIn("wrong domain sense", verdict.reason)

    def test_skeletal_muscle_hypertrophy_in_young_adults_passes(self):
        verdict = screen_one(
            paper(
                "Resistance training volume and skeletal muscle hypertrophy in young adults",
                "Exercise training induced skeletal muscle hypertrophy in young adults.",
            ),
            self.spec,
        )
        self.assertEqual(verdict.bucket, "in_scope")

    def test_rank_keeps_in_scope_above_highly_cited_adjacent(self):
        in_p = paper(
            "Resistance training volume and skeletal muscle hypertrophy in young adults",
            "Exercise training induced skeletal muscle hypertrophy in young adults.",
            cited_by=1,
            year=2023,
        )
        adj_p = paper(
            "Exercise training in older adults",
            "Muscle hypertrophy after exercise training in older adults.",
            cited_by=10000,
            year=2020,
        )
        in_v = screen_one(in_p, self.spec)
        adj_v = screen_one(adj_p, self.spec)
        self.assertEqual(in_v.bucket, "in_scope")
        self.assertEqual(adj_v.bucket, "adjacent")

        ranked = rank([adj_p, in_p], [adj_v, in_v])
        self.assertEqual(ranked[0][0].title, in_p.title)


class RegressionGuardTests(unittest.TestCase):
    def test_unicode_dashes_normalize_and_match_facets(self):
        spec = parse_intent("muscle strength in older adults")
        verdict = screen_one(
            paper(
                "Muscle\u2013strength training in older\u2014adults",
                "Non\u2010invasive training improved maximal\u2212force.",
            ),
            spec,
        )
        self.assertEqual(verdict.bucket, "in_scope")

    def test_rank_raises_on_length_mismatch(self):
        spec = parse_intent("muscle")
        papers = [paper("One", ""), paper("Two", ""), paper("Three", "")]
        verdicts = [screen_one(papers[0], spec)]
        with self.assertRaises(ValueError):
            rank(papers, verdicts)

    def test_parse_intent_rejects_invalid_min_scope(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                parse_intent("muscle strength", min_scope=bad)


if __name__ == "__main__":
    unittest.main()
