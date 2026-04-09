import unittest

from gnme_inflation import (
    GNMEProblem,
    build_top_down_block_task_feasibility_model,
    build_top_down_sdp_draft,
    validate_top_down_draft,
)


class TestGNMETopDownBlueprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.level2 = GNMEProblem(
            n_parties=3,
            inflation_level=2,
            local_dims_per_party=(2, 2, 2),
            verbose=0,
        )
        cls.level3 = GNMEProblem(
            n_parties=3,
            inflation_level=3,
            local_dims_per_party=(2, 2, 2),
            verbose=0,
        )

    def test_level2_top_down_counts(self):
        blueprint = self.level2.top_down_family_blueprint()
        self.assertEqual(len(blueprint.maximal_families), 3)
        self.assertEqual(len(blueprint.overlap_families), 3)
        self.assertEqual(len(blueprint.known_families), 4)
        self.assertEqual(len(blueprint.all_shared_subset_classes), 18)

    def test_level2_maximal_families_are_four_body_disconnected_pairs(self):
        blueprint = self.level2.top_down_family_blueprint()
        labels = {
            family.representative_occurrence.labels
            for family in blueprint.maximal_families
        }
        self.assertSetEqual(
            labels,
            {
                ("A_11", "A_22", "B_11", "B_22"),
                ("A_11", "A_22", "C_11", "C_22"),
                ("B_11", "B_22", "C_11", "C_22"),
            },
        )
        self.assertSetEqual(
            {len(family.representative_occurrence.labels) for family in blueprint.maximal_families},
            {4},
        )
        self.assertSetEqual(
            {len(family.local_symmetry_perms) for family in blueprint.maximal_families},
            {1, 2},
        )

    def test_level2_overlap_families_are_two_body_bridges(self):
        blueprint = self.level2.top_down_family_blueprint()
        labels = {
            family.representative_occurrence.labels
            for family in blueprint.overlap_families
        }
        self.assertSetEqual(
            labels,
            {
                ("A_11", "A_22"),
                ("B_11", "B_22"),
                ("C_11", "C_22"),
            },
        )
        self.assertTrue(all(len(family.routes) == 2 for family in blueprint.overlap_families))
        self.assertTrue(all(family.kind == "overlap" for family in blueprint.overlap_families))

    def test_level3_top_down_counts(self):
        blueprint = self.level3.top_down_family_blueprint()
        self.assertEqual(len(blueprint.maximal_families), 6)
        self.assertEqual(len(blueprint.overlap_families), 41)
        self.assertEqual(len(blueprint.known_families), 4)
        self.assertEqual(len(blueprint.all_shared_subset_classes), 173)

    def test_level3_maximal_families_are_seven_body(self):
        blueprint = self.level3.top_down_family_blueprint()
        self.assertSetEqual(
            {len(family.representative_occurrence.labels) for family in blueprint.maximal_families},
            {7},
        )
        self.assertSetEqual(
            {family.kind for family in blueprint.maximal_families},
            {"maximal"},
        )
        self.assertSetEqual(
            {len(family.local_symmetry_perms) for family in blueprint.maximal_families},
            {6, 36},
        )

    def test_level3_overlap_families_span_descendants_of_maximal_layer(self):
        blueprint = self.level3.top_down_family_blueprint()
        self.assertSetEqual(
            {len(family.representative_occurrence.labels) for family in blueprint.overlap_families},
            {2, 3, 4, 5, 6},
        )
        self.assertTrue(all(len(family.routes) >= 2 for family in blueprint.overlap_families))
        self.assertTrue(all(family.kind == "overlap" for family in blueprint.overlap_families))

    def test_level2_top_down_draft_shape(self):
        draft = build_top_down_sdp_draft(self.level2, verbose=0)
        self.assertEqual(len(draft.psd_variables), 2)
        self.assertEqual(len(draft.maximal_representatives), 3)
        self.assertEqual(len(draft.known_representatives), 1)
        self.assertEqual(len(draft.tau_representative_constraints), 7)
        self.assertEqual(len(draft.ppt_variables), 4)
        self.assertEqual(len(draft.ppt_constraints), 4)
        self.assertEqual(len(draft.cross_inflation_groups), 4)
        self.assertTrue(draft.verified_at_draft_time)

    def test_level2_top_down_draft_uses_maximal_family_ppts(self):
        draft = build_top_down_sdp_draft(self.level2, verbose=0)
        self.assertSetEqual(
            {rep.target_lexorder for rep in draft.maximal_representatives},
            {
                ("A_11", "A_22", "B_11", "B_22"),
                ("A_11", "A_22", "C_11", "C_22"),
                ("B_11", "B_22", "C_11", "C_22"),
            },
        )
        self.assertEqual(
            [rep.target_lexorder for rep in draft.known_representatives],
            [("A_11", "B_11", "C_11")],
        )
        self.assertEqual(
            sorted(constraint.source_variable_kind for constraint in draft.ppt_constraints),
            ["mu", "mu", "mu", "tau"],
        )

    def test_level2_top_down_draft_validation(self):
        draft = build_top_down_sdp_draft(self.level2, verbose=0)
        stats = validate_top_down_draft(draft, verbose=0)
        self.assertDictEqual(
            stats,
            {
                "errors": 0,
                "warnings": 0,
                "constraints_checked": 7,
                "cross_inflation_groups_checked": 4,
            },
        )

    def test_level2_top_down_task_builder_smoke(self):
        draft = build_top_down_sdp_draft(self.level2, verbose=0)
        model = build_top_down_block_task_feasibility_model(draft, Hermitian=False, verbose=0)
        self.assertEqual(model.task.getnumbarvar(), 5)
        self.assertEqual(model.constraint_counts["trace"], 2)
        self.assertEqual(model.constraint_counts["representative"], 408)
        self.assertEqual(model.constraint_counts["ppt_direct"], 1)
        self.assertEqual(len(model.auxiliary_bar_variables), 0)
        self.assertEqual(len(model.known_representative_bar_variables), 0)
        self.assertEqual(len(model.ppt_bar_variables), 1)
        self.assertEqual(len(model.representative_anchors), 4)
        self.assertEqual(
            set(model.representative_anchors),
            {rep.name for rep in draft.maximal_representatives + draft.known_representatives},
        )
        self.assertGreater(model.constraint_counts["representative"], 0)
        self.assertGreater(model.task.getnumcon(), 0)


if __name__ == "__main__":
    unittest.main()
