import unittest

from gnme_inflation import GNMEProblem, build_sdp_draft


class TestGNMEDefaultBlueprint(unittest.TestCase):
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

    def test_level2_default_drops_pairwise_known_anchors(self):
        draft = build_sdp_draft(self.level2, verbose=0)
        self.assertEqual(
            [rep.target_lexorder for rep in draft.known_representatives],
            [("A_11", "B_11", "C_11")],
        )
        self.assertEqual(len(draft.auxiliary_representatives), 20)
        self.assertEqual(len(draft.representative_constraints), 48)
        self.assertEqual(len(draft.ppt_constraints), 1)

    def test_level3_default_drops_pairwise_known_anchors(self):
        draft = build_sdp_draft(self.level3, verbose=0)
        self.assertEqual(
            [rep.target_lexorder for rep in draft.known_representatives],
            [("A_11", "B_11", "C_11")],
        )
        self.assertEqual(len(draft.auxiliary_representatives), 113)
        self.assertEqual(len(draft.representative_constraints), 693)
        self.assertEqual(len(draft.ppt_constraints), 4)


if __name__ == "__main__":
    unittest.main()
