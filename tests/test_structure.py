import unittest

from ase.build import bulk
from ase.constraints import FixAtoms

from interfacemethod.structure import (
    analyse_minimized_structure,
    analyse_structure,
    check_diamond,
    get_voronoi_volume,
    remove_selective_dynamics,
)


class TestCheckDiamond(unittest.TestCase):
    def test_fcc_structure_is_not_diamond(self):
        structure = bulk("Al", cubic=True).repeat(3)
        self.assertFalse(check_diamond(structure=structure))

    def test_diamond_structure_is_diamond(self):
        structure = bulk("Si", cubic=True).repeat(3)
        self.assertTrue(check_diamond(structure=structure))


class TestAnalyseStructure(unittest.TestCase):
    def setUp(self):
        self.fcc_structure = bulk("Al", cubic=True).repeat(3)
        self.diamond_structure = bulk("Si", cubic=True).repeat(3)

    def test_cna_total_mode_identifies_fcc(self):
        result = analyse_structure(structure=self.fcc_structure, mode="total")
        self.assertEqual(
            result["CommonNeighborAnalysis.counts.FCC"], len(self.fcc_structure)
        )
        self.assertEqual(result["CommonNeighborAnalysis.counts.OTHER"], 0)

    def test_cna_numeric_mode_returns_per_atom_array(self):
        result = analyse_structure(structure=self.fcc_structure, mode="numeric")
        self.assertEqual(len(result), len(self.fcc_structure))

    def test_cna_str_mode_returns_per_atom_labels(self):
        result = analyse_structure(structure=self.fcc_structure, mode="str")
        self.assertEqual(len(result), len(self.fcc_structure))
        self.assertTrue(all(label == "FCC" for label in result))

    def test_diamond_mode_total_identifies_diamond_structure(self):
        result = analyse_structure(
            structure=self.diamond_structure, mode="total", diamond=True
        )
        self.assertEqual(
            result["IdentifyDiamond.counts.CUBIC_DIAMOND"],
            len(self.diamond_structure),
        )
        self.assertEqual(result["IdentifyDiamond.counts.OTHER"], 0)

    def test_diamond_flag_false_uses_cna_not_diamond_detector(self):
        cna_result = analyse_structure(
            structure=self.diamond_structure, mode="total", diamond=False
        )
        self.assertIn("CommonNeighborAnalysis.counts.OTHER", cna_result)
        self.assertNotIn("IdentifyDiamond.counts.OTHER", cna_result)


class TestAnalyseMinimizedStructure(unittest.TestCase):
    def test_fcc_structure_returns_expected_summary(self):
        structure = bulk("Al", cubic=True).repeat(3)
        (
            returned_structure,
            key_max,
            number_of_atoms,
            distribution_initial_half,
            final_structure_dict,
        ) = analyse_minimized_structure(structure=structure)

        self.assertIs(returned_structure, structure)
        self.assertEqual(key_max, "CommonNeighborAnalysis.counts.FCC")
        self.assertEqual(number_of_atoms, len(structure))
        self.assertAlmostEqual(distribution_initial_half, 0.5)
        self.assertEqual(
            final_structure_dict[key_max],
            max(final_structure_dict.values()),
        )

    def test_diamond_structure_returns_expected_summary(self):
        structure = bulk("Si", cubic=True).repeat(3)
        (
            returned_structure,
            key_max,
            number_of_atoms,
            distribution_initial_half,
            final_structure_dict,
        ) = analyse_minimized_structure(structure=structure)

        self.assertIs(returned_structure, structure)
        self.assertEqual(key_max, "IdentifyDiamond.counts.CUBIC_DIAMOND")
        self.assertEqual(number_of_atoms, len(structure))
        self.assertAlmostEqual(distribution_initial_half, 0.5)


class TestGetVoronoiVolume(unittest.TestCase):
    def test_returns_max_and_mean_per_structure(self):
        structure_a = bulk("Al", cubic=True).repeat(2)
        structure_b = bulk("Al", cubic=True).repeat(3)

        max_lst, mean_lst = get_voronoi_volume([structure_a, structure_b])

        self.assertEqual(len(max_lst), 2)
        self.assertEqual(len(mean_lst), 2)
        for max_value, mean_value in zip(max_lst, mean_lst):
            self.assertAlmostEqual(max_value, mean_value, places=6)

    def test_empty_list_returns_empty_lists(self):
        max_lst, mean_lst = get_voronoi_volume([])
        self.assertEqual(max_lst, [])
        self.assertEqual(mean_lst, [])


class TestRemoveSelectiveDynamics(unittest.TestCase):
    def test_removes_fix_atoms_constraint(self):
        structure = bulk("Al", cubic=True).repeat(2)
        structure.set_constraint(FixAtoms(indices=[0, 1]))

        result = remove_selective_dynamics(basis=structure)

        self.assertEqual(result.constraints, [])

    def test_does_not_mutate_input_structure(self):
        structure = bulk("Al", cubic=True).repeat(2)
        structure.set_constraint(FixAtoms(indices=[0, 1]))

        result = remove_selective_dynamics(basis=structure)

        self.assertIsNot(result, structure)
        self.assertEqual(len(structure.constraints), 1)

    def test_structure_without_constraint_stays_unconstrained(self):
        structure = bulk("Al", cubic=True).repeat(2)

        result = remove_selective_dynamics(basis=structure)

        self.assertEqual(result.constraints, [])


if __name__ == "__main__":
    unittest.main()
