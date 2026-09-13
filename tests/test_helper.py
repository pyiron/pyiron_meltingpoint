import unittest

from ase.build import bulk
from ase.constraints import FixAtoms
import numpy as np

from interfacemethod.helper import (
    freeze_one_half,
    get_center_point,
    get_nve_job_name,
    get_strain_lst,
    initialise_iterators,
    round_temperature_next,
)


class TestInitialiseIterators(unittest.TestCase):
    def test_returns_iterators_over_project_parameter_lists(self):
        project_parameter = {
            "timestep_lst": [1.0, 2.0],
            "fit_range_lst": [0.02, 0.03],
            "nve_run_time_steps_lst": [1000, 5000],
        }

        timestep_it, fit_range_it, nve_run_time_steps_it = initialise_iterators(
            project_parameter
        )

        self.assertEqual(list(timestep_it), [1.0, 2.0])
        self.assertEqual(list(fit_range_it), [0.02, 0.03])
        self.assertEqual(list(nve_run_time_steps_it), [1000, 5000])


class TestFreezeOneHalf(unittest.TestCase):
    def test_fixes_only_atoms_in_upper_half(self):
        structure = bulk("Al", cubic=True).repeat(2)
        z = structure.get_scaled_positions()[:, 2]
        expected_indices = np.where(z >= 0.5)[0]

        constraint = freeze_one_half(structure)

        self.assertIsInstance(constraint, FixAtoms)
        np.testing.assert_array_equal(constraint.index, expected_indices)

    def test_does_not_mutate_input_structure(self):
        structure = bulk("Al", cubic=True).repeat(2)

        freeze_one_half(structure)

        self.assertEqual(structure.constraints, [])


class TestRoundTemperatureNext(unittest.TestCase):
    def test_rounds_to_two_decimal_places(self):
        self.assertAlmostEqual(round_temperature_next(500.567), 500.57)

    def test_rounds_down_when_below_midpoint(self):
        self.assertAlmostEqual(round_temperature_next(500.004), 500.0)


class TestGetNveJobName(unittest.TestCase):
    def test_builds_name_from_strain_temperature_and_step_index(self):
        job_name = get_nve_job_name(
            temperature_next=500.004,
            strain=1.02,
            steps_lst=[1000, 5000],
            nve_run_time_steps=5000,
        )

        self.assertEqual(job_name, "ham_nve_1_02_500_0_1")

    def test_step_index_reflects_position_in_steps_list(self):
        job_name = get_nve_job_name(
            temperature_next=300.0,
            strain=0.98,
            steps_lst=[1000, 2000, 5000],
            nve_run_time_steps=1000,
        )

        self.assertEqual(job_name, "ham_nve_0_98_300_0_0")

    def test_raises_when_steps_not_in_list(self):
        with self.assertRaises(ValueError):
            get_nve_job_name(
                temperature_next=300.0,
                strain=1.0,
                steps_lst=[1000, 2000],
                nve_run_time_steps=9999,
            )


class TestGetCenterPoint(unittest.TestCase):
    def test_defaults_to_one_when_no_data_or_center_given(self):
        self.assertEqual(get_center_point(), 1.0)

    def test_uses_explicit_center_when_no_result_lists(self):
        self.assertEqual(get_center_point(center=1.05), 1.05)

    def test_falls_back_to_center_for_empty_result_lists(self):
        result = get_center_point(
            strain_result_lst=[], pressure_result_lst=[], center=2.0
        )

        self.assertEqual(result, 2.0)

    def test_fits_linear_trend_to_find_zero_pressure_strain(self):
        center_point = get_center_point(
            strain_result_lst=[0.98, 1.0, 1.02],
            pressure_result_lst=[10.0, 0.0, -10.0],
        )

        self.assertAlmostEqual(center_point, 1.0)


class TestGetStrainLst(unittest.TestCase):
    def test_symmetric_range_around_default_center(self):
        result = get_strain_lst(fit_range=0.02, points=5)

        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 0.98)
        self.assertAlmostEqual(result[-1], 1.02)
        self.assertAlmostEqual(result[2], 1.0)

    def test_symmetric_range_around_explicit_center(self):
        result = get_strain_lst(fit_range=0.01, points=3, center=1.05)

        self.assertEqual(result, [1.04, 1.05, 1.06])

    def test_range_around_fitted_center(self):
        result = get_strain_lst(
            fit_range=0.01,
            points=3,
            strain_result_lst=[0.98, 1.0, 1.02],
            pressure_result_lst=[10.0, 0.0, -10.0],
        )

        self.assertEqual(result, [0.99, 1.0, 1.01])


if __name__ == "__main__":
    unittest.main()
