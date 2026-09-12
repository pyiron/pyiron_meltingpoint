import json
import os
import tempfile
import unittest
from unittest.mock import patch

from ase.build import bulk

from interfacemethod.workflow import bisection_step, validate_convergence


class TestBisectionStep(unittest.TestCase):
    def setUp(self):
        self.structure_left = bulk("Al", cubic=True).repeat(2)
        self.structure_right = bulk("Al", cubic=True).repeat(2)
        self.structure_after_minimization = bulk("Al", cubic=True).repeat(2)
        self.potential = object()
        self.common_kwargs = dict(
            number_of_atoms=100,
            key_max="key",
            structure_left=self.structure_left,
            structure_right=self.structure_right,
            potential=self.potential,
            temperature_left=900.0,
            temperature_right=1000.0,
            distribution_initial_half=0.5,
            structure_after_minimization=self.structure_after_minimization,
            run_time_steps=1000,
            diamond_flag=False,
            seed=1,
            project_path="/tmp/proj",
        )

    @patch("interfacemethod.workflow.run_npt_step")
    @patch("interfacemethod.workflow.analyse_structure")
    def test_both_solid_shifts_bracket_upward(self, mock_analyse, mock_run):
        mock_analyse.side_effect = [{"key": 60}, {"key": 70}]
        sentinel = object()
        mock_run.return_value = sentinel

        new_left, new_right, temp_left, temp_right = bisection_step(
            **self.common_kwargs
        )

        self.assertEqual(temp_left, 1000.0)
        self.assertEqual(temp_right, 1100.0)
        self.assertIsNot(new_left, self.structure_right)
        self.assertIs(new_right, sentinel)
        mock_run.assert_called_once_with(
            structure=self.structure_after_minimization,
            temperature=1100.0,
            potential=self.potential,
            seed=1,
            run_time_steps=1000,
            project_path="/tmp/proj",
            lmp_command="lmp -in lmp.in",
        )

    @patch("interfacemethod.workflow.run_npt_step")
    @patch("interfacemethod.workflow.analyse_structure")
    def test_left_solid_right_liquid_bisects_downward(self, mock_analyse, mock_run):
        mock_analyse.side_effect = [{"key": 70}, {"key": 30}]
        sentinel = object()
        mock_run.return_value = sentinel

        new_left, new_right, temp_left, temp_right = bisection_step(
            **{**self.common_kwargs, "lmp_command": "custom"}
        )

        self.assertEqual(temp_left, 950.0)
        self.assertEqual(temp_right, 1000.0)
        self.assertIs(new_left, sentinel)
        self.assertIs(new_right, self.structure_right)
        mock_run.assert_called_once_with(
            structure=self.structure_after_minimization,
            temperature=950.0,
            potential=self.potential,
            seed=1,
            run_time_steps=1000,
            project_path="/tmp/proj",
            lmp_command="custom",
        )

    @patch("interfacemethod.workflow.run_npt_step")
    @patch("interfacemethod.workflow.analyse_structure")
    def test_both_liquid_shifts_bracket_downward(self, mock_analyse, mock_run):
        mock_analyse.side_effect = [{"key": 20}, {"key": 10}]
        sentinel = object()
        mock_run.return_value = sentinel

        new_left, new_right, temp_left, temp_right = bisection_step(
            **self.common_kwargs
        )

        self.assertEqual(temp_left, 850.0)
        self.assertEqual(temp_right, 900.0)
        self.assertIs(new_left, sentinel)
        self.assertIsNot(new_right, self.structure_left)
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["temperature"], 850.0)
        self.assertNotIn("lmp_command", kwargs)

    @patch("interfacemethod.workflow.run_npt_step")
    @patch("interfacemethod.workflow.analyse_structure")
    def test_left_liquid_right_solid_raises_value_error(self, mock_analyse, mock_run):
        mock_analyse.side_effect = [{"key": 30}, {"key": 70}]

        with self.assertRaises(ValueError):
            bisection_step(**self.common_kwargs)

        mock_run.assert_not_called()


class TestValidateConvergence(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.output_file = os.path.join(self.tmp_dir.name, "melting.json")
        self.common_kwargs = dict(
            temperature_left=900.0,
            temperature_next=950.0,
            temperature_right=1000.0,
            strain_result_lst=[0.98, 1.0, 1.02],
            pressure_result_lst=[10.0, 0.0, -10.0],
            boundary_value=0.2,
            ratio_boundary=0.1,
            output_file=self.output_file,
        )

    def test_new_step_advances_iterators_and_writes_output_file(self):
        step_dict = {0: {"temperature_next": 1000.0}}

        result = validate_convergence(
            enable_iteration=True,
            timestep_iter=iter([2.0]),
            timestep_lst=[1.0, 2.0],
            timestep=1.0,
            fit_range_iter=iter([0.01]),
            fit_range_lst=[0.02, 0.01],
            fit_range=0.02,
            nve_run_time_steps_iter=iter([5000]),
            nve_run_time_steps_lst=[1000, 5000],
            nve_run_time_steps=1000,
            step_count=0,
            step_dict=step_dict,
            convergence_goal=100.0,
            **self.common_kwargs,
        )
        (
            convergence_goal_achieved,
            enable_iteration,
            step_count,
            returned_step_dict,
            timestep,
            fit_range,
            nve_run_time_steps,
            boundary_value,
            ratio_boundary,
            temperature_next,
            center,
        ) = result

        self.assertTrue(convergence_goal_achieved)
        self.assertFalse(enable_iteration)
        self.assertEqual(step_count, 1)
        self.assertEqual(timestep, 2.0)
        self.assertEqual(fit_range, 0.01)
        self.assertEqual(nve_run_time_steps, 5000)
        self.assertEqual(boundary_value, 0.2)
        self.assertEqual(ratio_boundary, 0.1)
        self.assertEqual(temperature_next, 950.0)
        self.assertAlmostEqual(center, 1.0)
        self.assertIn(1, returned_step_dict)

        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file) as f:
            written = json.load(f)
        self.assertEqual(written["1"]["temperature_next"], 950.0)

    def test_convergence_not_achieved_when_temperature_change_exceeds_goal(self):
        step_dict = {0: {"temperature_next": 1000.0}}

        result = validate_convergence(
            enable_iteration=False,
            timestep_iter=iter([]),
            timestep_lst=[1.0, 2.0],
            timestep=2.0,
            fit_range_iter=iter([]),
            fit_range_lst=[0.02, 0.01],
            fit_range=0.01,
            nve_run_time_steps_iter=iter([]),
            nve_run_time_steps_lst=[1000, 5000],
            nve_run_time_steps=5000,
            step_count=0,
            step_dict=step_dict,
            convergence_goal=10.0,
            **self.common_kwargs,
        )

        self.assertFalse(result[0])

    def test_resuming_existing_step_uses_stored_values_and_skips_file_write(self):
        step_dict = {
            0: {"temperature_next": 1000.0},
            1: {
                "timestep": 9.9,
                "fit_range": 0.005,
                "nve_run_time_steps": 42,
                "boundary_value": 0.11,
                "ratio_boundary": 0.22,
                "temperature_next": 955.0,
                "center": 3.3,
            },
        }

        result = validate_convergence(
            enable_iteration=True,
            timestep_iter=iter([2.0]),
            timestep_lst=[1.0, 2.0],
            timestep=1.0,
            fit_range_iter=iter([0.01]),
            fit_range_lst=[0.02, 0.01],
            fit_range=0.02,
            nve_run_time_steps_iter=iter([5000]),
            nve_run_time_steps_lst=[1000, 5000],
            nve_run_time_steps=1000,
            step_count=0,
            step_dict=step_dict,
            convergence_goal=50.0,
            **self.common_kwargs,
        )

        self.assertEqual(
            result[4:],
            (9.9, 0.005, 42, 0.11, 0.22, 955.0, 3.3),
        )
        self.assertFalse(os.path.exists(self.output_file))

    def test_iterators_not_advanced_when_temperature_next_outside_bracket(self):
        step_dict = {0: {"temperature_next": 1000.0}}

        result = validate_convergence(
            temperature_left=900.0,
            temperature_next=1200.0,
            temperature_right=1000.0,
            strain_result_lst=[0.98, 1.0, 1.02],
            pressure_result_lst=[10.0, 0.0, -10.0],
            boundary_value=0.2,
            ratio_boundary=0.1,
            output_file=self.output_file,
            enable_iteration=True,
            timestep_iter=iter([]),
            timestep_lst=[1.0, 2.0],
            timestep=1.5,
            fit_range_iter=iter([]),
            fit_range_lst=[0.02, 0.01],
            fit_range=0.015,
            nve_run_time_steps_iter=iter([]),
            nve_run_time_steps_lst=[1000, 5000],
            nve_run_time_steps=3000,
            step_count=0,
            step_dict=step_dict,
            convergence_goal=100.0,
        )

        self.assertEqual(result[4:7], (1.5, 0.015, 3000))
        self.assertTrue(result[1])


if __name__ == "__main__":
    unittest.main()
