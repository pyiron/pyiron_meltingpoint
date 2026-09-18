import os
import unittest
from unittest.mock import patch

from ase.build import bulk
import numpy as np

from interfacemethod.lammps import (
    get_press,
    minimize_structure_positions,
    minimize_structure_volume,
    npt_liquid,
    npt_solid,
    run_npt_step,
    run_strain_point,
    setup_liquid_job,
    structure_from_parsed_output,
)


def make_parsed_output(structure, n_steps=2, seed=0):
    rng = np.random.default_rng(seed)
    n_atoms = len(structure)
    positions = np.stack(
        [structure.get_positions() + step * 0.01 for step in range(n_steps)]
    )
    velocities = rng.random((n_steps, n_atoms, 3))
    indices = np.tile(np.arange(n_atoms), (n_steps, 1))
    cells = np.stack([structure.cell.array for _ in range(n_steps)])
    pressures = rng.random((n_steps, 3, 3))
    temperature = rng.random(n_steps) * 300
    return {
        "generic": {
            "indices": indices,
            "cells": cells,
            "positions": positions,
            "velocities": velocities,
            "pressures": pressures,
            "temperature": temperature,
        }
    }


class TestStructureFromParsedOutput(unittest.TestCase):
    def setUp(self):
        self.structure = bulk("Al", cubic=True).repeat(2)
        self.parsed_output = make_parsed_output(self.structure)

    def test_uses_last_snapshot_and_returns_copy(self):
        result = structure_from_parsed_output(self.structure, self.parsed_output)

        self.assertIsNot(result, self.structure)
        np.testing.assert_allclose(
            result.get_positions(), self.parsed_output["generic"]["positions"][-1]
        )
        np.testing.assert_allclose(
            result.get_velocities(), self.parsed_output["generic"]["velocities"][-1]
        )
        np.testing.assert_allclose(
            result.cell.array, self.parsed_output["generic"]["cells"][-1]
        )
        np.testing.assert_array_equal(
            result.get_array("indices"), self.parsed_output["generic"]["indices"][-1]
        )
        self.assertTrue(all(result.pbc))

    def test_wrap_false_by_default_leaves_positions_unwrapped(self):
        parsed_output = make_parsed_output(self.structure)
        parsed_output["generic"]["positions"][-1][0] = [100.0, 0.0, 0.0]

        result_unwrapped = structure_from_parsed_output(
            self.structure, parsed_output, wrap=False
        )
        result_wrapped = structure_from_parsed_output(
            self.structure, parsed_output, wrap=True
        )

        self.assertFalse(
            np.allclose(
                result_unwrapped.get_positions()[0], result_wrapped.get_positions()[0]
            )
        )
        self.assertTrue(np.all(result_wrapped.get_positions()[0] < 100.0))


class TestGetPress(unittest.TestCase):
    def test_averages_diagonal_of_pressure_tensor(self):
        pressures = np.arange(5 * 3 * 3).reshape(5, 3, 3).astype(float)
        parsed_output = {"generic": {"pressures": pressures}}

        result = get_press(parsed_output=parsed_output, step=2)

        expected = np.array(
            [
                np.mean(np.diag(pressures[2])),
                np.mean(np.diag(pressures[3])),
                np.mean(np.diag(pressures[4])),
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_default_step_uses_last_twenty_entries(self):
        pressures = np.random.default_rng(1).random((25, 3, 3))
        parsed_output = {"generic": {"pressures": pressures}}

        result = get_press(parsed_output=parsed_output)

        self.assertEqual(len(result), 5)


class TestMinimizeStructurePositions(unittest.TestCase):
    @patch("interfacemethod.lammps.lammps_file_interface_function")
    def test_calls_lammps_with_minimize_kwargs(self, mock_lammps):
        structure = bulk("Al", cubic=True).repeat(2)
        parsed_output = make_parsed_output(structure)
        mock_lammps.return_value = (None, parsed_output, None)
        potential = object()

        result = minimize_structure_positions(
            structure=structure,
            potential=potential,
            project_path="/tmp/proj",
            max_iter=500,
        )

        mock_lammps.assert_called_once_with(
            working_directory=os.path.join("/tmp/proj", "minimize_pos"),
            structure=structure,
            potential=potential,
            calc_mode="minimize",
            calc_kwargs={
                "max_iter": 500,
                "ionic_energy_tolerance": 1.0e-9,
                "ionic_force_tolerance": 1.0e-8,
                "n_print": 500,
            },
            lmp_command="lmp -in lmp.in",
        )
        np.testing.assert_allclose(
            result.get_positions(), parsed_output["generic"]["positions"][-1]
        )


class TestMinimizeStructureVolume(unittest.TestCase):
    @patch("interfacemethod.lammps.lammps_file_interface_function")
    def test_calls_lammps_with_box_relax_fix(self, mock_lammps):
        structure = bulk("Al", cubic=True).repeat(2)
        parsed_output = make_parsed_output(structure)
        mock_lammps.return_value = (None, parsed_output, None)
        potential = object()

        minimize_structure_volume(
            structure=structure,
            potential=potential,
            project_path="/tmp/proj",
        )

        _, kwargs = mock_lammps.call_args
        self.assertEqual(
            kwargs["working_directory"], os.path.join("/tmp/proj", "minimize_vol")
        )
        self.assertEqual(kwargs["calc_kwargs"]["pressure"], 0.0)
        self.assertEqual(
            kwargs["input_control_file"],
            {"fix": "ensemble all box/relax iso 0.0 vmax 0.001"},
        )


class TestRunNptStep(unittest.TestCase):
    @patch("interfacemethod.lammps.lammps_file_interface_function")
    def test_builds_npt_fix_and_working_directory(self, mock_lammps):
        structure = bulk("Al", cubic=True).repeat(2)
        parsed_output = make_parsed_output(structure)
        mock_lammps.return_value = (None, parsed_output, None)
        potential = object()

        run_npt_step(
            structure=structure,
            potential=potential,
            temperature=500.0,
            seed=42,
            project_path="/tmp/proj",
            run_time_steps=2000,
        )

        _, kwargs = mock_lammps.call_args
        self.assertEqual(
            kwargs["working_directory"],
            os.path.join("/tmp/proj", "temp_heating", "500_0"),
        )
        self.assertEqual(kwargs["calc_kwargs"]["temperature"], 500.0)
        self.assertEqual(kwargs["calc_kwargs"]["seed"], 42)
        self.assertEqual(kwargs["calc_kwargs"]["n_ionic_steps"], 2000)
        self.assertEqual(
            kwargs["input_control_file"],
            {
                "fix": "ensemble all npt temp 500.0 500.0 0.1 iso 0.0 0.0 1.0 couple xyz"
            },
        )


class TestNptSolid(unittest.TestCase):
    @patch("interfacemethod.lammps.lammps_file_interface_function")
    def test_uses_project_parameter_values(self, mock_lammps):
        structure = bulk("Al", cubic=True).repeat(2)
        parsed_output = make_parsed_output(structure)
        mock_lammps.return_value = (None, parsed_output, None)
        potential = object()
        project_parameter = {
            "potential": potential,
            "run_time_steps": 3000,
            "seed": 7,
        }

        npt_solid(
            temperature=300.0,
            basis=structure,
            project_parameter=project_parameter,
            project_path="/tmp/proj",
        )

        _, kwargs = mock_lammps.call_args
        self.assertEqual(
            kwargs["working_directory"], os.path.join("/tmp/proj", "npt_solid", "300_0")
        )
        self.assertIs(kwargs["potential"], potential)
        self.assertEqual(kwargs["calc_kwargs"]["n_ionic_steps"], 3000)
        self.assertEqual(kwargs["calc_kwargs"]["seed"], 7)


class TestSetupLiquidJob(unittest.TestCase):
    @patch("interfacemethod.lammps.lammps_file_interface_function")
    def test_couples_only_z_pressure(self, mock_lammps):
        structure = bulk("Al", cubic=True).repeat(2)
        parsed_output = make_parsed_output(structure)
        mock_lammps.return_value = (None, parsed_output, None)
        project_parameter = {
            "potential": object(),
            "run_time_steps": 1500,
            "seed": 3,
        }

        setup_liquid_job(
            job_name="high_300_0",
            basis=structure,
            temperature=300.0,
            project_parameter=project_parameter,
            project_path="/tmp/proj",
        )

        _, kwargs = mock_lammps.call_args
        self.assertEqual(
            kwargs["working_directory"], os.path.join("/tmp/proj", "liquid", "high_300_0")
        )
        self.assertEqual(kwargs["calc_kwargs"]["pressure"], [None, None, 0.0])
        self.assertNotIn("input_control_file", kwargs)


class TestNptLiquid(unittest.TestCase):
    @patch("interfacemethod.lammps.setup_liquid_job")
    def test_chains_high_then_low_temperature_jobs(self, mock_setup_liquid_job):
        structure = bulk("Al", cubic=True).repeat(2)
        intermediate_structure = structure.copy()
        final_structure = structure.copy()
        mock_setup_liquid_job.side_effect = [intermediate_structure, final_structure]
        project_parameter = {"potential": object(), "run_time_steps": 1000, "seed": 1}

        result = npt_liquid(
            temperature_solid=800.0,
            temperature_liquid=1200.0,
            basis=structure,
            project_parameter=project_parameter,
            project_path="/tmp/proj",
        )

        self.assertEqual(mock_setup_liquid_job.call_count, 2)
        first_call_kwargs = mock_setup_liquid_job.call_args_list[0].kwargs
        second_call_kwargs = mock_setup_liquid_job.call_args_list[1].kwargs

        self.assertEqual(first_call_kwargs["job_name"], "high_1200_0")
        self.assertEqual(first_call_kwargs["temperature"], 1200.0)
        self.assertIs(first_call_kwargs["basis"], structure)

        self.assertEqual(second_call_kwargs["job_name"], "low_800_0")
        self.assertEqual(second_call_kwargs["temperature"], 800.0)
        self.assertIs(second_call_kwargs["basis"], intermediate_structure)

        self.assertIs(result, final_structure)
        self.assertEqual(len(structure.constraints), 1)
        self.assertEqual(len(intermediate_structure.constraints), 1)


class TestRunStrainPoint(unittest.TestCase):
    @patch("interfacemethod.lammps.lammps_file_interface_function")
    def test_applies_strain_and_returns_strain_point_result(self, mock_lammps):
        basis_relative = bulk("Al", cubic=True).repeat(2)
        original_cell = basis_relative.cell.array.copy()
        nve_parsed_output = make_parsed_output(basis_relative)
        mock_lammps.side_effect = [
            (None, {}, None),
            (None, nve_parsed_output, None),
        ]
        project_parameter = {
            "potential": object(),
            "seed": 5,
            "nvt_run_time_steps": 4000,
            "nve_run_time_steps_lst": [1000, 5000],
        }

        result = run_strain_point(
            strain=1.02,
            basis_relative=basis_relative,
            temperature_next=500.004,
            nve_run_time_steps=5000,
            project_parameter=project_parameter,
            project_path="/tmp/proj",
        )

        self.assertEqual(mock_lammps.call_count, 2)
        nvt_kwargs = mock_lammps.call_args_list[0].kwargs
        nve_kwargs = mock_lammps.call_args_list[1].kwargs

        strained_structure = nvt_kwargs["structure"]
        expected_cell = original_cell.copy()
        expected_cell[2, 2] *= 1.02
        np.testing.assert_allclose(strained_structure.cell.array, expected_cell)

        self.assertEqual(nvt_kwargs["calc_kwargs"]["temperature"], 500.0)
        self.assertTrue(nvt_kwargs["write_restart_file"])
        self.assertIn("nvt", nvt_kwargs["working_directory"])

        self.assertTrue(nve_kwargs["read_restart_file"])
        self.assertEqual(
            nve_kwargs["restart_file"],
            os.path.join(nvt_kwargs["working_directory"], "restart.out"),
        )
        self.assertTrue(nve_kwargs["dump_final_structure"])
        self.assertIn("nve", nve_kwargs["working_directory"])

        self.assertEqual(result.strain, 1.02)
        expected_pressure = np.mean(
            get_press(parsed_output=nve_parsed_output, step=-20)
        )
        self.assertAlmostEqual(result.pressure, expected_pressure)
        self.assertAlmostEqual(
            result.temperature,
            np.mean(nve_parsed_output["generic"]["temperature"][-20:]),
        )
        np.testing.assert_allclose(
            result.structure.get_positions(),
            nve_parsed_output["generic"]["positions"][-1],
        )


if __name__ == "__main__":
    unittest.main()
