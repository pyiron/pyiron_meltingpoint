from ase.atoms import Atoms
import json
import numpy as np
import pandas

from interfacemethod.helper import get_center_point
from interfacemethod.lammps import run_npt_step
from interfacemethod.structure import analyse_structure


def bisection_step(
    number_of_atoms: int,
    key_max: str,
    structure_left: Atoms,
    structure_right: Atoms,
    potential: pandas.DataFrame,
    temperature_left: float,
    temperature_right: float,
    distribution_initial_half: float,
    structure_after_minimization: Atoms,
    run_time_steps: int,
    diamond_flag: bool,
    seed: int,
    project_path: str,
    lmp_command: str = "lmp -in lmp.in",
):
    """
    Perform one bisection step in the melting temperature search.

    Analyses the structural order of the left and right bracket structures and
    adjusts the temperature bracket accordingly:

    - Both solid -> shift the bracket upward.
    - Left solid, right liquid -> bisect downward.
    - Both liquid -> shift the bracket downward.

    Args:
        number_of_atoms (int): Total number of atoms in the simulation cell.
        key_max (str): OVITO key of the dominant structural motif in the solid phase.
        structure_left (Atoms): Structure equilibrated at ``temperature_left``.
        structure_right (Atoms): Structure equilibrated at ``temperature_right``.
        potential (pd.DataFrame): Interatomic potential DataFrame.
        temperature_left (float): Lower bound of the current temperature bracket in K.
        temperature_right (float): Upper bound of the current temperature bracket in K.
        distribution_initial_half (float): Half of the initial solid-phase atom fraction
            (threshold for solid/liquid classification).
        structure_after_minimization (Atoms): The energy-minimised reference structure used
            to seed new MD runs.
        run_time_steps (int): Number of NPT MD timesteps per bracket point.
        diamond_flag (bool): Whether to use the diamond structure detector instead of CNA.
        seed (int): Random seed for MD velocity initialisation.

    Returns:
        tuple[Atoms, Atoms, float, float]: Updated
            ``(structure_left, structure_right, temperature_left, temperature_right)``.

    Raises:
        ValueError: If none of the three expected cases is satisfied (should never happen).
    """
    structure_left_dict = analyse_structure(
        structure=structure_left,
        mode="total",
        diamond=diamond_flag,
    )
    structure_right_dict = analyse_structure(
        structure=structure_right,
        mode="total",
        diamond=diamond_flag,
    )
    temperature_diff = temperature_right - temperature_left
    if (
        structure_left_dict[key_max] / number_of_atoms > distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms > distribution_initial_half
    ):
        structure_left = structure_right.copy()
        temperature_left = temperature_right
        temperature_right += temperature_diff
        structure_right = run_npt_step(
            structure=structure_after_minimization,
            temperature=temperature_right,
            potential=potential,
            seed=seed,
            run_time_steps=run_time_steps,
            project_path=project_path,
            lmp_command=lmp_command,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms
        > distribution_initial_half
        > structure_right_dict[key_max] / number_of_atoms
    ):
        temperature_diff /= 2
        temperature_left += temperature_diff
        structure_left = run_npt_step(
            structure=structure_after_minimization,
            temperature=temperature_left,
            potential=potential,
            seed=seed,
            run_time_steps=run_time_steps,
            project_path=project_path,
            lmp_command=lmp_command,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms < distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms < distribution_initial_half
    ):
        temperature_diff /= 2
        temperature_right = temperature_left
        temperature_left -= temperature_diff
        structure_right = structure_left.copy()
        structure_left = run_npt_step(
            structure=structure_after_minimization,
            temperature=temperature_left,
            potential=potential,
            seed=seed,
            run_time_steps=run_time_steps,
            project_path=project_path,
        )
    else:
        raise ValueError("We should never reach this point!")
    return structure_left, structure_right, temperature_left, temperature_right


def validate_convergence(
    temperature_left: float,
    temperature_next: float,
    temperature_right: float,
    enable_iteration: bool,
    timestep_iter,
    timestep_lst: list[float],
    timestep: float,
    fit_range_iter,
    fit_range_lst: list[float],
    fit_range: float,
    nve_run_time_steps_iter,
    nve_run_time_steps_lst: list[int],
    nve_run_time_steps: int,
    strain_result_lst: list[float],
    pressure_result_lst: list[float],
    step_count: int,
    step_dict: dict,
    boundary_value: float,
    ratio_boundary: float,
    convergence_goal: float,
    output_file: str = "melting.json",
):
    if temperature_left < temperature_next < temperature_right and enable_iteration:
        timestep = next(timestep_iter)
        fit_range = next(fit_range_iter)
        nve_run_time_steps = next(nve_run_time_steps_iter)
    if (
        timestep == timestep_lst[-1]
        and fit_range == fit_range_lst[-1]
        and nve_run_time_steps == nve_run_time_steps_lst[-1]
    ):
        enable_iteration = False
    center = np.abs(
        get_center_point(
            strain_result_lst=strain_result_lst, pressure_result_lst=pressure_result_lst
        )
    )
    step_count += 1
    if step_count not in step_dict.keys():
        step_dict[step_count] = {
            "timestep": timestep,
            "fit_range": fit_range,
            "nve_run_time_steps": nve_run_time_steps,
            "boundary_value": boundary_value,
            "ratio_boundary": ratio_boundary,
            "temperature_next": temperature_next,
            "center": center,
        }
        with open(output_file, "w") as f:
            json.dump(step_dict, f)
    else:
        timestep = step_dict[step_count]["timestep"]
        fit_range = step_dict[step_count]["fit_range"]
        nve_run_time_steps = step_dict[step_count]["nve_run_time_steps"]
        boundary_value = step_dict[step_count]["boundary_value"]
        ratio_boundary = step_dict[step_count]["ratio_boundary"]
        temperature_next = step_dict[step_count]["temperature_next"]
        center = step_dict[step_count]["center"]
    if (
        np.abs(
            step_dict[step_count]["temperature_next"]
            - step_dict[step_count - 1]["temperature_next"]
        )
        <= convergence_goal
    ):
        convergence_goal_achieved = True
    else:
        convergence_goal_achieved = False
    return (
        convergence_goal_achieved,
        enable_iteration,
        step_count,
        step_dict,
        timestep,
        fit_range,
        nve_run_time_steps,
        boundary_value,
        ratio_boundary,
        temperature_next,
        center,
    )
