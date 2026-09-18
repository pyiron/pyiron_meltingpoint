from ase.atoms import Atoms
from ase.constraints import FixAtoms
import numpy as np


def initialise_iterators(project_parameter: dict):
    return (
        iter(project_parameter["timestep_lst"]),
        iter(project_parameter["fit_range_lst"]),
        iter(project_parameter["nve_run_time_steps_lst"]),
    )


def freeze_one_half(basis: Atoms) -> FixAtoms:
    """
    Split the structure into two parts along the z-axis and then freeze the position of the atoms
    of the upper part (z>0.5) by attaching an ASE FixAtoms constraint to them.

    Args:
        basis (ase.atoms.Atoms): Atomistic structure object

    Returns:
        ase.constraints.FixAtoms: Constraint fixing the upper half of the structure
    """
    basis = basis.copy()
    z = basis.get_scaled_positions()[:, 2]
    return FixAtoms(indices=np.where(z >= 0.5)[0])


def round_temperature_next(temperature_next: float) -> float:
    """
    Round temperature to the last two dicits

    Args:
        temperature_next (float): Temperature

    Returns:
        float: rounded temperature
    """
    return np.round(temperature_next, 2)


def get_nve_job_name(
    temperature_next: float,
    strain: float,
    steps_lst: list[int],
    nve_run_time_steps: int,
):
    temperature_next = round_temperature_next(temperature_next)
    temp_str = str(temperature_next).replace(".", "_")
    strain_str = str(strain).replace(".", "_")
    steps_str = str(steps_lst.index(nve_run_time_steps))
    return "ham_nve_" + strain_str + "_" + temp_str + "_" + steps_str


def get_center_point(strain_result_lst=None, pressure_result_lst=None, center=None):
    if (
        strain_result_lst is not None
        and len(strain_result_lst) != 0
        and pressure_result_lst is not None
        and len(pressure_result_lst) != 0
    ):
        center_point = np.round(
            np.roots(np.polyfit(strain_result_lst, pressure_result_lst, 1))[0], 2
        )
    elif center is not None:
        center_point = center
    else:
        center_point = 1.0
    return center_point


def get_strain_lst(
    fit_range=0.02,
    points=21,
    strain_result_lst=None,
    pressure_result_lst=None,
    center=None,
):
    center_point = get_center_point(
        strain_result_lst=strain_result_lst,
        pressure_result_lst=pressure_result_lst,
        center=center,
    )
    return [
        np.round(s, 3)
        for s in np.linspace(center_point - fit_range, center_point + fit_range, points)
    ]
