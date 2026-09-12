import os
from ase.atoms import Atoms
from lammpsparser import lammps_file_interface_function
import numpy as np
import pandas

from interfacemethod.helper import (
    freeze_one_half,
    round_temperature_next,
    get_nve_job_name,
)
from interfacemethod.result import StrainPointResult


def structure_from_parsed_output(
    initial_structure: Atoms, parsed_output: dict, *, wrap: bool = False
) -> Atoms:
    """Construct an `Atoms` object from parsed output data.

    Args:
        initial_structure: The initial atomic structure to use as a template.
        parsed_output: Parsed output containing atomic positions, cell, and indices.
        wrap: Whether to wrap the atomic positions to the simulation cell (default is False).
            Keeping the unwrapped positions is more beneficial if structures are passed between
            different LAMMPS simulations in one workflow to ensure continuity.

    Returns:
        An `Atoms` object with updated positions and cell.

    Example:
        >>> new_atoms = structure_from_parsed_output(atoms, lammps_output)

    """
    atoms_copy = initial_structure.copy()
    atoms_copy.set_array("indices", parsed_output["generic"]["indices"][-1])
    atoms_copy.set_cell(parsed_output["generic"]["cells"][-1], scale_atoms=True)
    atoms_copy.set_positions(parsed_output["generic"]["positions"][-1])
    atoms_copy.set_velocities(parsed_output["generic"]["velocities"][-1])
    atoms_copy.set_pbc(True)
    if wrap:
        atoms_copy.wrap()

    return atoms_copy


def minimize_structure_positions(
    structure: Atoms,
    potential: pandas.DataFrame,
    project_path: str,
    max_iter: int = 1000,
    lmp_command: str = "lmp -in lmp.in",
) -> Atoms:
    """Relax the atomic positions of a structure at fixed cell shape."""
    _, parsed_output, _ = lammps_file_interface_function(
        working_directory=os.path.join(project_path, "minimize_pos"),
        structure=structure,
        potential=potential,
        calc_mode="minimize",
        calc_kwargs={
            "max_iter": max_iter,
            "ionic_energy_tolerance": 1.0e-9,
            "ionic_force_tolerance": 1.0e-8,
            "n_print": max_iter,
        },
        lmp_command=lmp_command,
    )
    return structure_from_parsed_output(structure, parsed_output, wrap=True)


def minimize_structure_volume(
    structure: Atoms,
    potential: pandas.DataFrame,
    project_path: str,
    max_iter: int = 1000,
    lmp_command: str = "lmp -in lmp.in",
) -> Atoms:
    """Relax both atomic positions and cell volume of a structure at zero pressure."""
    _, parsed_output, _ = lammps_file_interface_function(
        working_directory=os.path.join(project_path, "minimize_vol"),
        structure=structure,
        potential=potential,
        calc_mode="minimize",
        calc_kwargs={
            "max_iter": max_iter,
            "ionic_energy_tolerance": 1.0e-9,
            "ionic_force_tolerance": 1.0e-8,
            "n_print": max_iter,
            "pressure": 0.0,
        },
        input_control_file={"fix": "ensemble all box/relax iso 0.0 vmax 0.001"},
        lmp_command=lmp_command,
    )
    return structure_from_parsed_output(structure, parsed_output, wrap=True)


def run_npt_step(
    structure: Atoms,
    potential: pandas.DataFrame,
    temperature: float,
    seed: int,
    project_path: str,
    run_time_steps: int = 10000,
    lmp_command: str = "lmp -in lmp.in",
):
    """
    Calculate NPT ensemble at a given temperature using the job defined in the project parameters:
    - job_type: Type of Simulation code to be used
    - project: Project object used to create the job
    - potential: Interatomic Potential
    - queue (optional): HPC Job queue to be used

    Args:
        structure (ase.atoms.Atoms): Atomistic Structure object to be set to the job as input sturcture
        temperature (float): Temperature of the Molecular dynamics calculation
        run_time_steps (int): Number of Molecular dynamics steps

    Returns:
        Final Atomistic Structure object
    """
    _, parsed_output, _ = lammps_file_interface_function(
        working_directory=os.path.join(
            project_path, "temp_heating", str(temperature).replace(".", "_")
        ),
        structure=structure,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature,
            "initial_temperature": temperature,
            "temperature_damping_timescale": 100.0,
            "pressure": 0.0,
            "pressure_damping_timescale": 1000.0,
            "n_print": run_time_steps,
            "n_ionic_steps": run_time_steps,
            "seed": seed,
        },
        input_control_file={
            "fix": f"ensemble all npt temp {temperature} {temperature} 0.1 iso 0.0 0.0 1.0 couple xyz"
        },
        lmp_command=lmp_command,
    )
    return structure_from_parsed_output(structure, parsed_output, wrap=True)


def npt_solid(
    temperature: float,
    basis: Atoms,
    project_parameter: str,
    project_path: str,
    timestep: float = 1.0,
    lmp_command: str = "lmp -in lmp.in",
) -> Atoms:
    """
    Calculate NPT ensemble at a given temperature using lammps_file_interface_function.

    Args:
        temperature (float): Temperature of the Molecular dynamics calculation
        basis (ase.atoms.Atoms): Atomistic Structure object to be used as input structure
        project_parameter (dict): Dictionary with the project parameters
        project_path (str): Working directory the calculation is executed in
        timestep (float): Molecular dynamics time step

    Returns:
        Atoms: Final Atomistic Structure object
    """
    _, parsed_output, _ = lammps_file_interface_function(
        working_directory=os.path.join(
            project_path, "npt_solid", str(temperature).replace(".", "_")
        ),
        structure=basis,
        potential=project_parameter["potential"],
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature,
            "initial_temperature": temperature,
            "temperature_damping_timescale": 100.0,
            "time_step": timestep,
            "pressure": 0.0,
            "pressure_damping_timescale": 1000.0,
            "n_print": project_parameter["run_time_steps"],
            "n_ionic_steps": project_parameter["run_time_steps"],
            "seed": project_parameter["seed"],
        },
        input_control_file={
            "fix": f"ensemble all npt temp {temperature} {temperature} 0.1 iso 0.0 0.0 1.0 couple xyz"
        },
        lmp_command=lmp_command,
    )
    return structure_from_parsed_output(basis, parsed_output, wrap=True)


def setup_liquid_job(
    job_name: str,
    basis: Atoms,
    temperature: float,
    project_parameter: dict,
    project_path: str,
    timestep: float = 1.0,
    lmp_command: str = "lmp -in lmp.in",
):
    """
    Calculate NPT ensemble at a given temperature while freezing the position of the atoms
    of the upper part (z>0.5) using lammps_file_interface_function. Only the z-component of
    the pressure is coupled to a barostat, matching the previous fix_z_dir behaviour.

    Args:
        job_name (str): Name used for the working directory of the calculation
        basis (ase.atoms.Atoms): Atomistic Structure object to be used as input structure
        temperature (float): Temperature of the Molecular dynamics calculation
        project_parameter (dict): Dictionary with the project parameters
        project_path (str): Working directory the calculation is executed in
        timestep (float): Molecular dynamics time step

    Returns:
        Atoms: Final Atomistic Structure object
    """
    _, parsed_output, _ = lammps_file_interface_function(
        working_directory=os.path.join(project_path, "liquid", job_name),
        structure=basis,
        potential=project_parameter["potential"],
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature,
            "initial_temperature": temperature,
            "temperature_damping_timescale": 100.0,
            "time_step": timestep,
            "pressure": [None, None, 0.0],
            "pressure_damping_timescale": 1000.0,
            "n_print": project_parameter["run_time_steps"],
            "n_ionic_steps": project_parameter["run_time_steps"],
            "seed": project_parameter["seed"],
        },
        lmp_command=lmp_command,
    )
    return structure_from_parsed_output(basis, parsed_output, wrap=True)


def npt_liquid(
    temperature_solid: float,
    temperature_liquid: float,
    basis: Atoms,
    project_parameter: dict,
    project_path: str,
    lmp_command: str = "lmp -in lmp.in",
    timestep: float = 1.0,
):
    """
    Calculate NPT ensemble at a given temperature while initially freezing the position of the atoms
    of the upper part (z>0.5) and afterwards calculating the full sample at a lower temperature.
    These steps are used to construct the solid liquid interface as part of the coexistence approach.

    Args:
        temperature_solid (float): Temperature to simulate the whole structure
        temperature_liquid (float): Temperature to simulate the upper half of the structure
        basis (ase.atoms.Atoms): Atomistic Structure object to be used as input structure
        project_parameter (dict): Dictionary with the project parameters
        project_path (str): Working directory the calculation is executed in
        timestep (float): Molecular dynamics time step

    Returns:
        Atoms: Final Atomistic Structure object
    """
    constraint = freeze_one_half(basis)
    basis.set_constraint(constraint)
    structure_liquid_high = setup_liquid_job(
        job_name="high_" + str(temperature_liquid).replace(".", "_"),
        basis=basis,
        temperature=temperature_liquid,
        project_parameter=project_parameter,
        project_path=project_path,
        timestep=timestep,
        lmp_command=lmp_command,
    )
    structure_liquid_high.set_constraint(constraint)
    structure_liquid_low = setup_liquid_job(
        job_name="low_" + str(temperature_solid).replace(".", "_"),
        basis=structure_liquid_high,
        temperature=temperature_solid,
        project_parameter=project_parameter,
        project_path=project_path,
        timestep=timestep,
        lmp_command=lmp_command,
    )
    return structure_liquid_low


def get_press(parsed_output, step: int = 20):
    """
    Args:
        parsed_output (dict): Output parsed from a LAMMPS MD calculation via
            lammps_file_interface_function
        step (int): Number of steps counted from the end of the trajectory to average over
    """
    return np.mean(
        parsed_output["generic"]["pressures"][step:, :, :].diagonal(0, 2), axis=1
    )


def run_strain_point(
    strain: float,
    basis_relative: Atoms,
    temperature_next: float,
    nve_run_time_steps: int,
    project_parameter: dict,
    project_path: str,
    timestep: float = 1.0,
    lmp_command: str = "lmp -in lmp.in",
) -> StrainPointResult:
    """
    Apply one strain to the interface structure along z and measure the resulting pressure and
    temperature via a short NVT equilibration followed by an NVE production run.

    Every strain point is independent of every other, so this is the unit of work for the strain
    scan: a `for strain in strain_lst: run_strain_point(strain, ...)` loop can be replaced with
    `executor.map(...)` on a `concurrent.futures.ProcessPoolExecutor` or an `executorlib.Executor`
    to run the scan in parallel.
    """
    temperature_next = round_temperature_next(temperature_next)
    job_name = get_nve_job_name(
        temperature_next=temperature_next,
        strain=strain,
        steps_lst=project_parameter["nve_run_time_steps_lst"],
        nve_run_time_steps=nve_run_time_steps,
    )
    nvt_working_directory = os.path.join(
        project_path, "strain_circle", job_name.replace("nve", "nvt")
    )
    nve_working_directory = os.path.join(project_path, "strain_circle", job_name)
    basis_strain = basis_relative.copy()
    cell = basis_strain.cell.copy()
    cell[2, 2] *= strain
    basis_strain.set_cell(cell=cell, scale_atoms=True)
    lammps_file_interface_function(
        working_directory=nvt_working_directory,
        structure=basis_strain,
        potential=project_parameter["potential"],
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature_next,
            "initial_temperature": temperature_next,
            "time_step": timestep,
            "temperature_damping_timescale": 100.0,
            "n_print": project_parameter["nvt_run_time_steps"],
            "n_ionic_steps": project_parameter["nvt_run_time_steps"],
            "seed": project_parameter["seed"],
        },
        input_control_file={
            "fix": f"ensemble all nvt temp {temperature_next} {temperature_next} 0.1 drag 1"
        },
        write_restart_file=True,
        lmp_command=lmp_command,
    )
    restart_file_path = os.path.join(nvt_working_directory, "restart.out")
    _, parsed_output, _ = lammps_file_interface_function(
        working_directory=nve_working_directory,
        structure=basis_strain,
        potential=project_parameter["potential"],
        calc_mode="md",
        calc_kwargs={
            "time_step": timestep,
            "n_print": max(1, int(nve_run_time_steps / 100)),
            "n_ionic_steps": nve_run_time_steps,
            "seed": project_parameter["seed"],
        },
        read_restart_file=True,
        restart_file=restart_file_path,
        dump_final_structure=True,
        lmp_command=lmp_command,
    )
    structure_nve = structure_from_parsed_output(basis_strain, parsed_output, wrap=True)
    return StrainPointResult(
        strain=strain,
        pressure=np.mean(get_press(parsed_output=parsed_output, step=-20)),
        pressure_std=np.std(get_press(parsed_output=parsed_output, step=-20)),
        temperature=np.mean(parsed_output["generic"]["temperature"][-20:]),
        temperature_std=np.std(parsed_output["generic"]["temperature"][-20:]),
        structure=structure_nve,
        parsed_output=parsed_output,
    )
