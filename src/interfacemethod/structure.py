import operator

from ase.atoms import Atoms
import numpy as np
import structuretoolkit as stk


def check_diamond(structure: Atoms) -> bool:
    """
    Utility function to check if the structure is fcc, bcc, hcp or diamond

    Args:
        structure (ase.atoms.Atoms): Atomistic Structure object to check

    Returns:
        bool: true if diamond else false
    """
    cna_dict = stk.analyse.get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=True
    )
    dia_dict = stk.analyse.get_diamond_structure_descriptors(
        structure=structure, mode="total", ovito_compatibility=True
    )
    return (
        cna_dict["CommonNeighborAnalysis.counts.OTHER"]
        > dia_dict["IdentifyDiamond.counts.OTHER"]
    )


def analyse_structure(structure: Atoms, mode: str = "total", diamond: bool = False):
    """
    Use either common neighbor analysis or the diamond structure detector

    Args:
        structure (ase.atoms.Atoms): The structure to analyze.
        mode ("total"/"numeric"/"str"): Controls the style and level
            of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        diamond (bool): Flag to either use the diamond structure detector or
            the common neighbor analysis.

    Returns:
        (depends on `mode`)
    """
    if not diamond:
        return stk.analyse.get_adaptive_cna_descriptors(
            structure=structure, mode=mode, ovito_compatibility=True
        )
    else:
        return stk.analyse.get_diamond_structure_descriptors(
            structure=structure, mode=mode, ovito_compatibility=True
        )


def analyse_minimized_structure(structure: Atoms):
    """
    Determine the dominant structural motif of a minimised structure.

    Runs the appropriate structure analysis (CNA or diamond detector) and returns
    the dominant phase key, total atom count, and a threshold fraction used to
    distinguish solid from liquid during bisection.

    Args:
        structure (Atoms): The energy-minimised structure to analyse.

    Returns:
        tuple: A 5-tuple containing:
            - structure (Atoms): The input structure (passed through).
            - key_max (str): The OVITO key of the dominant structural motif.
            - number_of_atoms (int): Total number of atoms.
            - distribution_initial_half (float): Half the initial fraction of atoms in the dominant phase.
            - final_structure_dict (dict): Full analysis result dictionary.
    """
    diamond_flag = check_diamond(structure=structure)
    final_structure_dict = analyse_structure(
        structure=structure, mode="total", diamond=diamond_flag
    )
    key_max = max(final_structure_dict.items(), key=operator.itemgetter(1))[0]
    number_of_atoms = len(structure)
    distribution_initial = final_structure_dict[key_max] / number_of_atoms
    distribution_initial_half = distribution_initial / 2
    return (
        structure,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        final_structure_dict,
    )


def get_voronoi_volume(structure_lst: list[Atoms]):
    max_lst, mean_lst = [], []
    for structure in structure_lst:
        structure_voronoi_lst = stk.analyse.get_voronoi_volumes(structure)
        max_lst.append(np.max(structure_voronoi_lst))
        mean_lst.append(np.mean(structure_voronoi_lst))
    return max_lst, mean_lst


def remove_selective_dynamics(basis: Atoms) -> Atoms:
    """
    If a FixAtoms constraint is set, allow all atoms to move again by clearing the constraint.

    Args:
        basis (ase.atoms.Atoms): Atomistic structure object

    Returns:
        Atoms: Atomistic structure object with all constraints removed
    """
    basis = basis.copy()
    basis.set_constraint()
    return basis
