from dataclasses import dataclass

from ase.atoms import Atoms


@dataclass
class StrainPointResult:
    """Outcome of a single strain point in the strain scan (NVT equilibration + NVE production)."""

    strain: float
    pressure: float
    pressure_std: float
    temperature: float
    temperature_std: float
    structure: Atoms
    parsed_output: dict
