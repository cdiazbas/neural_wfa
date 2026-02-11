"""
Probabilistic WFA Module.

This subpackage provides probabilistic extensions to the WFA inversion:
- ProbabilisticMagneticField: Stores distributions instead of point estimates
- ProbabilisticPixelSolver: Explicit probabilistic solver
- Losses: NLL, heteroscedastic loss, spatial priors
"""

from neural_wfa.probabilistic.field import ProbabilisticMagneticField
from neural_wfa.probabilistic.utils import broadcast_sigma_obs
from neural_wfa.probabilistic.pixel_solver import ProbabilisticPixelSolver

__all__ = [
    "ProbabilisticMagneticField",
    "ProbabilisticPixelSolver",
    "broadcast_sigma_obs",
]
