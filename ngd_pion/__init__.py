"""NGD-Pion: a curvature-preconditioned, spectrum-preserving optimizer.

The algorithm is specified in `ALGORITHM.md`; each module implements one of
its sections and says so. `reference.py` is a numpy transcription of the same
document, kept deliberately naive, and serves as the oracle the torch path is
tested against rather than as anything to train with.
"""

from .covariance import CovarianceAccumulator
from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .factorization import Basis, basis_congruence, basis_identity_anchor, build_bases
from .hooks import attach, attached
from .fast import FastNGDPion
from .linalg import cayley, floor_eigenvalues, floor_spectrum, skew, spectral_norm
from .optimizer import NGDPion
from .with_s import BackwardRecorder, NGDPionS, attach_backward
from .with_s_fast import FastNGDPionS
from .op_damped import OpDampedNGDPion
from .momentum import MomentumNGDPionS
from .shampoo import ShampooPion

__all__ = [
    "NGDPion",
    "NGDPionS",
    "FastNGDPionS",
    "OpDampedNGDPion",
    "ShampooPion",
    "MomentumNGDPionS",
    "attach_backward",
    "BackwardRecorder",
    "attach",
    "attached",
    "CovarianceAccumulator",
    "Basis",
    "build_bases",
    "basis_congruence",
    "basis_identity_anchor",
    "generators",
    "natural_gradient",
    "fisher_apply",
    "trust_region_alpha",
    "cayley",
    "FastNGDPion",
    "spectral_norm",
    "skew",
    "floor_eigenvalues",
    "floor_spectrum",
]
