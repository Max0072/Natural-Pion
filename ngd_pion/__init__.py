"""Natural Pion: spectrum-preserving rotations along the natural gradient.

The method is `natural_pion` (`NaturalPion`, exact form; `NaturalPionFast`, the form we run); see
docs/paper/FINAL.md. It is built from the primitives in `covariance`, `direction`, `factorization`,
`linalg` and fed by the hooks in `hooks` (forward) and `with_s.attach_backward` (backward).

`optimizer`, `with_s` and `unified` are the reference implementations the work grew from (with the
`alpha` trust ratio of docs/history/ALGORITHM.md); `reference` is a numpy oracle for the tests; `pion_baseline` is
published Pion, the baseline. `legacy/` holds variants that were tried and set aside.
"""

from .covariance import CovarianceAccumulator
from .direction import fisher_apply, generators, natural_gradient, trust_region_alpha
from .factorization import Basis, basis_congruence, basis_identity_anchor, build_bases
from .hooks import attach, attached
from .linalg import cayley, floor_eigenvalues, floor_spectrum, skew, spectral_norm
from .natural_pion import NaturalPion, NaturalPionFast
from .optimizer import NGDPion
from .pion_baseline import Pion
from .with_s import BackwardRecorder, NGDPionS, attach_backward

__all__ = [
    "NaturalPion",
    "NaturalPionFast",
    "Pion",
    "attach",
    "attached",
    "attach_backward",
    "BackwardRecorder",
    "NGDPion",
    "NGDPionS",
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
    "spectral_norm",
    "skew",
    "floor_eigenvalues",
    "floor_spectrum",
]
