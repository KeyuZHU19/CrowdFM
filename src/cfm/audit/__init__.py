"""Certified-by-Residual auditing utilities for CrowdFM."""

from .bootstrap import MonteCarloResult, conditional_monte_carlo_test
from .disagreement import (
    ResidualResult,
    build_residual_matrix,
    estimate_confusion_matrices,
    item_conditioned_disagreement,
    spectral_statistic,
)
from .pipeline import CBRConfig, run_cbr_audit
from .split import CrossFitSplit, make_crossfit_split

__all__ = [
    "CBRConfig",
    "CrossFitSplit",
    "MonteCarloResult",
    "ResidualResult",
    "build_residual_matrix",
    "conditional_monte_carlo_test",
    "estimate_confusion_matrices",
    "item_conditioned_disagreement",
    "make_crossfit_split",
    "run_cbr_audit",
    "spectral_statistic",
]
