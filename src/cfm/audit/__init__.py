"""Certified-by-Residual auditing utilities for CrowdFM."""

from .bootstrap import MonteCarloResult, conditional_monte_carlo_test
from .calibration import (
    CALIBRATION_VARIANTS,
    CalibrationWorldResult,
    run_calibration_world,
)
from .disagreement import (
    ResidualResult,
    build_residual_matrix,
    confusion_posterior_parameters,
    estimate_confusion_matrices,
    item_conditioned_disagreement,
    spectral_statistic,
)
from .pipeline import CBRConfig, run_cbr_audit
from .split import CrossFitSplit, make_crossfit_split
from .synthetic import (
    SyntheticWorld,
    SyntheticWorldConfig,
    generate_synthetic_world,
)

__all__ = [
    "CALIBRATION_VARIANTS",
    "CBRConfig",
    "CalibrationWorldResult",
    "CrossFitSplit",
    "MonteCarloResult",
    "ResidualResult",
    "SyntheticWorld",
    "SyntheticWorldConfig",
    "build_residual_matrix",
    "conditional_monte_carlo_test",
    "confusion_posterior_parameters",
    "estimate_confusion_matrices",
    "generate_synthetic_world",
    "item_conditioned_disagreement",
    "make_crossfit_split",
    "run_calibration_world",
    "run_cbr_audit",
    "spectral_statistic",
]
