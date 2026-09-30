"""Small, testable reference utilities for CGMR posterior projection."""

from .projection import (
    aggregate_expected_edit_risk,
    classify_state,
    gibbs_tilt_from_log_scores,
    solve_language_lambda,
)

__all__ = [
    "aggregate_expected_edit_risk",
    "classify_state",
    "gibbs_tilt_from_log_scores",
    "solve_language_lambda",
]
