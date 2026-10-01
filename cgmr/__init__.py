"""Reference implementation of Comparator-Guided Minimum-Information Repair."""

from .objective import (
    mixed_repair_loss,
    restricted_score_gradient,
    restricted_target_cross_entropy,
)
from .projection import (
    LanguageProjection,
    aggregate_expected_edit_risk,
    gibbs_tilt_from_log_scores,
    project_language_cohort,
    solve_language_lambda,
)
from .states import FirstPassage, classify_state, first_passage, state_path

__all__ = [
    "FirstPassage",
    "LanguageProjection",
    "aggregate_expected_edit_risk",
    "classify_state",
    "first_passage",
    "gibbs_tilt_from_log_scores",
    "mixed_repair_loss",
    "project_language_cohort",
    "restricted_score_gradient",
    "restricted_target_cross_entropy",
    "solve_language_lambda",
    "state_path",
]
