"""Core state definition and comparator-budgeted posterior projection for CGMR.

This module isolates the central math used by the executable micro experiment in
``scripts/run_er_cgmr_micro.py``.  It is intentionally NumPy-only so the core
projection can be unit-tested without loading an ASR model.

For candidate log scores s_i(h), candidate edit counts d_i(h), and a language-
shared multiplier lambda >= 0, the projected posterior is

    q_i(h) proportional to exp(s_i(h) - lambda * d_i(h)).

The language-level multiplier is chosen so that the aggregate expected edit risk
does not exceed the comparator edit budget, using one-dimensional bisection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def classify_state(
    baseline_error: float,
    adapted_1best_error: float,
    oracle_beam_error: float,
) -> str:
    """Return the paper's N/A/D state for one old-language utterance.

    N: adapted 1-best remains comparator-competitive.
    A: 1-best regresses, but a comparator-competitive candidate remains in beam.
    D: all comparator-competitive candidates have disappeared from the beam.
    """

    if adapted_1best_error <= baseline_error:
        return "N"
    if oracle_beam_error <= baseline_error:
        return "A"
    return "D"


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def gibbs_tilt_from_log_scores(
    log_scores: Sequence[float],
    edit_counts: Sequence[float],
    lam: float,
) -> np.ndarray:
    """Compute q(h) proportional to exp(log_score(h) - lam * edit_count(h))."""

    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lam must be a finite non-negative scalar")
    scores = _as_vector(log_scores, "log_scores")
    edits = _as_vector(edit_counts, "edit_counts")
    if scores.shape != edits.shape:
        raise ValueError("log_scores and edit_counts must have the same shape")
    if np.any(edits < 0):
        raise ValueError("edit_counts must be non-negative")

    logits = scores - lam * edits
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        raise FloatingPointError("failed to normalize projected posterior")
    return weights / total


def aggregate_expected_edit_risk(
    rows: Sequence[tuple[Sequence[float], Sequence[float]]],
    lam: float,
) -> float:
    """Return sum_i E_{q_i(lambda)}[d_i(h)] for a language cohort."""

    total = 0.0
    for log_scores, edit_counts in rows:
        edits = _as_vector(edit_counts, "edit_counts")
        q = gibbs_tilt_from_log_scores(log_scores, edits, lam)
        total += float(q @ edits)
    return total


def solve_language_lambda(
    rows: Sequence[tuple[Sequence[float], Sequence[float]]],
    comparator_budget: float,
    *,
    tol: float = 1e-10,
    max_bisection_steps: int = 100,
    max_bracket: float = 1e6,
) -> float:
    """Solve the language-level CGMR multiplier by monotone bisection.

    Returns 0 when the anchor posterior already satisfies the comparator budget.
    Raises ValueError when the requested budget is infeasible for the supplied
    candidate sets.
    """

    if not rows:
        raise ValueError("rows must be non-empty")
    if comparator_budget < 0 or not np.isfinite(comparator_budget):
        raise ValueError("comparator_budget must be finite and non-negative")
    if tol <= 0:
        raise ValueError("tol must be positive")

    risk0 = aggregate_expected_edit_risk(rows, 0.0)
    if risk0 <= comparator_budget + tol:
        return 0.0

    minimum_attainable = 0.0
    for _, edit_counts in rows:
        edits = _as_vector(edit_counts, "edit_counts")
        minimum_attainable += float(np.min(edits))
    if minimum_attainable > comparator_budget + tol:
        raise ValueError(
            "comparator budget is infeasible for the supplied candidate sets: "
            f"minimum attainable risk={minimum_attainable:.12g}, "
            f"budget={comparator_budget:.12g}"
        )

    low = 0.0
    high = 1.0
    while aggregate_expected_edit_risk(rows, high) > comparator_budget + tol:
        high *= 2.0
        if high > max_bracket:
            raise RuntimeError("failed to bracket a feasible lambda")

    for _ in range(max_bisection_steps):
        mid = (low + high) / 2.0
        risk = aggregate_expected_edit_risk(rows, mid)
        if risk > comparator_budget:
            low = mid
        else:
            high = mid
        if high - low <= tol * max(1.0, high):
            break

    return high
