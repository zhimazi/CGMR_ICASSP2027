"""Forgetting-state and first-passage utilities used by CGMR.

The paper distinguishes recognition regression from loss of competitive beam
support. This module keeps that diagnosis separate from the repair itself:

* N: the adapted 1-best remains comparator-competitive;
* A: recognition regresses, but a comparator-competitive candidate is still
  accessible in the adapted beam;
* D: no comparator-competitive candidate remains in the beam.

The first-passage helper mirrors the paper's temporal view by locating the first
observed recognition-regression and candidate-loss checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence


State = Literal["N", "A", "D"]
Onset = Literal["ranking_first", "candidate_loss_at_onset", "right_censored"]


@dataclass(frozen=True)
class FirstPassage:
    """First observed recognition-regression and candidate-loss times."""

    tau_rec: int | float | None
    tau_loss: int | float | None
    onset: Onset


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def classify_state(
    baseline_error: float,
    adapted_1best_error: float,
    oracle_beam_error: float,
) -> State:
    """Return the paper's N/A/D state for one old-language utterance.

    This is the direct code counterpart of the state definition in Eq. (2).
    """

    baseline = _finite_nonnegative(baseline_error, "baseline_error")
    top = _finite_nonnegative(adapted_1best_error, "adapted_1best_error")
    oracle = _finite_nonnegative(oracle_beam_error, "oracle_beam_error")
    if oracle > top:
        raise ValueError("oracle_beam_error cannot exceed adapted_1best_error")

    if top <= baseline:
        return "N"
    if oracle <= baseline:
        return "A"
    return "D"


def state_path(
    baseline_error: float,
    observations: Sequence[tuple[int | float, float, float]],
) -> list[tuple[int | float, State]]:
    """Convert checkpoint observations into an ordered N/A/D state path."""

    if not observations:
        raise ValueError("observations must be non-empty")
    steps = [item[0] for item in observations]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise ValueError("observation steps must be strictly increasing")
    return [
        (step, classify_state(baseline_error, top_error, oracle_error))
        for step, top_error, oracle_error in observations
    ]


def first_passage(
    baseline_error: float,
    observations: Sequence[tuple[int | float, float, float]],
) -> FirstPassage:
    """Compute observed-grid first-passage times from checkpoint errors.

    ``tau_rec`` is the first checkpoint whose 1-best error exceeds the
    comparator baseline. ``tau_loss`` is the first checkpoint whose beam-oracle
    error exceeds that baseline. Missing events are represented by ``None``.
    """

    baseline = _finite_nonnegative(baseline_error, "baseline_error")
    path = state_path(baseline, observations)

    tau_rec = next((step for step, state in path if state in {"A", "D"}), None)
    tau_loss = next((step for step, state in path if state == "D"), None)

    if tau_rec is None:
        return FirstPassage(None, tau_loss, "right_censored")
    if tau_loss is None or tau_rec < tau_loss:
        return FirstPassage(tau_rec, tau_loss, "ranking_first")
    if tau_rec == tau_loss:
        return FirstPassage(tau_rec, tau_loss, "candidate_loss_at_onset")
    raise ValueError("candidate loss cannot precede recognition regression")
