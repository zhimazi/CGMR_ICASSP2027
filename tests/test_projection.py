import numpy as np
import pytest

from cgmr.projection import (
    aggregate_expected_edit_risk,
    gibbs_tilt_from_log_scores,
    project_language_cohort,
    solve_language_lambda,
)


def test_gibbs_tilt_is_probability_distribution():
    q = gibbs_tilt_from_log_scores([0.0, -0.5, -1.0], [5, 2, 0], lam=0.7)
    assert np.all(q >= 0)
    assert np.isclose(q.sum(), 1.0)


def test_bisection_meets_language_budget():
    rows = [
        ([0.0, -0.4, -1.0], [4, 2, 0]),
        ([0.0, -0.2, -0.8], [3, 1, 0]),
    ]
    budget = 2.0
    assert aggregate_expected_edit_risk(rows, 0.0) > budget
    lam = solve_language_lambda(rows, budget)
    assert lam > 0
    assert aggregate_expected_edit_risk(rows, lam) <= budget + 1e-8


def test_complete_projection_returns_targets_and_diagnostics():
    rows = [
        ([0.0, -0.4, -1.0], [4, 2, 0]),
        ([0.0, -0.2, -0.8], [3, 1, 0]),
    ]
    result = project_language_cohort(rows, comparator_budget=2.0)
    assert result.lambda_value > 0
    assert result.risk_before > result.comparator_budget
    assert result.risk_after <= result.comparator_budget + 1e-8
    assert len(result.posteriors) == 2
    assert all(np.isclose(q.sum(), 1.0) for q in result.posteriors)


def test_zero_lambda_when_anchor_already_feasible():
    rows = [([0.0, -1.0], [1, 0])]
    assert solve_language_lambda(rows, comparator_budget=1.0) == 0.0


def test_infeasible_budget_raises():
    rows = [([0.0, -1.0], [3, 2])]
    with pytest.raises(ValueError, match="infeasible"):
        solve_language_lambda(rows, comparator_budget=1.0)
