from __future__ import annotations

import math

import pytest

from scripts.aggregate_seed_results import aggregate


def _rows(seeds):
    return [
        {
            "seed": str(seed),
            "section": "main",
            "backbone": "Whisper-small",
            "dataset": "CL-MASR",
            "method": "CGMR",
            "old_gain": str(index + 1),
            "current_delta": str(-(index + 1)),
        }
        for index, seed in enumerate(seeds)
    ]


def test_five_seed_sample_statistics():
    seeds = [2027, 2028, 2029, 2030, 2031]
    output = aggregate(
        _rows(seeds),
        ["section", "backbone", "dataset", "method"],
        ["old_gain", "current_delta"],
        seeds,
    )
    assert len(output) == 1
    row = output[0]
    assert row["n_seeds"] == 5
    assert row["seed_ids"] == "2027;2028;2029;2030;2031"
    assert math.isclose(row["old_gain_mean"], 3.0)
    assert math.isclose(row["old_gain_std"], math.sqrt(2.5))
    assert math.isclose(row["current_delta_mean"], -3.0)
    assert math.isclose(row["current_delta_std"], math.sqrt(2.5))


def test_incomplete_seed_set_fails_closed():
    expected = [2027, 2028, 2029, 2030, 2031]
    with pytest.raises(ValueError, match="expected exactly seeds"):
        aggregate(
            _rows(expected[:-1]),
            ["section", "backbone", "dataset", "method"],
            ["old_gain", "current_delta"],
            expected,
        )
