from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate manuscript-scale per-seed metrics with an explicit five-seed contract."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["section", "backbone", "dataset", "method"],
        help="Columns defining one reported experimental condition.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["old_gain", "current_delta"],
        help="Numeric columns to aggregate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Expected seed identifiers. Defaults to the manuscript protocol.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_columns(rows: list[dict[str, str]], required: set[str]) -> None:
    if not rows:
        raise ValueError("Input CSV is empty.")
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def aggregate(
    rows: list[dict[str, str]],
    group_by: list[str],
    metrics: list[str],
    expected_seeds: list[int],
) -> list[dict[str, object]]:
    validate_columns(rows, {"seed", *group_by, *metrics})
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_by)].append(row)

    expected = sorted(expected_seeds)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        observed = sorted(int(row["seed"]) for row in group)
        if observed != expected:
            condition = ", ".join(f"{name}={value}" for name, value in zip(group_by, key))
            raise ValueError(
                f"{condition}: expected exactly seeds {expected}, observed {observed}. "
                "Do not aggregate incomplete or duplicated seed sets."
            )

        result: dict[str, object] = dict(zip(group_by, key))
        result["n_seeds"] = len(group)
        result["seed_ids"] = ";".join(str(seed) for seed in expected)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(result)
    return output


def write_rows(
    path: Path,
    rows: list[dict[str, object]],
    group_by: list[str],
    metrics: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *group_by,
        "n_seeds",
        "seed_ids",
        *[name for metric in metrics for name in (f"{metric}_mean", f"{metric}_std")],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("--seeds must contain exactly five unique seed identifiers.")
    rows = read_rows(args.input_csv)
    aggregated = aggregate(rows, args.group_by, args.metrics, args.seeds)
    write_rows(args.output_csv, aggregated, args.group_by, args.metrics)


if __name__ == "__main__":
    main()
