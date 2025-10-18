"""Regression test ensuring FNmodel detail CSV matches the baseline output."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from degenerate_diffusion.internal.fn_model_pipeline import run_fnmodel_estimation

BASELINE_DETAIL_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "experiments"
    / "FNmodel_test_nh10_h0.05"
    / "detail.csv"
)


def _load_baseline(path: Path) -> dict[tuple[int, str, int, str], tuple[float, ...]]:
    records: dict[tuple[int, str, int, str], tuple[float, ...]] = {}
    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = (int(row["seed"]), row["phase"], int(row["k"]), row["component"])
            values = tuple(float(part) for part in row["values"].split())
            records[key] = values
    return records


def _flatten_component(array_like: npt.ArrayLike) -> tuple[float, ...]:
    flat = np.asarray(array_like, dtype=float).reshape(-1)
    return tuple(float(value) for value in flat)


@pytest.mark.slow
def test_fnmodel_detail_matches_baseline() -> None:
    baseline = _load_baseline(BASELINE_DETAIL_PATH)
    assert baseline, "基準 CSV が空です。"

    results = run_fnmodel_estimation(seeds=range(1), show_progress=False)

    generated: dict[tuple[int, str, int, str], tuple[float, ...]] = {}
    for seed, iteration_estimates in results.loop_results.items():
        for estimate in iteration_estimates:
            for phase_name, theta_components in (
                ("stage0", estimate.theta_stage0),
                ("final", estimate.theta_final),
            ):
                generated[(seed, phase_name, estimate.k, "theta1")] = _flatten_component(
                    theta_components[0]
                )
                generated[(seed, phase_name, estimate.k, "theta2")] = _flatten_component(
                    theta_components[1]
                )
                generated[(seed, phase_name, estimate.k, "theta3")] = _flatten_component(
                    theta_components[2]
                )

    unexpected = set(generated) - set(baseline)
    assert not unexpected, f"想定外のキーを生成しました: {sorted(unexpected)!r}"

    for key, expected_values in baseline.items():
        assert key in generated, f"{key} のデータが生成結果に存在しません。"
        actual_values = generated[key]
        np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=1e-5)
