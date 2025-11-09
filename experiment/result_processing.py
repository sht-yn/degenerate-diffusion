"""Utilities for converting seed-runner batches into tabular summaries.

This module centralizes the CSV export and summary-statistic logic that was
previously duplicated across notebooks. Each helper accepts JAX arrays produced
by the seed runner and returns host-side pandas objects for further analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ArrayMap = Mapping[str, jax.Array]

ColumnComponent = tuple[str, int]


def _parse_parameter_key(param: str) -> tuple[str, int, int]:
    parts = param.split("_")
    if len(parts) != 3:
        msg = f"Unexpected parameter format: {param}."
        raise ValueError(msg)
    prefix, component, stage = parts
    return prefix, int(component), int(stage)


def _gather_parameter_data(
    df: pd.DataFrame,
) -> tuple[dict[str, dict[int, dict[int, tuple[float, float]]]], set[int]]:
    parameter_data: dict[str, dict[int, dict[int, tuple[float, float]]]] = {}
    stages: set[int] = set()
    for row in df.itertuples(index=False):
        prefix, component_idx, stage_idx = _parse_parameter_key(str(row.parameter))
        parameter_data.setdefault(prefix, {}).setdefault(stage_idx, {})[component_idx] = (
            float(row.mean),
            float(row.std),
        )
        stages.add(stage_idx)
    return parameter_data, stages


def _resolve_column_components(
    parameter_data: dict[str, dict[int, dict[int, tuple[float, float]]]],
    label_templates: Mapping[str, str],
    preferred_order: Sequence[str],
) -> tuple[list[ColumnComponent], list[str]]:
    parameter_order = [name for name in preferred_order if name in parameter_data]
    parameter_order.extend(sorted(name for name in parameter_data if name not in preferred_order))

    for name in parameter_order:
        if name not in label_templates:
            msg = f"No LaTeX template registered for parameter prefix '{name}'."
            raise ValueError(msg)

    column_components: list[ColumnComponent] = []
    headers: list[str] = ["$k$"]

    for prefix in parameter_order:
        stage_data = parameter_data[prefix]
        components = sorted({idx for values in stage_data.values() for idx in values})
        template = label_templates[prefix]
        for component_idx in components:
            column_components.append((prefix, component_idx))
            formatted_header = template.replace("{i}", str(component_idx))
            headers.append(f"${formatted_header}$")

    return column_components, headers


def _format_numeric(value: float, decimal_places: int) -> str:
    fmt = f"{{:.{decimal_places}f}}"
    return fmt.format(value)


def _build_true_row(
    column_components: Sequence[ColumnComponent],
    true_values: Mapping[str, Sequence[float] | np.ndarray | jax.Array],
    decimal_places: int,
) -> str:
    row_values: list[str] = ["true"]
    for prefix, component_idx in column_components:
        if prefix not in true_values:
            msg = f"Missing true value entry for '{prefix}'."
            raise ValueError(msg)
        values_arr = np.asarray(true_values[prefix]).reshape(-1)
        if component_idx >= values_arr.size:
            msg = f"True values for '{prefix}' do not contain component index {component_idx}."
            raise ValueError(msg)
        formatted = _format_numeric(float(values_arr[component_idx]), decimal_places)
        row_values.append(formatted)
    return " & ".join(row_values) + " \\\\"


def _build_stage_rows(
    column_components: Sequence[ColumnComponent],
    parameter_data: Mapping[str, Mapping[int, Mapping[int, tuple[float, float]]]],
    stages: Sequence[int],
    decimal_places: int,
) -> list[str]:
    rows: list[str] = []
    for stage_idx in stages:
        row_values = [str(stage_idx)]
        for prefix, component_idx in column_components:
            component_data = parameter_data[prefix].get(stage_idx, {})
            if component_idx not in component_data:
                row_values.append("-")
                continue
            mean_val, std_val = component_data[component_idx]
            mean_str = _format_numeric(mean_val, decimal_places)
            std_str = _format_numeric(std_val, decimal_places)
            row_values.append(f"{mean_str} ({std_str})")
        rows.append(" & ".join(row_values) + " \\\\")
    return rows


def _batch_to_columns(batch: jax.Array, base_name: str) -> dict[str, np.ndarray]:
    """Expand a (seed, k, i) tensor into flat columns keyed by ``base_name``."""
    batch_host = np.asarray(jax.device_get(batch))
    if batch_host.ndim == 2:
        # Interpret shape (seed, k) as a single-parameter trajectory.
        columns: dict[str, np.ndarray] = {}
        for k_idx in range(batch_host.shape[1]):
            columns[f"{base_name}_0_{k_idx + 1}"] = batch_host[:, k_idx]
        return columns

    if batch_host.ndim != 3:
        msg = f"Expected array with ndim 2 or 3, received shape {batch_host.shape}."
        raise ValueError(msg)

    _, k_dim, i_dim = batch_host.shape
    columns: dict[str, np.ndarray] = {}
    for i_idx in range(i_dim):
        for k_idx in range(k_dim):
            columns[f"{base_name}_{i_idx}_{k_idx + 1}"] = batch_host[:, k_idx, i_idx]
    return columns


def build_results_dataframe(
    seeds: jax.Array,
    batches: ArrayMap,
    *,
    set_index: bool = True,
) -> pd.DataFrame:
    """Construct a ``DataFrame`` whose index enumerates seeds."""
    seed_values = np.asarray(jax.device_get(jnp.asarray(seeds))).reshape(-1)
    column_data: dict[str, np.ndarray] = {"seed": seed_values}
    for name, batch in batches.items():
        column_data.update(_batch_to_columns(batch, name))

    df = pd.DataFrame(column_data)
    return df.set_index("seed") if set_index else df


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std per column in tidy form."""
    return df.agg(["mean", "std"]).T.rename_axis("parameter").reset_index()


def save_results(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the provided ``DataFrame`` to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def process_and_save(
    seeds: jax.Array,
    batches: ArrayMap,
    *,
    output_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """High-level helper that returns the full results, summary, and saved path."""
    df = build_results_dataframe(seeds, batches)
    summary = compute_summary_table(df)
    csv_path = save_results(df, output_path)
    return df, summary, csv_path


def build_theta_latex_table(
    summary_csv: str | Path,
    true_values: Mapping[str, Sequence[float] | np.ndarray | jax.Array],
    *,
    decimal_places: int = 3,
) -> str:
    """Render a LaTeX table summarising parameter means/std alongside true values."""
    df = pd.read_csv(summary_csv)
    required_cols = {"parameter", "mean", "std"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        msg = f"Missing expected columns: {missing}."
        raise ValueError(msg)
    parameter_data, stage_set = _gather_parameter_data(df)
    preferred_order = ("theta10", "theta1", "theta2", "theta3")
    label_templates = {
        "theta10": "\\theta^{{k,0}}_{1,{i}}",
        "theta1": "\\theta^{{k}}_{1,{i}}",
        "theta2": "\\theta^{{k}}_{2,{i}}",
        "theta3": "\\theta^{{k}}_{3,{i}}",
    }
    column_components, headers = _resolve_column_components(
        parameter_data,
        label_templates,
        preferred_order,
    )
    column_spec = "c" * len(headers)
    header_row = " & ".join(headers)
    latex_lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        f"{header_row} \\",
        "\\hline",
    ]
    latex_lines.append(_build_true_row(column_components, true_values, decimal_places))
    stage_rows = _build_stage_rows(
        column_components,
        parameter_data,
        sorted(stage_set),
        decimal_places,
    )
    latex_lines.extend(stage_rows)
    latex_lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(latex_lines)
