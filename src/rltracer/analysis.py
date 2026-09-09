"""Prompt-level pass-rate analysis for saved PrimeRL rollout traces."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .primerl import PrimeRLAdapter, fingerprint
from .tracer import RLTracer


@dataclass(frozen=True)
class PassAnalysisConfig:
    """Definition of one prompt-level pass-rate analysis."""

    bin_size: int = 40
    split: str = "train"
    success_path: str = "metrics.correct_final_label"
    success_threshold: float = 1.0
    step_start: int | None = None
    step_end: int | None = None

    def __post_init__(self) -> None:
        if self.bin_size < 1:
            raise ValueError("bin_size must be positive")
        if not self.success_path:
            raise ValueError("success_path must not be empty")
        if self.step_start is not None and self.step_end is not None and self.step_start > self.step_end:
            raise ValueError("step_start must not exceed step_end")


@dataclass(frozen=True)
class PromptPassMetric:
    """Pass-rate statistics for one prompt in one inclusive step range."""

    step_start: int
    step_end: int
    prompt_key: str
    prompt_preview: str
    rollout_count: int
    success_count: int
    pass_at_1: float
    pass_at_4: float | None


def _value_at_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _is_success(record: dict[str, Any], config: PassAnalysisConfig) -> bool | None:
    value = _value_at_path(record, config.success_path)
    if isinstance(value, bool):
        value = float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value >= config.success_threshold


def pass_at_k(success_count: int, rollout_count: int, k: int) -> float | None:
    """Estimate the chance that k saved rollouts include one success.

    Uses 1 - C(n-c, k) / C(n, k). It is undefined when fewer than k rollouts
    were saved for a prompt/bin.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if rollout_count < k:
        return None
    failures = rollout_count - success_count
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(rollout_count, k)


def _step_range(step: int, bin_size: int) -> tuple[int, int]:
    start = ((step - 1) // bin_size) * bin_size + 1
    return start, start + bin_size - 1


def _prime_rl_adapter(run: RLTracer) -> PrimeRLAdapter:
    if not isinstance(run.adapter, PrimeRLAdapter):
        raise TypeError("Pass analysis currently requires a PrimeRL trace run")
    return run.adapter


def calculate_prompt_pass_metrics(
    run: RLTracer,
    config: PassAnalysisConfig = PassAnalysisConfig(),
) -> tuple[list[PromptPassMetric], dict[str, int]]:
    """Calculate prompt metrics from saved trace records.

    Missing success fields are excluded and counted in diagnostics. Trace files
    under all/ are used, avoiding the filtered effective/ subset.
    """
    adapter = _prime_rl_adapter(run)
    grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
    diagnostics = {"trace_count": 0, "included_rollout_count": 0, "missing_success_count": 0}

    for step in adapter.list_steps(config.split):
        if config.step_start is not None and step < config.step_start:
            continue
        if config.step_end is not None and step > config.step_end:
            continue
        for path in adapter._trace_files(step):
            if config.split not in Path(path).parts:
                continue
            for _line_index, messages, record in adapter._iter_trace_file(path):
                diagnostics["trace_count"] += 1
                success = _is_success(record, config)
                if success is None:
                    diagnostics["missing_success_count"] += 1
                    continue
                key, preview = fingerprint(messages, adapter.prompt_parser)
                start, end = _step_range(step, config.bin_size)
                group = grouped.setdefault(
                    (start, end, key),
                    {"preview": preview, "rollout_count": 0, "success_count": 0},
                )
                group["rollout_count"] += 1
                group["success_count"] += int(success)
                diagnostics["included_rollout_count"] += 1

    metrics = [
        PromptPassMetric(
            step_start=start,
            step_end=end,
            prompt_key=key,
            prompt_preview=group["preview"],
            rollout_count=group["rollout_count"],
            success_count=group["success_count"],
            pass_at_1=group["success_count"] / group["rollout_count"],
            pass_at_4=pass_at_k(group["success_count"], group["rollout_count"], 4),
        )
        for (start, end, key), group in grouped.items()
    ]
    return sorted(metrics, key=lambda item: (item.step_start, item.prompt_key)), diagnostics


def _histogram(values: Iterable[float | None]) -> list[dict[str, Any]]:
    counts = [0] * 10
    for value in values:
        if value is not None:
            counts[min(int(value * 10), 9)] += 1
    return [{"lower": index / 10, "upper": (index + 1) / 10, "prompt_count": count} for index, count in enumerate(counts)]


def _bin_summaries(metrics: list[PromptPassMetric]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[PromptPassMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.step_start, metric.step_end)].append(metric)
    summaries: list[dict[str, Any]] = []
    for (start, end), rows in sorted(grouped.items()):
        total = sum(row.rollout_count for row in rows)
        successes = sum(row.success_count for row in rows)
        pass4_rows = [row.pass_at_4 for row in rows if row.pass_at_4 is not None]
        summaries.append({
            "step_start": start, "step_end": end, "prompt_count": len(rows),
            "rollout_count": total, "success_count": successes,
            "micro_pass_at_1": successes / total if total else None,
            "mean_prompt_pass_at_1": sum(row.pass_at_1 for row in rows) / len(rows) if rows else None,
            "mean_prompt_pass_at_4": sum(pass4_rows) / len(pass4_rows) if pass4_rows else None,
        })
    return summaries


def write_pass_analysis(
    output_dir: str | Path,
    metrics: list[PromptPassMetric],
    config: PassAnalysisConfig,
    diagnostics: dict[str, int],
) -> Path:
    """Write portable CSV/JSON, a Markdown report, and histogram summaries."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(metric) for metric in metrics]
    summaries = _bin_summaries(metrics)
    histograms = {"pass_at_1": _histogram(metric.pass_at_1 for metric in metrics), "pass_at_4": _histogram(metric.pass_at_4 for metric in metrics)}
    with (output / "prompt_pass_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PromptPassMetric.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "bin_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["step_start", "step_end"])
        writer.writeheader()
        writer.writerows(summaries)
    (output / "analysis.json").write_text(json.dumps({"config": asdict(config), "diagnostics": diagnostics, "bin_summary": summaries, "histograms": histograms}, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# RL trace prompt pass analysis", "", f"- Split: `{config.split}`", f"- Step bin size: `{config.bin_size}`",
        f"- Success: `{config.success_path} >= {config.success_threshold}`",
        f"- Included rollouts: `{diagnostics['included_rollout_count']}` of `{diagnostics['trace_count']}` traces",
        f"- Missing success field: `{diagnostics['missing_success_count']}`", "",
        "## Per-bin summary", "", "| Steps | Prompts | Rollouts | Successes | Micro pass@1 | Mean prompt pass@1 | Mean prompt pass@4 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(f"| {row['step_start']}–{row['step_end']} | {row['prompt_count']} | {row['rollout_count']} | {row['success_count']} | {fmt(row['micro_pass_at_1'])} | {fmt(row['mean_prompt_pass_at_1'])} | {fmt(row['mean_prompt_pass_at_4'])} |")
    for name, histogram in histograms.items():
        lines.extend(["", f"## {name} histogram across prompt/bin rows", "", "| Range | Prompt rows |", "| --- | ---: |"])
        lines.extend(f"| {row['lower']:.1f}–{row['upper']:.1f} | {row['prompt_count']} |" for row in histogram)
    lines.extend(["", "pass@4 estimates at least one success in four distinct saved rollouts; it is blank when fewer than four rollouts exist for a prompt/bin.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def plot_pass_rate_distributions(
    metrics_csv: str | Path,
    output_path: str | Path,
    *,
    metric: str = "pass_at_1",
    histogram_bins: int = 10,
) -> Path:
    """Plot one histogram and KDE series per step range in a single PNG.

    The input is the ``prompt_pass_metrics.csv`` produced by
    :func:`write_pass_analysis`. KDEs are clipped to the valid pass-rate
    interval [0, 1] and use a small Gaussian fallback for constant bins.
    """
    if metric not in {"pass_at_1", "pass_at_4"}:
        raise ValueError("metric must be pass_at_1 or pass_at_4")
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2")
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    with Path(metrics_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_value = row.get(metric)
            if raw_value in {None, "", "None"}:
                continue
            try:
                value = float(raw_value)
                start, end = int(row["step_start"]), int(row["step_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                grouped[(start, end)].append(value)
    if not grouped:
        raise ValueError(f"No numeric {metric} values in {metrics_csv}")

    figure, (histogram_axis, kde_axis) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, layout="constrained")
    x = np.linspace(0, 1, 512)
    colors = plt.get_cmap("viridis")
    ordered_groups = sorted(grouped.items())
    edges = np.linspace(0, 1, histogram_bins + 1)
    for index, ((start, end), values) in enumerate(ordered_groups):
        color = colors(index / max(len(ordered_groups) - 1, 1))
        label = f"steps {start}–{end} (n={len(values)})"
        data = np.clip(np.asarray(values, dtype=float), 0, 1)
        histogram_axis.hist(data, bins=edges, density=True, histtype="step", linewidth=1.8, color=color, label=label)
        if len(data) >= 2 and not np.allclose(data, data[0]):
            density = gaussian_kde(data, bw_method="scott")(x)
        else:
            # A degenerate bin still needs a visible, labelled distribution.
            bandwidth = 0.035
            density = np.exp(-0.5 * ((x - data[0]) / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))
        kde_axis.plot(x, density, linewidth=2.0, color=color, label=label)
    histogram_axis.set_title(f"Prompt {metric.replace('_', '@')} distributions by training-step bin")
    histogram_axis.set_ylabel("Density")
    histogram_axis.set_xlim(0, 1)
    histogram_axis.grid(axis="y", alpha=0.22)
    kde_axis.set_xlabel("Prompt pass rate")
    kde_axis.set_ylabel("KDE density")
    kde_axis.set_xlim(0, 1)
    kde_axis.grid(axis="y", alpha=0.22)
    for axis in (histogram_axis, kde_axis):
        axis.legend(title="Step range", loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
