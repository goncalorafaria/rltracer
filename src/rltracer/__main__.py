"""Fire-powered command line interface for RLTracer."""

from __future__ import annotations

from pathlib import Path

import fire

from .analysis import (
    PassAnalysisConfig,
    calculate_prompt_pass_metrics,
    plot_pass_rate_distributions,
    write_pass_analysis,
)
from .primerl import open_prime_rl_run


def inspect_run(run: str, tokenizer: str, step: int | None = None) -> None:
    """List saved steps or the prompt groups for one step."""
    tracer = open_prime_rl_run(run, tokenizer)
    try:
        if step is None:
            print("\n".join(str(item.number) for item in tracer.steps()))
            return
        for prompt in tracer.step(step).prompts():
            print(f"{prompt.key}\t{prompt.rollout_count}\t{prompt.preview}")
    finally:
        tracer.close()


def analyze(
    run: str,
    tokenizer: str,
    output: str,
    bin_size: int = 40,
    split: str = "train",
    success_path: str = "metrics.correct_final_label",
    success_threshold: float = 1.0,
    step_start: int | None = None,
    step_end: int | None = None,
) -> None:
    """Write prompt-level pass-rate metrics for a PrimeRL trace run."""
    tracer = open_prime_rl_run(run, tokenizer)
    try:
        config = PassAnalysisConfig(
            bin_size=bin_size,
            split=split,
            success_path=success_path,
            success_threshold=success_threshold,
            step_start=step_start,
            step_end=step_end,
        )
        metrics, diagnostics = calculate_prompt_pass_metrics(tracer, config)
        destination = write_pass_analysis(output, metrics, config, diagnostics)
        print(f"Wrote {len(metrics)} prompt/bin rows to {destination}")
    finally:
        tracer.close()


def plot(
    metrics_csv: str,
    output: str,
    metric: str = "pass_at_1",
    histogram_bins: int = 10,
) -> None:
    """Plot pass-rate distributions from an analysis CSV."""
    destination = plot_pass_rate_distributions(
        Path(metrics_csv),
        Path(output),
        metric=metric,
        histogram_bins=histogram_bins,
    )
    print(f"Wrote {destination}")


def main() -> None:
    fire.Fire({"inspect": inspect_run, "analyze": analyze, "plot": plot})


if __name__ == "__main__":
    main()
