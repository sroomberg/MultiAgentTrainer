"""Generate markdown reports for training runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import TrainerConfig
from .runner import ExperimentResult


def generate_report(
    run_id: str,
    config: TrainerConfig,
    results: list[ExperimentResult],
) -> str:
    """Build a markdown report summarising a training run."""
    lines: list[str] = [
        f"# MultiAgentTrainer Report: {run_id}",
        "",
        f"**Date**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Autoresearch repo**: `{config.autoresearch.repo}`"
        f" ({config.autoresearch.branch})",
        f"**Train time budget**: {config.autoresearch.train_time}s",
        f"**Max experiments**: {config.training.max_experiments}",
        "",
    ]

    # Data sources
    if config.sources:
        lines.extend(["## Data Sources", ""])
        for src in config.sources:
            lines.append(f"- {src.describe()}")
        lines.append("")

    # Summary table
    lines.extend(
        [
            "## Experiment Summary",
            "",
            "| # | Status | Duration | val_bpb | Error |",
            "|---|--------|----------|---------|-------|",
        ]
    )

    best_bpb: float | None = None
    best_idx: int | None = None

    for r in results:
        status = "✅" if r.exit_code == 0 else "❌"
        bpb_str = f"{r.val_bpb:.4f}" if r.val_bpb is not None else "—"
        error_str = r.error or ""
        lines.append(
            f"| {r.experiment_id} | {status} | {r.duration:.1f}s "
            f"| {bpb_str} | {error_str} |"
        )
        if r.val_bpb is not None and (best_bpb is None or r.val_bpb < best_bpb):
            best_bpb = r.val_bpb
            best_idx = r.experiment_id

    lines.append("")

    if best_bpb is not None:
        lines.extend(
            [
                "## Best Result",
                "",
                f"**Experiment {best_idx}** achieved val_bpb = **{best_bpb:.4f}**",
                "",
            ]
        )

    # Aggregate stats
    total_time = sum(r.duration for r in results)
    successes = sum(1 for r in results if r.exit_code == 0)
    lines.extend(
        [
            "## Stats",
            "",
            f"- Total experiments: {len(results)}",
            f"- Successful: {successes}",
            f"- Failed: {len(results) - successes}",
            f"- Total wall time: {total_time:.1f}s ({total_time / 60:.1f} min)",
            "",
        ]
    )

    return "\n".join(lines)


def save_report(
    report: str,
    output_dir: Path,
    run_id: str,
) -> Path:
    """Write a report to disk and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"report-{run_id}.md"
    path.write_text(report)
    return path
