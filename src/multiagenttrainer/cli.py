"""CLI entry point."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .config import load_config
from .finetuner import FineTuneJob, create_fine_tuner
from .ingest import Ingester
from .notifications import build_notifier
from .progress import find_progress_files, read_progress
from .report import generate_report, save_report
from .runner import Runner
from .sources import GitHubRepoListSource

app = typer.Typer(
    name="mat",
    help="Multi-source data collection and autonomous LLM training.",
    no_args_is_help=True,
)
finetune_app = typer.Typer(
    name="finetune",
    help="Fine-tune models on collected corpus data.",
    no_args_is_help=True,
)
app.add_typer(finetune_app, name="finetune")
console = Console()


@app.command()
def train(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
    max_experiments: Annotated[
        int | None,
        typer.Option("--max-experiments", "-n", help="Override max experiments"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override output directory"),
    ] = None,
    name: Annotated[
        str,
        typer.Option("--name", help="Label for this run (shown in mat watch)"),
    ] = "",
    repos: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", help="Repo URL or owner/repo. Repeatable."),
    ] = None,
) -> None:
    """Run the full pipeline: ingest sources → prepare corpus → train."""
    cfg = load_config(config)

    if max_experiments is not None:
        cfg.training.max_experiments = max_experiments
    if output_dir is not None:
        cfg.training.output_dir = str(output_dir)
    if repos:
        cfg.sources.append(GitHubRepoListSource(repos=repos))

    if not cfg.sources:
        console.print(
            "[yellow]No data sources configured."
            " Training with autoresearch defaults.[/yellow]"
        )

    # Ingest
    staging = Path(cfg.training.output_dir).resolve() / ".staging"
    ingester = Ingester(cfg.sources, staging, console)

    corpus_path: Path | None = None
    if cfg.sources:
        console.print("\n[bold]Ingesting data sources…[/bold]")
        ingester.fetch_all()
        corpus_path = staging / "corpus.txt"
        count = ingester.build_corpus(corpus_path)
        if count == 0:
            console.print("[yellow]Warning: corpus is empty[/yellow]")
            corpus_path = None

    # Setup and run
    notifier = build_notifier(cfg.notifications.ses if cfg.notifications else None)
    runner = Runner(cfg.autoresearch, cfg.training, console, name=name, notifier=notifier)
    workspace = runner.setup_workspace(corpus_path)

    results = asyncio.run(runner.run_experiments(workspace))

    # Report
    report = generate_report(runner.run_id, cfg, results)
    report_path = save_report(
        report, Path(cfg.training.output_dir).resolve(), runner.run_id
    )
    console.print(f"\n[bold]Report:[/bold] {report_path}")


@app.command("sources")
def list_sources(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
    repos: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", help="Repo URL or owner/repo. Repeatable."),
    ] = None,
) -> None:
    """List configured data sources."""
    cfg = load_config(config)
    if repos:
        cfg.sources.append(GitHubRepoListSource(repos=repos))

    if not cfg.sources:
        console.print("[yellow]No data sources configured.[/yellow]")
        return

    console.print("[bold]Configured data sources:[/bold]\n")
    for i, src in enumerate(cfg.sources, 1):
        console.print(f"  {i}. {src.describe()}")
    console.print()


@app.command("ingest")
def ingest_cmd(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Corpus output path"),
    ] = None,
    repos: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", help="Repo URL or owner/repo. Repeatable."),
    ] = None,
) -> None:
    """Ingest data sources without training (useful for inspection)."""
    cfg = load_config(config)
    if repos:
        cfg.sources.append(GitHubRepoListSource(repos=repos))

    if not cfg.sources:
        console.print("[red]No data sources configured.[/red]")
        raise typer.Exit(1)

    staging = Path(cfg.training.output_dir).resolve() / ".staging"
    ingester = Ingester(cfg.sources, staging, console)

    console.print("\n[bold]Ingesting data sources…[/bold]")
    ingester.fetch_all()

    corpus_path = output or (staging / "corpus.txt")
    count = ingester.build_corpus(corpus_path)
    console.print(f"\n[bold]Done:[/bold] {count} files ingested → {corpus_path}")


@app.command("watch")
def watch(
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Training runs directory"),
    ] = None,
    interval: Annotated[
        float,
        typer.Option("--interval", "-i", help="Refresh interval in seconds"),
    ] = 2.0,
) -> None:
    """Live dashboard showing progress across all active training runs."""
    runs_dir = (output_dir or Path("./training-runs")).resolve()

    def _fmt(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "—"

    def _build_table() -> Table:
        table = Table(box=None, pad_edge=False, show_header=True)
        table.add_column("Name", style="bold")
        table.add_column("Run ID", style="dim")
        table.add_column("Progress", justify="right")
        table.add_column("Best val_bpb", justify="right")
        table.add_column("Last val_bpb", justify="right")
        table.add_column("Status")

        paths = find_progress_files(runs_dir) if runs_dir.exists() else []
        if not paths:
            table.add_row("[dim]No active runs found.[/dim]", "", "", "", "", "")
            return table

        for path in paths:
            p = read_progress(path)
            if p is None:
                continue
            progress_str = f"{p.current_experiment} / {p.max_experiments}"
            status_str = (
                "[green]done[/green]"
                if p.status == "done"
                else "[yellow]running…[/yellow]"
            )
            table.add_row(
                p.name,
                p.run_id,
                progress_str,
                _fmt(p.best_val_bpb),
                _fmt(p.last_val_bpb),
                status_str,
            )
        return table

    try:
        with Live(_build_table(), refresh_per_second=1, screen=False) as live:
            while True:
                time.sleep(interval)
                live.update(_build_table())
                paths = find_progress_files(runs_dir) if runs_dir.exists() else []
                if paths and all(
                    (p := read_progress(path)) is not None and p.status != "running"
                    for path in paths
                ):
                    break
    except KeyboardInterrupt:
        pass


@app.command("status")
def status(
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Training runs directory"),
    ] = None,
) -> None:
    """Show past training runs."""
    runs_dir = (output_dir or Path("./training-runs")).resolve()

    if not runs_dir.exists():
        console.print("[dim]No training runs found.[/dim]")
        return

    reports = sorted(runs_dir.glob("report-*.md"))
    if not reports:
        console.print("[dim]No training reports found.[/dim]")
        return

    console.print("[bold]Training runs:[/bold]\n")
    for report in reports:
        run_id = report.stem.removeprefix("report-")
        size = report.stat().st_size
        console.print(f"  [bold]{run_id}[/bold]  {report.name} ({size} bytes)")
    console.print()


# ---------------------------------------------------------------------------
# mat finetune …
# ---------------------------------------------------------------------------


def _require_finetuner_config(config: Path | None) -> tuple:
    cfg = load_config(config)
    if cfg.finetuner is None:
        console.print(
            "[red]No [bold]finetuner[/bold] section found in config. "
            "Add one to your multiagenttrainer.yaml.[/red]"
        )
        raise typer.Exit(1)
    notifier = build_notifier(cfg.notifications.ses if cfg.notifications else None)
    return cfg, create_fine_tuner(cfg.finetuner, console, notifier)


def _print_job(job: FineTuneJob) -> None:
    console.print(f"  [bold]Job ID:[/bold]  {job.job_id}")
    console.print(f"  [bold]Backend:[/bold] {job.backend}")
    console.print(f"  [bold]Model:[/bold]   {job.model}")
    console.print(f"  [bold]Status:[/bold]  {job.status}")
    console.print(f"  [bold]Created:[/bold] {job.created_at[:19]}")
    if job.output_model:
        console.print(f"  [bold]Output:[/bold]  {job.output_model}")
    if job.metrics:
        console.print("  [bold]Metrics:[/bold]")
        for k, v in job.metrics.items():
            console.print(f"    {k}: {v:.4f}")
    if job.error:
        console.print(f"  [red]Error:[/red] {job.error}")


@finetune_app.command("start")
def finetune_start(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", help="Corpus file (defaults to last ingest output)"),
    ] = None,
    name: Annotated[
        str,
        typer.Option("--name", help="Human-readable label for this job"),
    ] = "finetune",
) -> None:
    """Prepare dataset from corpus and start a fine-tuning job."""
    cfg, tuner = _require_finetuner_config(config)

    if corpus is None:
        default_corpus = (
            Path(cfg.training.output_dir).resolve() / ".staging" / "corpus.txt"
        )
        if default_corpus.exists():
            corpus = default_corpus
        else:
            console.print(
                "[red]No corpus found. Run [bold]mat ingest[/bold] first "
                "or pass [bold]--corpus[/bold].[/red]"
            )
            raise typer.Exit(1)

    console.print(f"\n[bold]Backend:[/bold] {tuner.describe()}")
    console.print(f"[bold]Corpus:[/bold]  {corpus}\n")

    console.print("[bold]Preparing dataset…[/bold]")
    dataset = tuner.prepare_dataset(corpus)

    console.print("\n[bold]Starting fine-tuning job…[/bold]")
    job = tuner.start_job(dataset, name)

    console.print()
    _print_job(job)


@finetune_app.command("status")
def finetune_status(
    job_id: Annotated[str, typer.Argument(help="Job ID to check")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
) -> None:
    """Check the status of a fine-tuning job."""
    _, tuner = _require_finetuner_config(config)
    job = tuner.get_status(job_id)
    console.print()
    _print_job(job)


@finetune_app.command("cancel")
def finetune_cancel(
    job_id: Annotated[str, typer.Argument(help="Job ID to cancel")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
) -> None:
    """Cancel a running fine-tuning job."""
    _, tuner = _require_finetuner_config(config)
    tuner.cancel_job(job_id)
    console.print(f"[green]Cancelled:[/green] {job_id}")


@finetune_app.command("list")
def finetune_list(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML"),
    ] = None,
) -> None:
    """List all fine-tuning jobs."""
    cfg = load_config(config)
    if cfg.finetuner is None:
        console.print("[dim]No finetuner configured.[/dim]")
        return

    jobs = FineTuneJob.list_all(Path(cfg.finetuner.jobs_dir))
    if not jobs:
        console.print("[dim]No fine-tuning jobs found.[/dim]")
        return

    table = Table(box=None, pad_edge=False, show_header=True)
    table.add_column("Job ID")
    table.add_column("Backend")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Output")

    status_styles = {
        "running": "[yellow]running[/yellow]",
        "completed": "[green]completed[/green]",
        "failed": "[red]failed[/red]",
        "cancelled": "[dim]cancelled[/dim]",
    }

    for job in jobs:
        short_id = job.job_id if len(job.job_id) <= 48 else job.job_id[:45] + "…"
        table.add_row(
            short_id,
            job.backend,
            job.model,
            status_styles.get(job.status, job.status),
            job.created_at[:19],
            job.output_model or "—",
        )

    console.print(table)
