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
from .ingest import Ingester
from .progress import find_progress_files, read_progress
from .report import generate_report, save_report
from .runner import Runner

app = typer.Typer(
    name="mat",
    help="Multi-source data collection and autonomous LLM training.",
    no_args_is_help=True,
)
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
) -> None:
    """Run the full pipeline: ingest sources → prepare corpus → train."""
    cfg = load_config(config)

    if max_experiments is not None:
        cfg.training.max_experiments = max_experiments
    if output_dir is not None:
        cfg.training.output_dir = str(output_dir)

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
    runner = Runner(cfg.autoresearch, cfg.training, console, name=name)
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
) -> None:
    """List configured data sources."""
    cfg = load_config(config)

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
) -> None:
    """Ingest data sources without training (useful for inspection)."""
    cfg = load_config(config)

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
