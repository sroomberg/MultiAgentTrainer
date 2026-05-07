"""Run autonomous training experiments via autoresearch."""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import git as gitpython
from rich.console import Console

from .config import AutoresearchConfig, TrainingConfig
from .executor import Executor, build_executor
from .progress import new_progress, write_progress

log = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single autonomous training experiment."""

    experiment_id: int
    exit_code: int
    duration: float
    val_bpb: float | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class Runner:
    """Set up autoresearch and run autonomous training experiments."""

    def __init__(
        self,
        autoresearch_cfg: AutoresearchConfig,
        training_cfg: TrainingConfig,
        console: Console | None = None,
        executor: Executor | None = None,
        name: str = "",
    ) -> None:
        self.ar_cfg = autoresearch_cfg
        self.tr_cfg = training_cfg
        self.console = console or Console()
        self.run_id = uuid.uuid4().hex[:8]
        self.executor: Executor = executor or build_executor(training_cfg.execution)
        self.name = name or self.run_id

    def setup_workspace(self, corpus_path: Path | None = None) -> Path:
        """Clone or copy autoresearch into a working directory.

        If *corpus_path* is provided, inject it as custom training data.

        Returns the workspace path.
        """
        output_dir = Path(self.tr_cfg.output_dir).resolve()
        workspace = output_dir / f"run-{self.run_id}"
        workspace.mkdir(parents=True, exist_ok=True)

        ar_path = Path(self.ar_cfg.repo).expanduser()
        if ar_path.is_dir():
            # Local autoresearch checkout — copy it.
            shutil.copytree(ar_path, workspace / "autoresearch", dirs_exist_ok=True)
        else:
            # Remote — clone.
            self.console.print(
                f"  [dim]Cloning autoresearch ({self.ar_cfg.branch})…[/dim]"
            )
            gitpython.Repo.clone_from(
                self.ar_cfg.repo,
                str(workspace / "autoresearch"),
                branch=self.ar_cfg.branch,
                depth=1,
            )

        ar_dir = workspace / "autoresearch"

        # Inject custom corpus if provided.
        if corpus_path and corpus_path.exists():
            data_dir = ar_dir / "data"
            data_dir.mkdir(exist_ok=True)
            shutil.copy2(corpus_path, data_dir / "custom_corpus.txt")
            self.console.print("  [dim]Injected custom corpus[/dim]")

        # Override program.md if configured.
        if self.ar_cfg.program_md:
            src = Path(self.ar_cfg.program_md).expanduser().resolve()
            if src.exists():
                shutil.copy2(src, ar_dir / "program.md")
                self.console.print("  [dim]Overrode program.md[/dim]")

        return workspace

    async def run_experiments(
        self,
        workspace: Path,
    ) -> list[ExperimentResult]:
        """Run up to ``max_experiments`` autonomous experiments."""
        ar_dir = workspace / "autoresearch"
        results: list[ExperimentResult] = []

        output_dir = Path(self.tr_cfg.output_dir).resolve()
        progress_path = output_dir / f"progress-{self.run_id}.json"
        progress = new_progress(self.run_id, self.name, self.tr_cfg.max_experiments)
        write_progress(progress_path, progress)

        self.console.print(
            f"\n[bold]Starting run [cyan]{self.run_id}[/cyan] "
            f"— max {self.tr_cfg.max_experiments} experiments[/bold]\n"
        )

        # Upload workspace to the execution target (no-op for LocalExecutor).
        exec_cfg = self.tr_cfg.execution
        suffix = f"/{self.run_id}/autoresearch"
        if exec_cfg.type == "ssh":
            remote_ar_dir = exec_cfg.remote_dir.rstrip("/") + suffix
        elif exec_cfg.type == "docker":
            remote_ar_dir = exec_cfg.container_dir.rstrip("/") + suffix
        else:
            remote_ar_dir = str(ar_dir)

        if exec_cfg.type in ("ssh", "docker"):
            self.console.print(f"  [dim]Uploading workspace → {remote_ar_dir}…[/dim]")
            await self.executor.upload(ar_dir, remote_ar_dir)
            self.console.print("  [dim]Upload complete[/dim]")

        prompt = (
            "Hi have a look at program.md and let's kick off a new experiment! "
            "let's do the setup first."
        )

        for i in range(1, self.tr_cfg.max_experiments + 1):
            self.console.print(
                f"  [bold]Experiment {i}/{self.tr_cfg.max_experiments}[/bold]"
            )
            result = await self._run_one(remote_ar_dir, prompt, i)
            results.append(result)

            icon = "✅" if result.exit_code == 0 else "❌"
            bpb = f" val_bpb={result.val_bpb:.4f}" if result.val_bpb else ""
            self.console.print(
                f"  {icon} experiment {i} — {result.duration:.1f}s, "
                f"exit {result.exit_code}{bpb}"
                + (f" ({result.error})" if result.error else "")
            )

            progress.experiments.append(
                {
                    "experiment_id": result.experiment_id,
                    "exit_code": result.exit_code,
                    "duration": result.duration,
                    "val_bpb": result.val_bpb,
                    "error": result.error,
                }
            )
            write_progress(progress_path, progress)

            # After the first experiment, subsequent prompts continue iteration.
            prompt = (
                "Great, let's kick off the next experiment. "
                "Review the results so far and try something new."
            )

        progress.status = "done"
        write_progress(progress_path, progress)

        return results

    async def _run_one(
        self,
        ar_dir: str,
        prompt: str,
        experiment_id: int,
    ) -> ExperimentResult:
        """Run a single experiment by invoking the agent command via the executor."""
        start = time.monotonic()

        cmd = self.tr_cfg.agent_command
        if "{prompt}" in cmd:
            cmd = cmd.replace("{prompt}", shlex.quote(prompt))

        result = await self.executor.run(
            cmd=cmd,
            cwd=ar_dir,
            timeout=self.ar_cfg.train_time + 120,
        )

        val_bpb = self._parse_val_bpb(result.stdout + result.stderr)

        return ExperimentResult(
            experiment_id=experiment_id,
            exit_code=result.exit_code,
            duration=time.monotonic() - start,
            val_bpb=val_bpb,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
        )

    @staticmethod
    def _parse_val_bpb(output: str) -> float | None:
        """Extract the last val_bpb value from agent output."""
        matches = re.findall(r"val_bpb[=:\s]+([0-9]+\.?[0-9]*)", output)
        if matches:
            return float(matches[-1])
        return None
