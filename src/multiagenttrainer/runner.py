"""Run autonomous training experiments via autoresearch."""

from __future__ import annotations

import asyncio
import logging
import os
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
    ) -> None:
        self.ar_cfg = autoresearch_cfg
        self.tr_cfg = training_cfg
        self.console = console or Console()
        self.run_id = uuid.uuid4().hex[:8]

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

        self.console.print(
            f"\n[bold]Starting run [cyan]{self.run_id}[/cyan] "
            f"— max {self.tr_cfg.max_experiments} experiments[/bold]\n"
        )

        prompt = (
            "Hi have a look at program.md and let's kick off a new experiment! "
            "let's do the setup first."
        )

        for i in range(1, self.tr_cfg.max_experiments + 1):
            self.console.print(
                f"  [bold]Experiment {i}/{self.tr_cfg.max_experiments}[/bold]"
            )
            result = await self._run_one(ar_dir, prompt, i)
            results.append(result)

            icon = "✅" if result.exit_code == 0 else "❌"
            bpb = f" val_bpb={result.val_bpb:.4f}" if result.val_bpb else ""
            self.console.print(
                f"  {icon} experiment {i} — {result.duration:.1f}s, "
                f"exit {result.exit_code}{bpb}"
                + (f" ({result.error})" if result.error else "")
            )

            # After the first experiment, subsequent prompts continue iteration.
            prompt = (
                "Great, let's kick off the next experiment. "
                "Review the results so far and try something new."
            )

        return results

    async def _run_one(
        self,
        ar_dir: Path,
        prompt: str,
        experiment_id: int,
    ) -> ExperimentResult:
        """Run a single experiment by invoking the agent command."""
        start = time.monotonic()

        cmd = self.tr_cfg.agent_command
        if "{prompt}" in cmd:
            cmd = cmd.replace("{prompt}", shlex.quote(prompt))

        env = os.environ.copy()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ar_dir,
                env=env,
                start_new_session=True,
            )

            try:
                stdout_raw, stderr_raw = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.ar_cfg.train_time + 120,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ExperimentResult(
                    experiment_id=experiment_id,
                    exit_code=-1,
                    duration=time.monotonic() - start,
                    error="Timed out",
                )

            stdout_text = stdout_raw.decode("utf-8", errors="replace")
            stderr_text = stderr_raw.decode("utf-8", errors="replace")
            stdout_lines = stdout_text.splitlines()
            stderr_lines = stderr_text.splitlines()

            val_bpb = self._parse_val_bpb(stdout_text + stderr_text)

            return ExperimentResult(
                experiment_id=experiment_id,
                exit_code=proc.returncode or 0,
                duration=time.monotonic() - start,
                val_bpb=val_bpb,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        except Exception as e:
            return ExperimentResult(
                experiment_id=experiment_id,
                exit_code=-1,
                duration=time.monotonic() - start,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                error=str(e),
            )

    @staticmethod
    def _parse_val_bpb(output: str) -> float | None:
        """Extract the last val_bpb value from agent output."""
        matches = re.findall(r"val_bpb[=:\s]+([0-9]+\.?[0-9]*)", output)
        if matches:
            return float(matches[-1])
        return None
