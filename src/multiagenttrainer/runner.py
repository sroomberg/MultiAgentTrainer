"""Run autonomous training experiments via autoresearch."""

import asyncio
import logging
import re
import shlex
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import git as gitpython
from rich.console import Console

from .config import AutoresearchConfig, ExecutionConfig, MachineConfig, TrainingConfig
from .executor import Executor, build_executor, build_executor_for_machine
from .notifications.base import FailureEvent, Notifier
from .progress import RunProgress, new_progress, write_progress

log = logging.getLogger(__name__)

_AGENT_GRACE_SECONDS = 120

_SETUP_PROMPT = (
    "Hi have a look at program.md and let's kick off a new experiment! "
    "let's do the setup first."
)
_CONTINUE_PROMPT = (
    "Great, let's kick off the next experiment. "
    "Review the results so far and try something new."
)


@dataclass
class ExperimentResult:
    """Result of a single autonomous training experiment."""

    experiment_id: int
    exit_code: int
    duration: float
    val_bpb: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    machine: Optional[str] = None


class Runner:
    """Set up autoresearch and run autonomous training experiments."""

    def __init__(
        self,
        autoresearch_cfg: AutoresearchConfig,
        training_cfg: TrainingConfig,
        console: Optional[Console] = None,
        executor: Optional[Executor] = None,
        name: str = "",
        notifier: Optional[Notifier] = None,
        machines: Optional[list[MachineConfig]] = None,
    ) -> None:
        self.ar_cfg = autoresearch_cfg
        self.tr_cfg = training_cfg
        self.console = console or Console()
        self.run_id = uuid.uuid4().hex[:8]
        self.machines: list[MachineConfig] = machines or []
        self.executor: Executor = executor or build_executor(training_cfg.execution)
        self.name = name or self.run_id
        self.notifier = notifier

    def setup_workspace(self, corpus_path: Optional[Path] = None) -> Path:
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
        """Run up to ``max_experiments`` experiments, distributing across machines
        if multiple are configured."""
        if self.machines:
            return await self._run_multi_machine(workspace)
        return await self._run_single_machine(workspace)

    # ------------------------------------------------------------------
    # Single-machine path (original behaviour, fully preserved)
    # ------------------------------------------------------------------

    async def _run_single_machine(self, workspace: Path) -> list[ExperimentResult]:
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

        remote_ar_dir = _remote_ar_dir(self.tr_cfg.execution, self.run_id, ar_dir)

        if self.tr_cfg.execution.type in ("ssh", "docker"):
            self.console.print(f"  [dim]Uploading workspace → {remote_ar_dir}…[/dim]")
            await self.executor.upload(ar_dir, remote_ar_dir)
            self.console.print("  [dim]Upload complete[/dim]")

        prompt = _SETUP_PROMPT

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

            if result.exit_code != 0:
                self._notify_experiment_failure(result)

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

            prompt = _CONTINUE_PROMPT

        progress.status = "done"
        write_progress(progress_path, progress)

        return results

    # ------------------------------------------------------------------
    # Multi-machine path
    # ------------------------------------------------------------------

    async def _run_multi_machine(self, workspace: Path) -> list[ExperimentResult]:
        """Distribute experiments across all configured machines concurrently."""
        ar_dir = workspace / "autoresearch"
        output_dir = Path(self.tr_cfg.output_dir).resolve()
        progress_path = output_dir / f"progress-{self.run_id}.json"
        progress = new_progress(self.run_id, self.name, self.tr_cfg.max_experiments)
        write_progress(progress_path, progress)

        n = len(self.machines)
        total = self.tr_cfg.max_experiments
        self.console.print(
            f"\n[bold]Starting multi-machine run [cyan]{self.run_id}[/cyan] "
            f"— {n} machines, max {total} experiments[/bold]\n"
        )

        # Divide experiment IDs evenly; first (total % n) machines get one extra.
        counts = [total // n + (1 if i < total % n else 0) for i in range(n)]

        lock = asyncio.Lock()
        results: list[ExperimentResult] = []

        batches = []
        offset = 0
        for machine, count in zip(self.machines, counts, strict=True):
            exp_ids = range(offset + 1, offset + count + 1)
            batches.append(
                self._run_machine_batch(
                    machine, ar_dir, exp_ids, results, progress, progress_path, lock
                )
            )
            offset += count

        await asyncio.gather(*batches)

        progress.status = "done"
        write_progress(progress_path, progress)

        results.sort(key=lambda r: r.experiment_id)
        return results

    async def _run_machine_batch(
        self,
        machine: MachineConfig,
        ar_dir: Path,
        experiment_ids: range,
        results: list[ExperimentResult],
        progress: RunProgress,
        progress_path: Path,
        lock: asyncio.Lock,
    ) -> None:
        executor = build_executor_for_machine(machine)
        remote_ar_dir = _remote_ar_dir(machine.execution, self.run_id, ar_dir)

        if machine.execution.type in ("ssh", "docker"):
            self.console.print(
                f"  [dim][{machine.name}] Uploading workspace → {remote_ar_dir}…[/dim]"
            )
            await executor.upload(ar_dir, remote_ar_dir)

        agent_command = machine.agent_command or self.tr_cfg.agent_command

        prompt = _SETUP_PROMPT

        for exp_id in experiment_ids:
            result = await self._run_one(
                remote_ar_dir, prompt, exp_id, executor, agent_command
            )
            result.machine = machine.name

            icon = "✅" if result.exit_code == 0 else "❌"
            bpb = f" val_bpb={result.val_bpb:.4f}" if result.val_bpb else ""
            self.console.print(
                f"  [{machine.name}] {icon} experiment {exp_id} — "
                f"{result.duration:.1f}s, exit {result.exit_code}{bpb}"
                + (f" ({result.error})" if result.error else "")
            )

            if result.exit_code != 0:
                self._notify_experiment_failure(result, machine.name)

            async with lock:
                results.append(result)
                progress.experiments.append(
                    {
                        "experiment_id": result.experiment_id,
                        "exit_code": result.exit_code,
                        "duration": result.duration,
                        "val_bpb": result.val_bpb,
                        "error": result.error,
                        "machine": machine.name,
                    }
                )
                write_progress(progress_path, progress)

            prompt = _CONTINUE_PROMPT

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _run_one(
        self,
        ar_dir: str,
        prompt: str,
        experiment_id: int,
        executor: Optional[Executor] = None,
        agent_command: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a single experiment by invoking the agent command via the executor."""
        start = time.monotonic()
        executor = executor or self.executor
        cmd = agent_command or self.tr_cfg.agent_command
        if "{prompt}" in cmd:
            cmd = cmd.replace("{prompt}", shlex.quote(prompt))

        result = await executor.run(
            cmd=cmd,
            cwd=ar_dir,
            timeout=self.ar_cfg.train_time + _AGENT_GRACE_SECONDS,
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

    def _notify_experiment_failure(
        self,
        result: ExperimentResult,
        machine_name: Optional[str] = None,
    ) -> None:
        if not self.notifier:
            return
        details: dict = {"experiment": str(result.experiment_id)}
        if machine_name is not None:
            details["machine"] = machine_name
        self.notifier.notify_failure(
            FailureEvent(
                run_id=self.run_id,
                backend="runner",
                error=result.error or f"exit code {result.exit_code}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=details,
            )
        )

    @staticmethod
    def _parse_val_bpb(output: str) -> Optional[float]:
        """Extract the last val_bpb value from agent output."""
        matches = re.findall(r"val_bpb[=:\s]+([0-9]+\.?[0-9]*)", output)
        if matches:
            return float(matches[-1])
        return None


def _remote_ar_dir(exec_cfg: ExecutionConfig, run_id: str, ar_dir: Path) -> str:
    """Derive the remote working directory for a given execution config."""
    suffix = f"/{run_id}/autoresearch"
    if exec_cfg.type == "ssh":
        return exec_cfg.remote_dir.rstrip("/") + suffix
    if exec_cfg.type == "docker":
        return exec_cfg.container_dir.rstrip("/") + suffix
    return str(ar_dir)
