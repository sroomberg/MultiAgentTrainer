"""Multi-target fine-tuning orchestrator."""

from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from .config import FineTunerConfig, FineTuneTargetConfig
from .finetuner import BedrockFineTuner, FineTuneJob, FineTuner, OpenSourceFineTuner

if TYPE_CHECKING:
    from ..notifications.base import Notifier

log = logging.getLogger(__name__)


class MultiTargetFineTuner:
    """Concurrent fine-tuning orchestrator across multiple (model, machine) targets.

    Each target in ``FineTunerConfig.targets`` specifies a model ID and
    optional per-target overrides. Jobs are submitted concurrently via a
    thread pool so that fast-returning backends (Bedrock) are not blocked
    by long-running in-process training (OpenSource).

    Note: running multiple OpenSource targets against the same GPU will
    contend for device memory. Assign each OpenSource target to a distinct
    machine (different GPU or host) to avoid OOM errors.
    """

    def __init__(
        self,
        base_cfg: FineTunerConfig,
        console: Console,
        notifier: Notifier | None = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.console = console
        self.notifier = notifier

    def describe_targets(self) -> list[str]:
        return [
            f"{t.name} — {t.model_id} "
            f"({t.backend or self.base_cfg.backend})"
            + (f" on {t.machine}" if t.machine else "")
            for t in self.base_cfg.targets
        ]

    def start_all(self, corpus_path: Path, job_name: str) -> list[FineTuneJob]:
        """Prepare datasets and start all targets concurrently.

        Returns one FineTuneJob per target in the same order as
        ``base_cfg.targets``. Failed targets are logged but do not
        prevent other targets from completing.
        """
        targets = self.base_cfg.targets
        tuners = [self._build_tuner(t) for t in targets]

        jobs_by_name: dict[str, FineTuneJob] = {}

        def _run(target: FineTuneTargetConfig, tuner: FineTuner) -> FineTuneJob:
            self.console.print(
                f"  [dim]Preparing dataset for [bold]{target.name}[/bold]…[/dim]"
            )
            dataset = tuner.prepare_dataset(corpus_path)
            self.console.print(
                f"  [dim]Starting job for [bold]{target.name}[/bold]…[/dim]"
            )
            return tuner.start_job(dataset, job_name or target.name)

        with ThreadPoolExecutor(max_workers=len(tuners)) as pool:
            future_to_target = {
                pool.submit(_run, t, tuner): t
                for t, tuner in zip(targets, tuners, strict=True)
            }
            for future in as_completed(future_to_target):
                target = future_to_target[future]
                try:
                    jobs_by_name[target.name] = future.result()
                except Exception as exc:
                    log.error(
                        "Fine-tuning target %r failed to start: %s", target.name, exc
                    )

        # Return in original target order, omitting any that failed to start.
        return [jobs_by_name[t.name] for t in targets if t.name in jobs_by_name]

    def _build_tuner(self, target: FineTuneTargetConfig) -> FineTuner:
        backend = target.backend or self.base_cfg.backend
        # Each target gets its own jobs subdirectory to avoid file collisions.
        jobs_dir = Path(self.base_cfg.jobs_dir) / target.name

        if backend == "bedrock":
            overrides: dict = {"base_model_id": target.model_id}
            if target.customization_type:
                overrides["customization_type"] = target.customization_type
            br_cfg = dataclasses.replace(self.base_cfg.bedrock, **overrides)
            return BedrockFineTuner(br_cfg, jobs_dir, self.console, self.notifier)

        os_overrides: dict = {"model_id": target.model_id}
        if target.num_epochs is not None:
            os_overrides["num_epochs"] = target.num_epochs
        if target.batch_size is not None:
            os_overrides["batch_size"] = target.batch_size
        if target.lora_r is not None:
            os_overrides["lora_r"] = target.lora_r
        os_cfg = dataclasses.replace(self.base_cfg.opensource, **os_overrides)
        return OpenSourceFineTuner(os_cfg, jobs_dir, self.console, self.notifier)
