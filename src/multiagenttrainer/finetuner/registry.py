"""Factory for creating FineTuner instances from config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from .config import FineTunerConfig
from .finetuner import BedrockFineTuner, FineTuner, OpenSourceFineTuner

if TYPE_CHECKING:
    from ..notifications.base import Notifier


def create_fine_tuner(
    cfg: FineTunerConfig,
    console: Console,
    notifier: Notifier | None = None,
) -> FineTuner:
    """Return the appropriate FineTuner for the configured backend."""
    jobs_dir = Path(cfg.jobs_dir)
    if cfg.backend == "opensource":
        return OpenSourceFineTuner(cfg.opensource, jobs_dir, console, notifier)
    if cfg.backend == "bedrock":
        return BedrockFineTuner(cfg.bedrock, jobs_dir, console, notifier)
    raise ValueError(
        f"Unknown fine-tuning backend: {cfg.backend!r}. "
        "Expected one of: opensource, bedrock"
    )
