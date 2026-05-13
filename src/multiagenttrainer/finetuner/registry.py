"""Factory for creating FineTuner instances from config."""

from pathlib import Path
from typing import Optional

from rich.console import Console

from ..notifications.base import Notifier
from .config import FineTunerConfig
from .finetuner import BedrockFineTuner, FineTuner, OpenSourceFineTuner
from .multi import MultiTargetFineTuner


def create_fine_tuner(
    cfg: FineTunerConfig,
    console: Console,
    notifier: Optional[Notifier] = None,
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


def create_multi_target_fine_tuner(
    cfg: FineTunerConfig,
    console: Console,
    notifier: Optional[Notifier] = None,
) -> MultiTargetFineTuner:
    """Return a MultiTargetFineTuner for all targets in the config."""
    if not cfg.targets:
        raise ValueError(
            "create_multi_target_fine_tuner requires at least one entry in "
            "finetuner.targets"
        )
    return MultiTargetFineTuner(cfg, console, notifier)
