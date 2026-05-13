"""Notification backends for training/fine-tuning failure alerts."""

from typing import Optional

from .base import FailureEvent, Notifier
from .ses import SESConfig, SESNotifier

__all__ = ["FailureEvent", "Notifier", "SESConfig", "SESNotifier", "build_notifier"]


def build_notifier(ses_cfg: Optional[SESConfig]) -> Optional[Notifier]:
    """Return an SESNotifier if configured, else None."""
    if ses_cfg is None:
        return None
    return SESNotifier(ses_cfg)
