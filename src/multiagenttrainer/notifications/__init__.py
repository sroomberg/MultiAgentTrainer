"""Notification backends for training/fine-tuning failure alerts."""

from .base import FailureEvent, Notifier
from .ses import SESConfig, SESNotifier

__all__ = ["FailureEvent", "Notifier", "SESConfig", "SESNotifier", "build_notifier"]


def build_notifier(ses_cfg: SESConfig | None) -> Notifier | None:
    """Return an SESNotifier if configured, else None."""
    if ses_cfg is None:
        return None
    return SESNotifier(ses_cfg)
