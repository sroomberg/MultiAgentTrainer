"""Notification abstractions for training/fine-tuning failure alerts."""

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FailureEvent:
    """Describes a training or fine-tuning failure."""

    run_id: str
    backend: str  # "opensource", "bedrock", "runner"
    error: str
    model: Optional[str] = None
    timestamp: str = ""
    details: dict[str, str] = field(default_factory=dict)


class Notifier(abc.ABC):
    """Abstract base for failure notification backends."""

    @abc.abstractmethod
    def notify_failure(self, event: FailureEvent) -> None:
        """Send a failure notification. Implementations must not raise."""
