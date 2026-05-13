"""AWS SES failure notifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .base import FailureEvent, Notifier

log = logging.getLogger(__name__)


@dataclass
class SESConfig:
    """Settings for the AWS SES notifier."""

    from_email: str
    to_emails: list[str] = field(default_factory=list)
    region: str = "us-east-1"
    subject_prefix: str = "[MultiAgentTrainer]"


class SESNotifier(Notifier):
    """Sends failure alerts via AWS Simple Email Service."""

    def __init__(self, cfg: SESConfig) -> None:
        self.cfg = cfg

    def notify_failure(self, event: FailureEvent) -> None:
        try:
            import boto3

            ses = boto3.client("ses", region_name=self.cfg.region)
            subject = f"{self.cfg.subject_prefix} Training failed ({event.backend})"
            ses.send_email(
                Source=self.cfg.from_email,
                Destination={"ToAddresses": self.cfg.to_emails},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": self._format_body(event)}},
                },
            )
        except Exception:
            log.exception("SESNotifier failed to send failure notification")

    def _format_body(self, event: FailureEvent) -> str:
        lines = [
            "A training or fine-tuning job has failed.",
            "",
            f"Backend:   {event.backend}",
            f"Run ID:    {event.run_id}",
        ]
        if event.model:
            lines.append(f"Model:     {event.model}")
        if event.timestamp:
            lines.append(f"Timestamp: {event.timestamp}")
        for key, val in event.details.items():
            lines.append(f"{key.capitalize()}: {val}")
        lines += ["", "Error:", f"  {event.error}"]
        return "\n".join(lines)
