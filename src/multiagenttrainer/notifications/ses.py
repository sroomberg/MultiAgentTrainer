"""AWS SES failure notifier."""

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

    _SUBJECT = "Training job failed"
    _BODY = (
        "A MultiAgentTrainer training or fine-tuning job has failed.\n\n"
        "Please check your training logs for details."
    )

    def notify_failure(self, event: FailureEvent) -> None:
        try:
            import boto3

            ses = boto3.client("ses", region_name=self.cfg.region)
            ses.send_email(
                Source=self.cfg.from_email,
                Destination={"ToAddresses": self.cfg.to_emails},
                Message={
                    "Subject": {"Data": f"{self.cfg.subject_prefix} {self._SUBJECT}"},
                    "Body": {"Text": {"Data": self._BODY}},
                },
            )
        except Exception:
            log.exception("SESNotifier failed to send failure notification")
