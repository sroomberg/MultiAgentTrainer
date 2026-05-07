"""Progress tracking for active training runs.

Each run writes a ``progress-{run_id}.json`` file to the output directory
after every experiment so that ``mat watch`` can display a live dashboard
across multiple concurrent runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunProgress:
    """Live state of a single training run."""

    run_id: str
    name: str
    max_experiments: int
    status: str  # "running" | "done" | "failed"
    started_at: str
    experiments: list[dict] = field(default_factory=list)

    @property
    def current_experiment(self) -> int:
        return len(self.experiments)

    @property
    def best_val_bpb(self) -> float | None:
        vals = [e["val_bpb"] for e in self.experiments if e.get("val_bpb") is not None]
        return min(vals) if vals else None

    @property
    def last_val_bpb(self) -> float | None:
        for e in reversed(self.experiments):
            if e.get("val_bpb") is not None:
                return e["val_bpb"]
        return None


def new_progress(run_id: str, name: str, max_experiments: int) -> RunProgress:
    return RunProgress(
        run_id=run_id,
        name=name,
        max_experiments=max_experiments,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def write_progress(path: Path, progress: RunProgress) -> None:
    path.write_text(json.dumps(asdict(progress), indent=2))


def read_progress(path: Path) -> RunProgress | None:
    try:
        data = json.loads(path.read_text())
        return RunProgress(**data)
    except Exception:
        return None


def find_progress_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("progress-*.json"))
