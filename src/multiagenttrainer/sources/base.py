"""Base class for data sources."""

from __future__ import annotations

import abc
import shutil
from pathlib import Path


class DataSource(abc.ABC):
    """Abstract base for all data sources.

    Subclasses must implement :meth:`fetch` and :meth:`describe`.
    """

    @abc.abstractmethod
    def fetch(self, staging_dir: Path) -> Path:
        """Download / copy source data into *staging_dir*.

        Returns the path to the fetched content (a directory).
        """

    @abc.abstractmethod
    def describe(self) -> str:
        """Return a short human-readable description of this source."""

    def _prepare_dest(self, staging_dir: Path) -> Path:
        """Return a clean, empty destination directory inside *staging_dir*."""
        dest = staging_dir / self.name  # type: ignore[attr-defined]
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        return dest

    @staticmethod
    def _branch_suffix(branch: str | None) -> str:
        return f" ({branch})" if branch else ""
