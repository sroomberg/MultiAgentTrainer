"""Base class for data sources."""

from __future__ import annotations

import abc
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
