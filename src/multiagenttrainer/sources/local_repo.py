"""Data source for local git repositories."""

import shutil
from pathlib import Path
from typing import Optional

from .base import DataSource


class LocalRepoSource(DataSource):
    """Copy a local git repository into the staging area."""

    def __init__(self, path: str, name: Optional[str] = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.name = name or self.path.name

    def fetch(self, staging_dir: Path) -> Path:
        if not self.path.is_dir():
            msg = f"Local repo not found: {self.path}"
            raise FileNotFoundError(msg)

        dest = self._prepare_dest(staging_dir)
        shutil.copytree(
            self.path,
            dest,
            ignore=shutil.ignore_patterns(".git"),
            dirs_exist_ok=True,
        )
        return dest

    def describe(self) -> str:
        return f"local_repo: {self.path}"
