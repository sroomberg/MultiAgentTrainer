"""Data source for remote git repositories (any git URL)."""

import shutil
from pathlib import Path
from typing import Optional

import git as gitpython

from .base import DataSource


class RemoteRepoSource(DataSource):
    """Clone a remote git repository by URL."""

    def __init__(
        self,
        url: str,
        branch: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.url = url
        self.branch = branch
        self.name = name or url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")

    def fetch(self, staging_dir: Path) -> Path:
        dest = self._prepare_dest(staging_dir)

        clone_kwargs: dict[str, object] = {"depth": 1}
        if self.branch:
            clone_kwargs["branch"] = self.branch

        gitpython.Repo.clone_from(self.url, str(dest), **clone_kwargs)

        # Remove .git to save space — we only need the content.
        git_dir = dest / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

        return dest

    def describe(self) -> str:
        return f"remote_repo: {self.url}{self._branch_suffix(self.branch)}"
