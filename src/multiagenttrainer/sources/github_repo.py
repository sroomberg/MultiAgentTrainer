"""Data source for a single GitHub repository URL."""

from __future__ import annotations

import re
from pathlib import Path

from .base import DataSource
from .remote_repo import RemoteRepoSource

_GH_REPO_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


class GitHubRepoSource(DataSource):
    """Clone a single GitHub repository given its web URL."""

    def __init__(
        self,
        url: str,
        branch: str | None = None,
        name: str | None = None,
    ) -> None:
        m = _GH_REPO_RE.match(url)
        if not m:
            msg = f"Not a valid GitHub repo URL: {url}"
            raise ValueError(msg)

        self.owner = m.group("owner")
        self.repo = m.group("repo")
        self.url = url
        self.branch = branch
        self.name = name or self.repo

        # Delegate to RemoteRepoSource for the actual clone.
        clone_url = f"https://github.com/{self.owner}/{self.repo}.git"
        self._inner = RemoteRepoSource(url=clone_url, branch=branch, name=self.name)

    def fetch(self, staging_dir: Path) -> Path:
        return self._inner.fetch(staging_dir)

    def describe(self) -> str:
        return (
            f"github_repo: {self.owner}/{self.repo}"
            f"{self._branch_suffix(self.branch)}"
        )
