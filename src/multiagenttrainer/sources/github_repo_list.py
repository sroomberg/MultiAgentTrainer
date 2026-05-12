"""Data source for an explicit list of GitHub repository URLs."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import DataSource
from .remote_repo import RemoteRepoSource

log = logging.getLogger(__name__)


def _normalise_repo_url(repo: str) -> str:
    """Accept either a full URL or an ``owner/repo`` shorthand."""
    if repo.startswith(("https://", "http://", "git@")):
        return repo
    parts = repo.strip("/").split("/")
    if len(parts) == 2:  # noqa: PLR2004
        return f"https://github.com/{parts[0]}/{parts[1]}.git"
    raise ValueError(
        f"Cannot parse repo {repo!r}: expected a URL or 'owner/repo' shorthand"
    )


class GitHubRepoListSource(DataSource):
    """Clone an explicit list of GitHub (or any git) repositories.

    Each repo is cloned into its own subdirectory inside a shared parent
    directory, mirroring the layout produced by GitHubOrgSource.

    Entries in *repos* may be full URLs or ``owner/repo`` shorthands.
    """

    def __init__(
        self,
        repos: list[str],
        branch: str | None = None,
        name: str = "github-repos",
    ) -> None:
        if not repos:
            raise ValueError("repos list must not be empty")
        self.repos = [_normalise_repo_url(r) for r in repos]
        self.branch = branch
        self.name = name

    def fetch(self, staging_dir: Path) -> Path:
        parent = self._prepare_dest(staging_dir)

        for url in self.repos:
            repo_name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            source = RemoteRepoSource(url=url, branch=self.branch, name=repo_name)
            try:
                source.fetch(parent)
                log.info("  cloned %s", repo_name)
            except Exception:
                log.warning("  failed to clone %s", url, exc_info=True)

        return parent

    def describe(self) -> str:
        suffix = self._branch_suffix(self.branch)
        return f"github_repo_list: {len(self.repos)} repos{suffix} → {self.name}"
