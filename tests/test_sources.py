"""Tests for data source implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiagenttrainer.sources import (
    LocalRepoSource,
    create_source,
)
from multiagenttrainer.sources.github_repo import GitHubRepoSource


class TestLocalRepoSource:
    def test_fetch_copies_files(self, local_repo: Path, tmp_staging: Path) -> None:
        src = LocalRepoSource(str(local_repo))
        dest = src.fetch(tmp_staging)

        assert dest.is_dir()
        assert (dest / "main.py").exists()
        assert (dest / "README.md").exists()
        # .git should be excluded
        assert not (dest / ".git").exists()

    def test_fetch_missing_raises(self, tmp_staging: Path) -> None:
        src = LocalRepoSource("/tmp/does-not-exist-xyz")
        with pytest.raises(FileNotFoundError):
            src.fetch(tmp_staging)

    def test_describe(self, local_repo: Path) -> None:
        src = LocalRepoSource(str(local_repo))
        assert "local_repo" in src.describe()


class TestGitHubRepoSource:
    def test_valid_url(self) -> None:
        src = GitHubRepoSource("https://github.com/owner/repo")
        assert src.owner == "owner"
        assert src.repo == "repo"

    def test_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="Not a valid GitHub repo URL"):
            GitHubRepoSource("https://example.com/not-github")

    def test_describe(self) -> None:
        src = GitHubRepoSource("https://github.com/foo/bar", branch="dev")
        assert "foo/bar" in src.describe()
        assert "dev" in src.describe()


class TestCreateSource:
    def test_local_repo(self) -> None:
        src = create_source({"type": "local_repo", "path": "/tmp/test"})
        assert isinstance(src, LocalRepoSource)

    def test_github_repo(self) -> None:
        src = create_source({"type": "github_repo", "url": "https://github.com/a/b"})
        assert isinstance(src, GitHubRepoSource)

    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source({"type": "magic"})
