"""Tests for data source implementations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from multiagenttrainer.sources import (
    GitHubRepoListSource,
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


class TestGitHubRepoListSource:
    def test_describe_no_branch(self) -> None:
        src = GitHubRepoListSource(
            repos=["https://github.com/a/repo1", "https://github.com/b/repo2"],
        )
        desc = src.describe()
        assert "2 repos" in desc
        assert "github-repos" in desc
        assert "(" not in desc  # no branch suffix

    def test_describe_with_branch(self) -> None:
        src = GitHubRepoListSource(
            repos=["https://github.com/a/repo1"],
            branch="main",
            name="my-repos",
        )
        desc = src.describe()
        assert "main" in desc
        assert "my-repos" in desc

    def test_empty_repos_raises(self) -> None:
        with pytest.raises(ValueError, match="repos list must not be empty"):
            GitHubRepoListSource(repos=[])

    def test_shorthand_owner_repo_normalised(self) -> None:
        src = GitHubRepoListSource(repos=["owner/myrepo"])
        assert src.repos[0] == "https://github.com/owner/myrepo.git"

    def test_full_url_preserved(self) -> None:
        url = "https://github.com/owner/myrepo"
        src = GitHubRepoListSource(repos=[url])
        assert src.repos[0] == url

    def test_invalid_shorthand_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse repo"):
            GitHubRepoListSource(repos=["not/a/valid/shorthand"])

    def test_fetch_clones_each_repo(self, tmp_staging: Path) -> None:
        src = GitHubRepoListSource(
            repos=[
                "https://github.com/owner/alpha",
                "https://github.com/owner/beta",
            ],
            name="cloned",
        )

        def fake_fetch(self_inner: object, staging: Path) -> Path:
            dest = staging / self_inner.name  # type: ignore[attr-defined]
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "file.txt").write_text("content")
            return dest

        with patch(
            "multiagenttrainer.sources.github_repo_list.RemoteRepoSource.fetch",
            fake_fetch,
        ):
            result = src.fetch(tmp_staging)

        assert result.is_dir()
        assert (result / "alpha").is_dir()
        assert (result / "beta").is_dir()

    def test_fetch_continues_on_clone_failure(self, tmp_staging: Path) -> None:
        src = GitHubRepoListSource(
            repos=[
                "https://github.com/owner/good",
                "https://github.com/owner/bad",
            ],
            name="partial",
        )
        call_count = 0

        def fake_fetch(self_inner: object, staging: Path) -> Path:
            nonlocal call_count
            call_count += 1
            if self_inner.name == "bad":  # type: ignore[attr-defined]
                raise RuntimeError("clone failed")
            dest = staging / self_inner.name  # type: ignore[attr-defined]
            dest.mkdir(parents=True, exist_ok=True)
            return dest

        with patch(
            "multiagenttrainer.sources.github_repo_list.RemoteRepoSource.fetch",
            fake_fetch,
        ):
            result = src.fetch(tmp_staging)

        assert result.is_dir()
        assert call_count == 2  # both attempted
        assert (result / "good").is_dir()
        assert not (result / "bad").exists()


class TestCreateSource:
    def test_local_repo(self) -> None:
        src = create_source({"type": "local_repo", "path": "/tmp/test"})
        assert isinstance(src, LocalRepoSource)

    def test_github_repo(self) -> None:
        src = create_source({"type": "github_repo", "url": "https://github.com/a/b"})
        assert isinstance(src, GitHubRepoSource)

    def test_github_repo_list(self) -> None:
        src = create_source(
            {
                "type": "github_repo_list",
                "repos": ["https://github.com/a/r1", "https://github.com/b/r2"],
                "branch": "dev",
                "name": "test-repos",
            }
        )
        assert isinstance(src, GitHubRepoListSource)
        assert len(src.repos) == 2
        assert src.branch == "dev"
        assert src.name == "test-repos"

    def test_github_repo_list_defaults(self) -> None:
        src = create_source(
            {
                "type": "github_repo_list",
                "repos": ["https://github.com/a/r1"],
            }
        )
        assert isinstance(src, GitHubRepoListSource)
        assert src.branch is None
        assert src.name == "github-repos"

    def test_github_repo_list_missing_repos_raises(self) -> None:
        with pytest.raises(ValueError):
            create_source({"type": "github_repo_list", "repos": "not-a-list"})

    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source({"type": "magic"})
