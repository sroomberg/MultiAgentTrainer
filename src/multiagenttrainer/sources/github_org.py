"""Data source for all accessible repos in a GitHub organization."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from .base import DataSource
from .remote_repo import RemoteRepoSource

log = logging.getLogger(__name__)

_GH_ORG_RE = re.compile(r"https?://github\.com/(?P<org>[^/]+)/?$")


def _list_org_repos(
    org: str,
    visibility: str = "all",
    max_repos: int = 100,
) -> list[dict[str, str]]:
    """Use the ``gh`` CLI to list repos for an org.

    Falls back to the GitHub REST API via ``requests`` if ``gh`` is not
    available.
    """
    try:
        result = subprocess.run(
            [
                "gh",
                "repo",
                "list",
                org,
                "--json",
                "name,url,visibility",
                "--limit",
                str(max_repos),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        repos: list[dict[str, str]] = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback to requests + GitHub API
        import os

        import requests

        headers: dict[str, str] = {}

        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        repos = []
        page = 1
        per_page = min(max_repos, 100)
        while len(repos) < max_repos:
            resp = requests.get(
                f"https://api.github.com/orgs/{org}/repos",
                params={"per_page": per_page, "page": page, "type": visibility},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for r in batch:
                repos.append(
                    {
                        "name": r["name"],
                        "url": r["clone_url"],
                        "visibility": r.get("visibility", "public"),
                    }
                )
            page += 1

    # Filter by visibility
    if visibility != "all":
        repos = [r for r in repos if r.get("visibility") == visibility]

    return repos[:max_repos]


class GitHubOrgSource(DataSource):
    """Clone all accessible repositories from a GitHub organization."""

    def __init__(
        self,
        url: str,
        max_repos: int = 100,
        visibility: str = "all",
        name: str | None = None,
    ) -> None:
        m = _GH_ORG_RE.match(url)
        if not m:
            msg = f"Not a valid GitHub org URL: {url}"
            raise ValueError(msg)

        self.org = m.group("org")
        self.url = url
        self.max_repos = max_repos
        self.visibility = visibility
        self.name = name or self.org

    def fetch(self, staging_dir: Path) -> Path:
        org_dir = self._prepare_dest(staging_dir)

        repos = _list_org_repos(self.org, self.visibility, self.max_repos)
        log.info("Found %d repos in %s", len(repos), self.org)

        for repo in repos:
            clone_url = repo.get("url") or (
                f"https://github.com/{self.org}/{repo['name']}.git"
            )
            source = RemoteRepoSource(url=clone_url, name=repo["name"])
            try:
                source.fetch(org_dir)
                log.info("  cloned %s", repo["name"])
            except Exception:
                log.warning("  failed to clone %s", repo["name"], exc_info=True)

        return org_dir

    def describe(self) -> str:
        return (
            f"github_org: {self.org} "
            f"(max_repos={self.max_repos}, visibility={self.visibility})"
        )
