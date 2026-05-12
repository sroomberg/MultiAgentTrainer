"""Data source implementations."""

from .base import DataSource
from .bedrock_kb import BedrockKnowledgeBaseSource
from .github_org import GitHubOrgSource
from .github_repo import GitHubRepoSource
from .github_repo_list import GitHubRepoListSource
from .local_repo import LocalRepoSource
from .remote_repo import RemoteRepoSource

__all__ = [
    "BedrockKnowledgeBaseSource",
    "DataSource",
    "GitHubOrgSource",
    "GitHubRepoSource",
    "GitHubRepoListSource",
    "LocalRepoSource",
    "RemoteRepoSource",
    "create_source",
]


def create_source(source_cfg: dict[str, object]) -> DataSource:
    """Instantiate a DataSource from a config dict (as parsed from YAML)."""
    src_type = source_cfg.get("type")

    if src_type == "local_repo":
        return LocalRepoSource(
            path=str(source_cfg["path"]),
            name=source_cfg.get("name"),  # type: ignore[arg-type]
        )

    if src_type == "remote_repo":
        return RemoteRepoSource(
            url=str(source_cfg["url"]),
            branch=source_cfg.get("branch"),  # type: ignore[arg-type]
            name=source_cfg.get("name"),  # type: ignore[arg-type]
        )

    if src_type == "github_repo":
        return GitHubRepoSource(
            url=str(source_cfg["url"]),
            branch=source_cfg.get("branch"),  # type: ignore[arg-type]
            name=source_cfg.get("name"),  # type: ignore[arg-type]
        )

    if src_type == "github_org":
        return GitHubOrgSource(
            url=str(source_cfg["url"]),
            max_repos=int(source_cfg.get("max_repos", 100)),  # type: ignore[arg-type]
            visibility=str(source_cfg.get("visibility", "all")),
            name=source_cfg.get("name"),  # type: ignore[arg-type]
        )

    if src_type == "github_repo_list":
        raw_repos = source_cfg.get("repos", [])
        if not isinstance(raw_repos, list):
            raise ValueError("github_repo_list source requires a 'repos' list")
        return GitHubRepoListSource(
            repos=[str(r) for r in raw_repos],
            branch=source_cfg.get("branch"),  # type: ignore[arg-type]
            name=str(source_cfg.get("name", "github-repos")),
        )

    if src_type == "bedrock_knowledge_base":
        return BedrockKnowledgeBaseSource(
            knowledge_base_id=str(source_cfg["knowledge_base_id"]),
            region=str(source_cfg.get("region", "us-east-1")),
            query=str(source_cfg.get("query", "training data")),
            max_results=int(source_cfg.get("max_results", 100)),  # type: ignore[arg-type]
            name=source_cfg.get("name"),  # type: ignore[arg-type]
        )

    msg = f"Unknown source type: {src_type}"
    raise ValueError(msg)
