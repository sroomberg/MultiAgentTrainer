"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_staging(tmp_path: Path) -> Path:
    """Return a temporary staging directory."""
    d = tmp_path / "staging"
    d.mkdir()
    return d


@pytest.fixture()
def sample_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    cfg = tmp_path / "multiagenttrainer.yaml"
    cfg.write_text(
        """\
autoresearch:
  repo: "https://github.com/karpathy/autoresearch"
  branch: master
  train_time: 60

sources:
  - type: local_repo
    path: /tmp/nonexistent-repo

training:
  agent_command: "echo done"
  max_experiments: 1
  output_dir: ./out
"""
    )
    return cfg


@pytest.fixture()
def local_repo(tmp_path: Path) -> Path:
    """Create a tiny fake local repo for testing."""
    repo = tmp_path / "fakerepo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n")
    (repo / "README.md").write_text("# Fake\n")
    (repo / ".git").mkdir()  # fake .git dir
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return repo
