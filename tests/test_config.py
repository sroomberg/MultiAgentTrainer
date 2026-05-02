"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from multiagenttrainer.config import (
    AutoresearchConfig,
    TrainerConfig,
    TrainingConfig,
    load_config,
)


def test_load_defaults() -> None:
    """Loading with no file returns defaults."""
    cfg = load_config(Path("/nonexistent/path.yaml"))
    assert isinstance(cfg, TrainerConfig)
    assert cfg.autoresearch.branch == "master"
    assert cfg.training.max_experiments == 50
    assert cfg.sources == []


def test_load_from_yaml(sample_config_yaml: Path) -> None:
    """Config values are parsed from YAML."""
    cfg = load_config(sample_config_yaml)
    assert cfg.autoresearch.train_time == 60
    assert cfg.training.agent_command == "echo done"
    assert cfg.training.max_experiments == 1
    assert len(cfg.sources) == 1


def test_autoresearch_defaults() -> None:
    ac = AutoresearchConfig()
    assert "autoresearch" in ac.repo
    assert ac.train_time == 300


def test_training_defaults() -> None:
    tc = TrainingConfig()
    assert tc.max_experiments == 50
    assert "claude" in tc.agent_command
