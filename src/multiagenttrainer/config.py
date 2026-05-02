"""Configuration loading and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .sources import DataSource, create_source

CONFIG_CANDIDATES = [
    "multiagenttrainer.yaml",
    "multiagenttrainer.yml",
    ".multiagenttrainer.yaml",
    ".multiagenttrainer.yml",
]

_DEFAULT_AUTORESEARCH_REPO = "https://github.com/karpathy/autoresearch"


@dataclass
class AutoresearchConfig:
    """Settings for the autoresearch training framework."""

    repo: str = _DEFAULT_AUTORESEARCH_REPO
    branch: str = "master"
    train_time: int = 300  # seconds
    program_md: str | None = None  # optional override path


@dataclass
class ExecutionConfig:
    """Where and how to run experiments."""

    type: Literal["local", "ssh", "docker"] = "local"
    # SSH options
    ssh_host: str | None = None
    ssh_key: str | None = None  # path to private key; None = SSH default
    remote_dir: str = "/tmp/mat-runs"
    # Docker options
    container: str | None = None
    container_dir: str = "/tmp/mat-runs"


@dataclass
class TrainingConfig:
    """Settings for autonomous training runs."""

    agent_command: str = (
        "claude -p {prompt}"
        ' --allowedTools "Bash,Read,Edit"'
        " --permission-mode acceptEdits"
    )
    max_experiments: int = 50
    output_dir: str = "./training-runs"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


@dataclass
class TrainerConfig:
    """Top-level configuration for MultiAgentTrainer."""

    autoresearch: AutoresearchConfig = field(default_factory=AutoresearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sources: list[DataSource] = field(default_factory=list)
    source_configs: list[dict[str, object]] = field(default_factory=list)


def load_config(config_path: Path | None = None) -> TrainerConfig:
    """Load trainer config from YAML, falling back to defaults."""
    if config_path is None:
        for candidate in CONFIG_CANDIDATES:
            p = Path(candidate)
            if p.exists():
                config_path = p
                break

    if config_path is None or not config_path.exists():
        return TrainerConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Autoresearch settings
    ar_data = data.get("autoresearch", {})
    autoresearch = AutoresearchConfig(
        repo=ar_data.get("repo", _DEFAULT_AUTORESEARCH_REPO),
        branch=ar_data.get("branch", "master"),
        train_time=ar_data.get("train_time", 300),
        program_md=ar_data.get("program_md"),
    )

    # Training settings
    tr_data = data.get("training", {})
    ex_data = tr_data.get("execution", {})
    execution = ExecutionConfig(
        type=ex_data.get("type", "local"),
        ssh_host=ex_data.get("ssh_host"),
        ssh_key=ex_data.get("ssh_key"),
        remote_dir=ex_data.get("remote_dir", "/tmp/mat-runs"),
        container=ex_data.get("container"),
        container_dir=ex_data.get("container_dir", "/tmp/mat-runs"),
    )
    training = TrainingConfig(
        agent_command=tr_data.get("agent_command", TrainingConfig.agent_command),
        max_experiments=tr_data.get("max_experiments", 50),
        output_dir=tr_data.get("output_dir", "./training-runs"),
        execution=execution,
    )

    # Data sources
    raw_sources: list[dict[str, object]] = data.get("sources", [])
    sources: list[DataSource] = []
    for src_cfg in raw_sources:
        sources.append(create_source(src_cfg))

    return TrainerConfig(
        autoresearch=autoresearch,
        training=training,
        sources=sources,
        source_configs=raw_sources,
    )
