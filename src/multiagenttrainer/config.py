"""Configuration loading and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from .finetuner.config import BedrockConfig, FineTunerConfig, OpenSourceConfig
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
    finetuner: FineTunerConfig | None = None


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

    # Fine-tuner settings
    finetuner: FineTunerConfig | None = None
    ft_data = data.get("finetuner")
    if ft_data:
        finetuner = _parse_finetuner_config(ft_data)

    return TrainerConfig(
        autoresearch=autoresearch,
        training=training,
        sources=sources,
        source_configs=raw_sources,
        finetuner=finetuner,
    )


def _parse_finetuner_config(data: dict[str, object]) -> FineTunerConfig:
    os_data = data.get("opensource", {})
    assert isinstance(os_data, dict)
    opensource = OpenSourceConfig(
        model_id=os_data.get("model_id", "meta-llama/Llama-3.2-1B"),  # type: ignore[arg-type]
        output_dir=os_data.get("output_dir", "./finetuned-models"),  # type: ignore[arg-type]
        lora_r=os_data.get("lora_r", 16),  # type: ignore[arg-type]
        lora_alpha=os_data.get("lora_alpha", 32),  # type: ignore[arg-type]
        lora_dropout=os_data.get("lora_dropout", 0.05),  # type: ignore[arg-type]
        max_seq_length=os_data.get("max_seq_length", 2048),  # type: ignore[arg-type]
        num_epochs=os_data.get("num_epochs", 3),  # type: ignore[arg-type]
        batch_size=os_data.get("batch_size", 4),  # type: ignore[arg-type]
        learning_rate=os_data.get("learning_rate", 2e-4),  # type: ignore[arg-type]
        gradient_accumulation_steps=os_data.get("gradient_accumulation_steps", 4),  # type: ignore[arg-type]
        use_4bit=os_data.get("use_4bit", True),  # type: ignore[arg-type]
        target_modules=os_data.get("target_modules", ["q_proj", "v_proj"]),  # type: ignore[arg-type]
    )

    br_data = data.get("bedrock", {})
    assert isinstance(br_data, dict)
    bedrock = BedrockConfig(
        base_model_id=br_data.get("base_model_id", "amazon.titan-text-lite-v1"),  # type: ignore[arg-type]
        region=br_data.get("region", "us-east-1"),  # type: ignore[arg-type]
        role_arn=br_data.get("role_arn", ""),  # type: ignore[arg-type]
        output_s3_uri=br_data.get("output_s3_uri", ""),  # type: ignore[arg-type]
        training_data_s3_uri=br_data.get("training_data_s3_uri", ""),  # type: ignore[arg-type]
        customization_type=br_data.get("customization_type", "CONTINUED_PRE_TRAINING"),  # type: ignore[arg-type]
        epochs=br_data.get("epochs", 1),  # type: ignore[arg-type]
        batch_size=br_data.get("batch_size", 8),  # type: ignore[arg-type]
        learning_rate=br_data.get("learning_rate", 1e-5),  # type: ignore[arg-type]
        job_name_prefix=br_data.get("job_name_prefix", "mat-finetune"),  # type: ignore[arg-type]
    )

    return FineTunerConfig(
        backend=data.get("backend", "opensource"),  # type: ignore[arg-type]
        jobs_dir=data.get("jobs_dir", "./finetune-jobs"),  # type: ignore[arg-type]
        opensource=opensource,
        bedrock=bedrock,
    )
