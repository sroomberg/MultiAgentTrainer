"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from multiagenttrainer.config import (
    AutoresearchConfig,
    TrainerConfig,
    TrainingConfig,
    load_config,
)
from multiagenttrainer.finetuner.config import FineTunerConfig


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


def test_finetuner_absent_by_default(sample_config_yaml: Path) -> None:
    """Config without a finetuner section should have finetuner=None."""
    cfg = load_config(sample_config_yaml)
    assert cfg.finetuner is None


def test_finetuner_opensource_parsed(tmp_path: Path) -> None:
    """Opensource finetuner config is parsed into correct dataclasses."""
    cfg_file = tmp_path / "multiagenttrainer.yaml"
    cfg_file.write_text(
        """\
finetuner:
  backend: opensource
  jobs_dir: ./my-jobs
  opensource:
    model_id: meta-llama/Llama-3.2-1B
    num_epochs: 5
    use_4bit: false
    lora_r: 8
"""
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg.finetuner, FineTunerConfig)
    assert cfg.finetuner.backend == "opensource"
    assert cfg.finetuner.jobs_dir == "./my-jobs"
    assert cfg.finetuner.opensource.model_id == "meta-llama/Llama-3.2-1B"
    assert cfg.finetuner.opensource.num_epochs == 5
    assert cfg.finetuner.opensource.use_4bit is False
    assert cfg.finetuner.opensource.lora_r == 8
    # defaults for speed fields
    assert cfg.finetuner.opensource.packing is True
    assert cfg.finetuner.opensource.use_bf16 is False
    assert cfg.finetuner.opensource.use_flash_attention is False


def test_finetuner_opensource_speed_fields_parsed(tmp_path: Path) -> None:
    """packing / use_bf16 / use_flash_attention are parsed from YAML."""
    cfg_file = tmp_path / "multiagenttrainer.yaml"
    cfg_file.write_text(
        """\
finetuner:
  backend: opensource
  opensource:
    model_id: test/model
    packing: false
    use_bf16: true
    use_flash_attention: true
"""
    )
    cfg = load_config(cfg_file)
    os_cfg = cfg.finetuner.opensource  # type: ignore[union-attr]
    assert os_cfg.packing is False
    assert os_cfg.use_bf16 is True
    assert os_cfg.use_flash_attention is True


def test_finetuner_bedrock_parsed(tmp_path: Path) -> None:
    """Bedrock finetuner config is parsed into correct dataclasses."""
    cfg_file = tmp_path / "multiagenttrainer.yaml"
    cfg_file.write_text(
        """\
finetuner:
  backend: bedrock
  bedrock:
    base_model_id: amazon.titan-text-express-v1
    region: eu-west-1
    role_arn: arn:aws:iam::999:role/MyRole
    output_s3_uri: s3://out-bucket/models/
    training_data_s3_uri: s3://data-bucket/train/
    customization_type: FINE_TUNING
    epochs: 2
"""
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg.finetuner, FineTunerConfig)
    assert cfg.finetuner.backend == "bedrock"
    br = cfg.finetuner.bedrock
    assert br.base_model_id == "amazon.titan-text-express-v1"
    assert br.region == "eu-west-1"
    assert br.customization_type == "FINE_TUNING"
    assert br.epochs == 2
    assert br.output_s3_uri == "s3://out-bucket/models/"


def test_machines_parsed(tmp_path: Path) -> None:
    """Named machines are parsed from the top-level machines key."""
    cfg_file = tmp_path / "multiagenttrainer.yaml"
    cfg_file.write_text(
        """\
machines:
  - name: gpu-large
    execution:
      type: ssh
      ssh_host: trainer1.example.com
      ssh_key: ~/.ssh/id_ed25519
      remote_dir: /tmp/mat-runs
    agent_command: claude -p {prompt} --model claude-opus-4-7
  - name: gpu-small
    execution:
      type: local
"""
    )
    from multiagenttrainer.config import MachineConfig

    cfg = load_config(cfg_file)
    assert len(cfg.machines) == 2
    big = cfg.machines[0]
    assert isinstance(big, MachineConfig)
    assert big.name == "gpu-large"
    assert big.execution.type == "ssh"
    assert big.execution.ssh_host == "trainer1.example.com"
    assert big.execution.ssh_key == "~/.ssh/id_ed25519"
    assert big.agent_command is not None
    assert "claude-opus-4-7" in big.agent_command
    small = cfg.machines[1]
    assert small.name == "gpu-small"
    assert small.execution.type == "local"
    assert small.agent_command is None


def test_machines_empty_by_default(sample_config_yaml: Path) -> None:
    cfg = load_config(sample_config_yaml)
    assert cfg.machines == []


def test_finetuner_targets_parsed(tmp_path: Path) -> None:
    """finetuner.targets are parsed into FineTuneTargetConfig objects."""
    cfg_file = tmp_path / "multiagenttrainer.yaml"
    cfg_file.write_text(
        """\
finetuner:
  backend: opensource
  targets:
    - name: big-model
      model_id: meta-llama/Llama-3.2-3B
      machine: gpu-large
      backend: opensource
      num_epochs: 3
      batch_size: 2
      lora_r: 32
    - name: bedrock-job
      model_id: amazon.titan-text-lite-v1
      backend: bedrock
      customization_type: FINE_TUNING
"""
    )
    from multiagenttrainer.finetuner.config import FineTuneTargetConfig

    cfg = load_config(cfg_file)
    assert cfg.finetuner is not None
    targets = cfg.finetuner.targets
    assert len(targets) == 2
    t0 = targets[0]
    assert isinstance(t0, FineTuneTargetConfig)
    assert t0.name == "big-model"
    assert t0.model_id == "meta-llama/Llama-3.2-3B"
    assert t0.machine == "gpu-large"
    assert t0.backend == "opensource"
    assert t0.num_epochs == 3
    assert t0.batch_size == 2
    assert t0.lora_r == 32
    t1 = targets[1]
    assert t1.name == "bedrock-job"
    assert t1.backend == "bedrock"
    assert t1.customization_type == "FINE_TUNING"
    assert t1.machine is None


def test_finetuner_targets_empty_by_default(tmp_path: Path) -> None:
    cfg_file = tmp_path / "multiagenttrainer.yaml"
    cfg_file.write_text("finetuner:\n  backend: opensource\n")
    cfg = load_config(cfg_file)
    assert cfg.finetuner is not None
    assert cfg.finetuner.targets == []
