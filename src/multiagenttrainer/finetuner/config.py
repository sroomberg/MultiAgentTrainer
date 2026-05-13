"""Configuration dataclasses for fine-tuning backends."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class FineTuneTargetConfig:
    """A (model, machine) pairing for right-sized fine-tuning.

    Each target overrides the model and optionally the backend and
    per-backend hyperparameters from the parent FineTunerConfig.
    The ``machine`` field references a top-level MachineConfig by name
    and is metadata — it communicates which instance this target was
    designed for but does not currently drive remote dispatch.
    """

    name: str
    model_id: str
    machine: Optional[str] = None
    backend: Optional[Literal["opensource", "bedrock"]] = None
    # OpenSource overrides (None → inherit from FineTunerConfig.opensource)
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lora_r: Optional[int] = None
    # Bedrock overrides
    customization_type: Optional[str] = None


@dataclass
class OpenSourceConfig:
    """Settings for HuggingFace + PEFT/LoRA fine-tuning."""

    model_id: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "./finetuned-models"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    use_4bit: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    packing: bool = False
    use_bf16: bool = True
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    hf_token: Optional[str] = None


@dataclass
class BedrockConfig:
    """Settings for AWS Bedrock model customization."""

    base_model_id: str = "amazon.titan-text-lite-v1"
    region: str = "us-east-1"
    role_arn: str = ""
    output_s3_uri: str = ""
    training_data_s3_uri: str = ""
    customization_type: Literal["FINE_TUNING", "CONTINUED_PRE_TRAINING"] = (
        "CONTINUED_PRE_TRAINING"
    )
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    job_name_prefix: str = "mat-finetune"


@dataclass
class FineTunerConfig:
    """Top-level fine-tuner configuration."""

    backend: Literal["opensource", "bedrock"] = "opensource"
    jobs_dir: str = "./finetune-jobs"
    opensource: OpenSourceConfig = field(default_factory=OpenSourceConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    targets: list[FineTuneTargetConfig] = field(default_factory=list)
