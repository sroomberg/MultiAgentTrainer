"""Configuration dataclasses for fine-tuning backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    packing: bool = True
    use_bf16: bool = False
    use_flash_attention: bool = False


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
