"""Fine-tuning backends: FineTuner ABC, OpenSourceFineTuner, BedrockFineTuner."""

from __future__ import annotations

import abc
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rich.console import Console

from .config import BedrockConfig, OpenSourceConfig
from .dataset import chunk_corpus, write_jsonl


@dataclass
class FineTuneJob:
    """Snapshot of a fine-tuning job's state."""

    job_id: str
    backend: str
    status: Literal["running", "completed", "failed", "cancelled"]
    model: str
    created_at: str  # ISO-8601
    output_model: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def save(self, jobs_dir: Path) -> None:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        safe_id = self.job_id.replace("/", "_").replace(":", "_")
        (jobs_dir / f"{safe_id}.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, jobs_dir: Path, job_id: str) -> FineTuneJob | None:
        safe_id = job_id.replace("/", "_").replace(":", "_")
        path = jobs_dir / f"{safe_id}.json"
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))

    @classmethod
    def list_all(cls, jobs_dir: Path) -> list[FineTuneJob]:
        if not jobs_dir.exists():
            return []
        jobs = []
        for path in sorted(jobs_dir.glob("*.json")):
            try:
                jobs.append(cls(**json.loads(path.read_text())))
            except Exception:
                continue
        return jobs


class FineTuner(abc.ABC):
    """Abstract base for fine-tuning backends.

    Subclasses implement prepare_dataset / start_job / get_status / cancel_job.
    Adding a new backend (Anthropic, OpenAI, …) means subclassing this and
    registering it in registry.py.
    """

    def __init__(self, jobs_dir: Path, console: Console) -> None:
        self.jobs_dir = jobs_dir
        self.console = console
        jobs_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def prepare_dataset(self, corpus_path: Path) -> Any:
        """Convert a corpus file into a backend-specific dataset object."""

    @abc.abstractmethod
    def start_job(self, dataset: Any, job_name: str) -> FineTuneJob:
        """Start fine-tuning. May block (open-source) or return immediately (APIs)."""

    @abc.abstractmethod
    def get_status(self, job_id: str) -> FineTuneJob:
        """Return the current state of a job."""

    @abc.abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""

    @abc.abstractmethod
    def describe(self) -> str:
        """Short human-readable description of this backend."""


# ---------------------------------------------------------------------------
# Open-source backend
# ---------------------------------------------------------------------------


class OpenSourceFineTuner(FineTuner):
    """HuggingFace Transformers + PEFT/LoRA fine-tuner for open-source models.

    Requires: pip install 'multiagenttrainer[opensource]'
    Training runs in-process and blocks until complete.
    """

    def __init__(
        self, cfg: OpenSourceConfig, jobs_dir: Path, console: Console
    ) -> None:
        super().__init__(jobs_dir, console)
        self.cfg = cfg

    def describe(self) -> str:
        parts = ["QLoRA (4-bit)" if self.cfg.use_4bit else "LoRA"]
        parts.append("bf16" if self.cfg.use_bf16 else "fp16")
        if self.cfg.use_flash_attention:
            parts.append("Flash Attn 2")
        if self.cfg.packing:
            parts.append("packed")
        return f"OpenSource / {', '.join(parts)} — {self.cfg.model_id}"

    def prepare_dataset(self, corpus_path: Path) -> list[str]:
        """Chunk corpus into text samples sized for the configured sequence length."""
        chars_per_chunk = self.cfg.max_seq_length * 4  # ~4 chars per token
        chunks = chunk_corpus(corpus_path, chars_per_chunk)
        self.console.print(f"  [dim]Prepared {len(chunks)} text chunks[/dim]")
        return chunks

    def start_job(self, dataset: list[str], job_name: str) -> FineTuneJob:
        """Train locally with SFTTrainer + LoRA. Blocks until done."""
        self._check_deps()

        import torch
        from datasets import Dataset as HFDataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer

        job_id = (
            f"opensource-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            f"-{uuid.uuid4().hex[:6]}"
        )
        job = FineTuneJob(
            job_id=job_id,
            backend="opensource",
            status="running",
            model=self.cfg.model_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        job.save(self.jobs_dir)

        output_dir = Path(self.cfg.output_dir) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.console.print(f"[bold]Loading model:[/bold] {self.cfg.model_id}")

            compute_dtype = torch.bfloat16 if self.cfg.use_bf16 else torch.float16

            bnb_config = None
            if self.cfg.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )

            model_kwargs: dict[str, Any] = dict(
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            if self.cfg.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id, **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_id, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if self.cfg.use_4bit:
                model = prepare_model_for_kbit_training(model)

            lora_cfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.cfg.target_modules or "all-linear",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

            hf_dataset = HFDataset.from_dict({"text": dataset})

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.cfg.num_epochs,
                per_device_train_batch_size=self.cfg.batch_size,
                gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                learning_rate=self.cfg.learning_rate,
                bf16=self.cfg.use_bf16,
                fp16=not self.cfg.use_bf16 and not self.cfg.use_4bit,
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=hf_dataset,
                dataset_text_field="text",
                max_seq_length=self.cfg.max_seq_length,
                tokenizer=tokenizer,
                packing=self.cfg.packing,
            )

            self.console.print("[bold]Training…[/bold]")
            train_result = trainer.train()
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            job.status = "completed"
            job.output_model = str(output_dir)
            job.metrics = {k: float(v) for k, v in train_result.metrics.items()}

        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            self.console.print(f"[red]Training failed:[/red] {exc}")

        job.save(self.jobs_dir)
        return job

    def get_status(self, job_id: str) -> FineTuneJob:
        job = FineTuneJob.load(self.jobs_dir, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        return job

    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError(
            "Open-source training runs in-process and cannot be cancelled via this "
            "interface. Send SIGINT (Ctrl+C) to interrupt training."
        )

    @staticmethod
    def _check_deps() -> None:
        missing = []
        for pkg in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                f"Missing packages for open-source fine-tuning: {', '.join(missing)}.\n"
                "Install with: pip install 'multiagenttrainer[opensource]'"
            )


# ---------------------------------------------------------------------------
# AWS Bedrock backend
# ---------------------------------------------------------------------------


class BedrockFineTuner(FineTuner):
    """AWS Bedrock model customization fine-tuner.

    Submits a Bedrock customization job (non-blocking) and returns a job ID
    that can be polled with get_status().
    """

    def __init__(
        self, cfg: BedrockConfig, jobs_dir: Path, console: Console
    ) -> None:
        super().__init__(jobs_dir, console)
        self.cfg = cfg

    def describe(self) -> str:
        return (
            f"Bedrock ({self.cfg.customization_type}) — {self.cfg.base_model_id} "
            f"[{self.cfg.region}]"
        )

    def prepare_dataset(self, corpus_path: Path) -> Path:
        """Convert corpus to JSONL and return the local file path."""
        import tempfile

        fmt = (
            "pretraining"
            if self.cfg.customization_type == "CONTINUED_PRE_TRAINING"
            else "instruction"
        )
        jsonl_path = Path(tempfile.mkdtemp()) / "training_data.jsonl"
        count = write_jsonl(chunk_corpus(corpus_path), jsonl_path, format=fmt)
        self.console.print(f"  [dim]Wrote {count} records → {jsonl_path}[/dim]")
        return jsonl_path

    def start_job(self, dataset: Path, job_name: str) -> FineTuneJob:
        """Upload dataset to S3 and submit a Bedrock customization job."""
        import boto3

        bedrock = boto3.client("bedrock", region_name=self.cfg.region)
        s3 = boto3.client("s3", region_name=self.cfg.region)

        s3_uri = self.cfg.training_data_s3_uri.rstrip("/") + "/training_data.jsonl"
        bucket, key = self._parse_s3_uri(s3_uri)
        self.console.print(f"  [dim]Uploading → s3://{bucket}/{key}[/dim]")
        s3.upload_file(str(dataset), bucket, key)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        bedrock_job_name = f"{self.cfg.job_name_prefix}-{ts}"
        custom_model_name = f"{bedrock_job_name}-model"

        self.console.print(f"  [dim]Submitting Bedrock job: {bedrock_job_name}[/dim]")
        response = bedrock.create_model_customization_job(
            jobName=bedrock_job_name,
            customModelName=custom_model_name,
            roleArn=self.cfg.role_arn,
            baseModelIdentifier=self.cfg.base_model_id,
            customizationType=self.cfg.customization_type,
            trainingDataConfig={"s3Uri": s3_uri},
            outputDataConfig={"s3Uri": self.cfg.output_s3_uri},
            hyperParameters={
                "epochCount": str(self.cfg.epochs),
                "batchSize": str(self.cfg.batch_size),
                "learningRate": str(self.cfg.learning_rate),
            },
        )

        job = FineTuneJob(
            job_id=response["jobArn"],
            backend="bedrock",
            status="running",
            model=self.cfg.base_model_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        job.save(self.jobs_dir)
        return job

    def get_status(self, job_id: str) -> FineTuneJob:
        import boto3

        bedrock = boto3.client("bedrock", region_name=self.cfg.region)
        resp = bedrock.get_model_customization_job(jobIdentifier=job_id)

        status_map = {
            "InProgress": "running",
            "Completed": "completed",
            "Failed": "failed",
            "Stopping": "running",
            "Stopped": "cancelled",
        }
        status: Literal["running", "completed", "failed", "cancelled"] = status_map.get(
            resp.get("status", ""), "running"
        )  # type: ignore[assignment]

        job = FineTuneJob.load(self.jobs_dir, job_id) or FineTuneJob(
            job_id=job_id,
            backend="bedrock",
            status=status,
            model=self.cfg.base_model_id,
            created_at=str(
                resp.get("creationTime", datetime.now(timezone.utc).isoformat())
            ),
        )
        job.status = status
        job.output_model = resp.get("outputModelArn")
        if resp.get("status") == "Failed":
            job.error = resp.get("failureMessage")
        job.save(self.jobs_dir)
        return job

    def cancel_job(self, job_id: str) -> None:
        import boto3

        bedrock = boto3.client("bedrock", region_name=self.cfg.region)
        bedrock.stop_model_customization_job(jobIdentifier=job_id)
        job = FineTuneJob.load(self.jobs_dir, job_id)
        if job:
            job.status = "cancelled"
            job.save(self.jobs_dir)

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri!r}")
        parts = uri[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
