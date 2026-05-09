"""Tests for the finetuner module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from multiagenttrainer.finetuner import (
    BedrockFineTuner,
    FineTuneJob,
    FineTunerConfig,
    OpenSourceFineTuner,
    create_fine_tuner,
)
from multiagenttrainer.finetuner.config import BedrockConfig, OpenSourceConfig
from multiagenttrainer.finetuner.dataset import chunk_corpus, write_jsonl

console = Console(quiet=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jobs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jobs"
    d.mkdir()
    return d


@pytest.fixture()
def corpus_file(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("A" * 10_000 + "\n" + "B" * 10_000, encoding="utf-8")
    return p


@pytest.fixture()
def os_tuner(jobs_dir: Path) -> OpenSourceFineTuner:
    cfg = OpenSourceConfig(model_id="test-model/tiny", use_4bit=False)
    return OpenSourceFineTuner(cfg, jobs_dir, console)


@pytest.fixture()
def bedrock_tuner(jobs_dir: Path) -> BedrockFineTuner:
    cfg = BedrockConfig(
        base_model_id="amazon.titan-text-lite-v1",
        region="us-east-1",
        role_arn="arn:aws:iam::123:role/TestRole",
        output_s3_uri="s3://my-bucket/output/",
        training_data_s3_uri="s3://my-bucket/training/",
    )
    return BedrockFineTuner(cfg, jobs_dir, console)


# ---------------------------------------------------------------------------
# FineTuneJob — save / load / list
# ---------------------------------------------------------------------------


def test_job_save_and_load(jobs_dir: Path) -> None:
    job = FineTuneJob(
        job_id="test-job-001",
        backend="opensource",
        status="completed",
        model="llama",
        created_at="2026-01-01T00:00:00+00:00",
        output_model="/tmp/out",
        metrics={"train_loss": 1.23},
    )
    job.save(jobs_dir)
    loaded = FineTuneJob.load(jobs_dir, "test-job-001")
    assert loaded is not None
    assert loaded.job_id == "test-job-001"
    assert loaded.status == "completed"
    assert loaded.metrics == {"train_loss": 1.23}
    assert loaded.output_model == "/tmp/out"


def test_job_load_missing_returns_none(jobs_dir: Path) -> None:
    assert FineTuneJob.load(jobs_dir, "does-not-exist") is None


def test_job_list_all_empty(tmp_path: Path) -> None:
    assert FineTuneJob.list_all(tmp_path / "no-such-dir") == []


def test_job_list_all(jobs_dir: Path) -> None:
    for i in range(3):
        FineTuneJob(
            job_id=f"job-{i}",
            backend="bedrock",
            status="completed",
            model="titan",
            created_at="2026-01-01T00:00:00+00:00",
        ).save(jobs_dir)

    jobs = FineTuneJob.list_all(jobs_dir)
    assert len(jobs) == 3
    assert {j.job_id for j in jobs} == {"job-0", "job-1", "job-2"}


def test_job_save_sanitises_arn_in_filename(jobs_dir: Path) -> None:
    """Job IDs that contain ARN characters should not break the filename."""
    arn = "arn:aws:bedrock:us-east-1:123:model-customization-job/my-job"
    job = FineTuneJob(
        job_id=arn,
        backend="bedrock",
        status="running",
        model="titan",
        created_at="2026-01-01T00:00:00+00:00",
    )
    job.save(jobs_dir)
    files = list(jobs_dir.glob("*.json"))
    assert len(files) == 1
    assert ":" not in files[0].name
    assert "/" not in files[0].name
    loaded = FineTuneJob.load(jobs_dir, arn)
    assert loaded is not None
    assert loaded.job_id == arn


# ---------------------------------------------------------------------------
# dataset.py — chunk_corpus / write_jsonl
# ---------------------------------------------------------------------------


def test_chunk_corpus_basic(corpus_file: Path) -> None:
    chunks = chunk_corpus(corpus_file, chars_per_chunk=4096)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) > 50


def test_chunk_corpus_skips_tiny_tail(tmp_path: Path) -> None:
    p = tmp_path / "c.txt"
    p.write_text("x" * 100, encoding="utf-8")
    chunks = chunk_corpus(p, chars_per_chunk=60)
    # Second chunk would be 40 chars — still > 50? No, but 40 < 50 so it should be skipped.
    # First chunk: 60 chars, second: 40 chars (stripped) → 40 < 50, dropped.
    assert all(len(c) > 50 for c in chunks)


def test_write_jsonl_pretraining(tmp_path: Path) -> None:
    out = tmp_path / "data.jsonl"
    chunks = ["hello world", "foo bar baz"]
    count = write_jsonl(chunks, out, format="pretraining")
    assert count == 2
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    for line, chunk in zip(lines, chunks):
        record = json.loads(line)
        assert record == {"input": chunk}


def test_write_jsonl_instruction(tmp_path: Path) -> None:
    out = tmp_path / "data.jsonl"
    chunk = "ABCDEFGH"
    write_jsonl([chunk], out, format="instruction")
    record = json.loads(out.read_text().splitlines()[0])
    assert "prompt" in record and "completion" in record
    assert record["prompt"] + record["completion"] == chunk


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "data.jsonl"
    write_jsonl(["hello"], out)
    assert out.exists()


# ---------------------------------------------------------------------------
# registry — create_fine_tuner
# ---------------------------------------------------------------------------


def test_create_fine_tuner_opensource(jobs_dir: Path) -> None:
    cfg = FineTunerConfig(backend="opensource", jobs_dir=str(jobs_dir))
    tuner = create_fine_tuner(cfg, console)
    assert isinstance(tuner, OpenSourceFineTuner)


def test_create_fine_tuner_bedrock(jobs_dir: Path) -> None:
    cfg = FineTunerConfig(backend="bedrock", jobs_dir=str(jobs_dir))
    tuner = create_fine_tuner(cfg, console)
    assert isinstance(tuner, BedrockFineTuner)


def test_create_fine_tuner_unknown(jobs_dir: Path) -> None:
    cfg = FineTunerConfig(backend="opensource", jobs_dir=str(jobs_dir))
    cfg.backend = "openai"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown fine-tuning backend"):
        create_fine_tuner(cfg, console)


# ---------------------------------------------------------------------------
# OpenSourceFineTuner
# ---------------------------------------------------------------------------


def test_os_describe(os_tuner: OpenSourceFineTuner) -> None:
    desc = os_tuner.describe()
    assert "test-model/tiny" in desc
    assert "LoRA" in desc


def test_os_prepare_dataset(os_tuner: OpenSourceFineTuner, corpus_file: Path) -> None:
    chunks = os_tuner.prepare_dataset(corpus_file)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_os_check_deps_raises_on_missing(os_tuner: OpenSourceFineTuner) -> None:
    # Temporarily hide 'torch' from sys.modules so _check_deps sees it as missing.
    real_torch = sys.modules.pop("torch", None)
    try:
        with pytest.raises(ImportError, match="multiagenttrainer\\[opensource\\]"):
            os_tuner._check_deps()
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch


def test_os_get_status_missing_job(os_tuner: OpenSourceFineTuner) -> None:
    with pytest.raises(ValueError, match="Job not found"):
        os_tuner.get_status("nonexistent-id")


def test_os_cancel_raises(os_tuner: OpenSourceFineTuner) -> None:
    with pytest.raises(NotImplementedError):
        os_tuner.cancel_job("any-id")


def test_os_start_job_saves_failed_job_on_error(
    os_tuner: OpenSourceFineTuner, jobs_dir: Path
) -> None:
    """If training crashes, start_job should still write a failed job file."""
    chunks = ["some text chunk"] * 5

    # Patch _check_deps to pass, then make AutoModelForCausalLM blow up.
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError(
        "CUDA out of memory"
    )
    mock_peft = MagicMock()
    mock_trl = MagicMock()
    mock_datasets = MagicMock()
    mock_bnb = MagicMock()

    fake_modules = {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "trl": mock_trl,
        "datasets": mock_datasets,
        "bitsandbytes": mock_bnb,
    }
    with patch.dict("sys.modules", fake_modules):
        job = os_tuner.start_job(chunks, "crash-test")

    assert job.status == "failed"
    assert job.error is not None
    assert "CUDA" in job.error
    saved = FineTuneJob.load(jobs_dir, job.job_id)
    assert saved is not None
    assert saved.status == "failed"


def test_os_start_job_success(
    os_tuner: OpenSourceFineTuner, jobs_dir: Path
) -> None:
    """Happy path: mock HF stack and verify a completed job is returned."""
    chunks = ["some text chunk"] * 5

    mock_train_result = MagicMock()
    mock_train_result.metrics = {"train_loss": 0.42, "epoch": 3.0}

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_train_result

    mock_torch = MagicMock()
    mock_torch.float16 = "float16"

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.BitsAndBytesConfig.return_value = MagicMock()
    mock_transformers.TrainingArguments.return_value = MagicMock()

    mock_peft = MagicMock()
    mock_peft.get_peft_model.return_value = mock_model

    mock_trl = MagicMock()
    mock_trl.SFTTrainer.return_value = mock_trainer

    mock_datasets = MagicMock()
    mock_hf_dataset = MagicMock()
    mock_datasets.Dataset.from_dict.return_value = mock_hf_dataset

    fake_modules = {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "trl": mock_trl,
        "datasets": mock_datasets,
        "bitsandbytes": MagicMock(),
    }
    with patch.dict("sys.modules", fake_modules):
        job = os_tuner.start_job(chunks, "success-test")

    assert job.status == "completed"
    assert job.output_model is not None
    assert job.metrics["train_loss"] == pytest.approx(0.42)
    saved = FineTuneJob.load(jobs_dir, job.job_id)
    assert saved is not None
    assert saved.status == "completed"


# ---------------------------------------------------------------------------
# OpenSourceFineTuner — speed optimisations (packing / bf16 / flash attn)
# ---------------------------------------------------------------------------


class _HFMocks:
    """Holds the key mock objects from a _run_start_job call."""

    def __init__(
        self,
        from_pretrained: MagicMock,
        training_args_cls: MagicMock,
        sft_trainer_cls: MagicMock,
        torch: MagicMock,
        bnb_config_cls: MagicMock,
    ) -> None:
        self.from_pretrained = from_pretrained
        self.training_args_cls = training_args_cls
        self.sft_trainer_cls = sft_trainer_cls
        self.torch = torch
        self.bnb_config_cls = bnb_config_cls


def _run_start_job(tuner: OpenSourceFineTuner) -> tuple[FineTuneJob, _HFMocks]:
    """Run start_job with a fully mocked HF stack and return (job, mocks)."""
    mock_train_result = MagicMock()
    mock_train_result.metrics = {}

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_train_result

    mock_sft_trainer_cls = MagicMock(return_value=mock_trainer)
    mock_training_args_cls = MagicMock(return_value=MagicMock())
    mock_from_pretrained = MagicMock(return_value=MagicMock())
    mock_bnb_config_cls = MagicMock(return_value=MagicMock())

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    mock_torch = MagicMock()
    mock_torch.float16 = "float16"
    mock_torch.bfloat16 = "bfloat16"

    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained = mock_from_pretrained
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.BitsAndBytesConfig = mock_bnb_config_cls
    mock_transformers.TrainingArguments = mock_training_args_cls

    mock_peft = MagicMock()
    mock_peft.get_peft_model.return_value = MagicMock()

    mock_trl = MagicMock()
    mock_trl.SFTTrainer = mock_sft_trainer_cls

    fake_modules = {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "trl": mock_trl,
        "datasets": MagicMock(),
        "bitsandbytes": MagicMock(),
    }
    with patch.dict("sys.modules", fake_modules):
        job = tuner.start_job(["chunk one", "chunk two"], "test")

    return job, _HFMocks(
        from_pretrained=mock_from_pretrained,
        training_args_cls=mock_training_args_cls,
        sft_trainer_cls=mock_sft_trainer_cls,
        torch=mock_torch,
        bnb_config_cls=mock_bnb_config_cls,
    )


def test_os_describe_shows_precision_and_packing(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, use_bf16=False, packing=True)
    tuner = OpenSourceFineTuner(cfg, jobs_dir, console)
    desc = tuner.describe()
    assert "fp16" in desc
    assert "packed" in desc
    assert "bf16" not in desc
    assert "Flash" not in desc


def test_os_describe_all_optimisations(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(
        model_id="m",
        use_4bit=True,
        use_bf16=True,
        use_flash_attention=True,
        packing=True,
    )
    tuner = OpenSourceFineTuner(cfg, jobs_dir, console)
    desc = tuner.describe()
    assert "QLoRA" in desc
    assert "bf16" in desc
    assert "Flash Attn 2" in desc
    assert "packed" in desc


def test_os_start_job_packing_true(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, packing=True)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.sft_trainer_cls.call_args
    assert kwargs["packing"] is True


def test_os_start_job_packing_false(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, packing=False)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.sft_trainer_cls.call_args
    assert kwargs["packing"] is False


def test_os_start_job_bf16_sets_training_args(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, use_bf16=True)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.training_args_cls.call_args
    assert kwargs["bf16"] is True
    assert kwargs["fp16"] is False


def test_os_start_job_no_bf16_sets_fp16(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, use_bf16=False)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.training_args_cls.call_args
    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is True


def test_os_start_job_bf16_sets_bnb_compute_dtype(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=True, use_bf16=True)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.bnb_config_cls.call_args
    assert kwargs["bnb_4bit_compute_dtype"] == mocks.torch.bfloat16


def test_os_start_job_flash_attention_sets_attn_impl(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, use_flash_attention=True)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.from_pretrained.call_args
    assert kwargs.get("attn_implementation") == "flash_attention_2"


def test_os_start_job_no_flash_attention_omits_attn_impl(jobs_dir: Path) -> None:
    cfg = OpenSourceConfig(model_id="m", use_4bit=False, use_flash_attention=False)
    _, mocks = _run_start_job(OpenSourceFineTuner(cfg, jobs_dir, console))
    _, kwargs = mocks.from_pretrained.call_args
    assert "attn_implementation" not in kwargs


# ---------------------------------------------------------------------------
# BedrockFineTuner
# ---------------------------------------------------------------------------


def test_bedrock_describe(bedrock_tuner: BedrockFineTuner) -> None:
    desc = bedrock_tuner.describe()
    assert "CONTINUED_PRE_TRAINING" in desc
    assert "titan" in desc


def test_bedrock_parse_s3_uri(bedrock_tuner: BedrockFineTuner) -> None:
    bucket, key = bedrock_tuner._parse_s3_uri("s3://my-bucket/path/to/file.jsonl")
    assert bucket == "my-bucket"
    assert key == "path/to/file.jsonl"


def test_bedrock_parse_s3_uri_bucket_only(bedrock_tuner: BedrockFineTuner) -> None:
    bucket, key = bedrock_tuner._parse_s3_uri("s3://my-bucket/")
    assert bucket == "my-bucket"


def test_bedrock_parse_s3_uri_invalid(bedrock_tuner: BedrockFineTuner) -> None:
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        bedrock_tuner._parse_s3_uri("https://not-s3.com/bucket/key")


def test_bedrock_prepare_dataset(
    bedrock_tuner: BedrockFineTuner, corpus_file: Path
) -> None:
    jsonl_path = bedrock_tuner.prepare_dataset(corpus_file)
    assert isinstance(jsonl_path, Path)
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert "input" in record  # CONTINUED_PRE_TRAINING format


def test_bedrock_start_job(
    bedrock_tuner: BedrockFineTuner, tmp_path: Path, jobs_dir: Path
) -> None:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text('{"input": "hello"}\n')

    mock_bedrock = MagicMock()
    mock_bedrock.create_model_customization_job.return_value = {
        "jobArn": "arn:aws:bedrock:us-east-1:123:model-customization-job/test-job"
    }
    mock_s3 = MagicMock()

    def fake_client(service: str, **_: Any) -> MagicMock:
        return mock_bedrock if service == "bedrock" else mock_s3

    with patch("boto3.client", side_effect=fake_client):
        job = bedrock_tuner.start_job(dataset_path, "my-job")

    assert job.status == "running"
    assert job.backend == "bedrock"
    assert "model-customization-job" in job.job_id
    mock_s3.upload_file.assert_called_once()
    mock_bedrock.create_model_customization_job.assert_called_once()


def test_bedrock_get_status_completed(
    bedrock_tuner: BedrockFineTuner, jobs_dir: Path
) -> None:
    job_id = "arn:aws:bedrock:us-east-1:123:model-customization-job/done-job"
    FineTuneJob(
        job_id=job_id,
        backend="bedrock",
        status="running",
        model="titan",
        created_at="2026-01-01T00:00:00+00:00",
    ).save(jobs_dir)

    mock_bedrock = MagicMock()
    mock_bedrock.get_model_customization_job.return_value = {
        "status": "Completed",
        "outputModelArn": "arn:aws:bedrock:us-east-1:123:custom-model/done-model",
        "creationTime": "2026-01-01T00:00:00+00:00",
    }

    with patch("boto3.client", return_value=mock_bedrock):
        job = bedrock_tuner.get_status(job_id)

    assert job.status == "completed"
    assert job.output_model == "arn:aws:bedrock:us-east-1:123:custom-model/done-model"


def test_bedrock_get_status_failed(
    bedrock_tuner: BedrockFineTuner, jobs_dir: Path
) -> None:
    job_id = "arn:aws:bedrock:us-east-1:123:model-customization-job/fail-job"
    mock_bedrock = MagicMock()
    mock_bedrock.get_model_customization_job.return_value = {
        "status": "Failed",
        "failureMessage": "Insufficient training data",
        "creationTime": "2026-01-01T00:00:00+00:00",
    }

    with patch("boto3.client", return_value=mock_bedrock):
        job = bedrock_tuner.get_status(job_id)

    assert job.status == "failed"
    assert job.error == "Insufficient training data"


def test_bedrock_cancel_job(
    bedrock_tuner: BedrockFineTuner, jobs_dir: Path
) -> None:
    job_id = "arn:aws:bedrock:us-east-1:123:model-customization-job/running-job"
    FineTuneJob(
        job_id=job_id,
        backend="bedrock",
        status="running",
        model="titan",
        created_at="2026-01-01T00:00:00+00:00",
    ).save(jobs_dir)

    mock_bedrock = MagicMock()
    with patch("boto3.client", return_value=mock_bedrock):
        bedrock_tuner.cancel_job(job_id)

    mock_bedrock.stop_model_customization_job.assert_called_once_with(
        jobIdentifier=job_id
    )
    saved = FineTuneJob.load(jobs_dir, job_id)
    assert saved is not None
    assert saved.status == "cancelled"
