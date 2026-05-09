"""Corpus-to-dataset conversion utilities for fine-tuning backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


def chunk_corpus(corpus_path: Path, chars_per_chunk: int = 8192) -> list[str]:
    """Split corpus text into fixed-size chunks, skipping near-empty ones."""
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for i in range(0, len(text), chars_per_chunk):
        chunk = text[i : i + chars_per_chunk].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


def write_jsonl(
    chunks: list[str],
    output_path: Path,
    format: Literal["pretraining", "instruction"] = "pretraining",
) -> int:
    """Write text chunks to JSONL for Bedrock fine-tuning.

    pretraining  → {"input": "<text>"}            (CONTINUED_PRE_TRAINING)
    instruction  → {"prompt": ..., "completion": ...}  (FINE_TUNING, split at midpoint)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if format == "pretraining":
                record: dict[str, str] = {"input": chunk}
            else:
                mid = len(chunk) // 2
                record = {"prompt": chunk[:mid], "completion": chunk[mid:]}
            f.write(json.dumps(record) + "\n")
    return len(chunks)
