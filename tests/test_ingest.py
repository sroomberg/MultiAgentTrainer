"""Tests for the ingestion layer."""

from __future__ import annotations

from pathlib import Path

from multiagenttrainer.ingest import Ingester
from multiagenttrainer.sources import LocalRepoSource


def test_fetch_all_and_build_corpus(
    local_repo: Path, tmp_staging: Path, tmp_path: Path
) -> None:
    src = LocalRepoSource(str(local_repo))
    ingester = Ingester([src], tmp_staging)

    fetched = ingester.fetch_all()
    assert len(fetched) == 1

    corpus = tmp_path / "corpus.txt"
    count = ingester.build_corpus(corpus)
    assert count >= 2  # main.py + README.md
    assert corpus.exists()

    text = corpus.read_text()
    assert "### FILE:" in text
    assert "print('hello')" in text


def test_exclude_filter(local_repo: Path, tmp_staging: Path, tmp_path: Path) -> None:
    src = LocalRepoSource(str(local_repo))
    ingester = Ingester([src], tmp_staging, exclude=["**/README.md"])
    ingester.fetch_all()

    corpus = tmp_path / "corpus.txt"
    count = ingester.build_corpus(corpus)
    assert count == 1  # only main.py

    text = corpus.read_text()
    assert "main.py" in text
    assert "README.md" not in text


def test_empty_sources(tmp_staging: Path, tmp_path: Path) -> None:
    ingester = Ingester([], tmp_staging)
    fetched = ingester.fetch_all()
    assert fetched == []

    corpus = tmp_path / "corpus.txt"
    count = ingester.build_corpus(corpus)
    assert count == 0
