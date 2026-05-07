"""Tests for multiagenttrainer.progress."""

from __future__ import annotations

from pathlib import Path

from multiagenttrainer.progress import (
    find_progress_files,
    new_progress,
    read_progress,
    write_progress,
)


class TestRunProgress:
    def test_current_experiment_reflects_list_length(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        assert p.current_experiment == 0
        p.experiments.append({"experiment_id": 1, "val_bpb": 3.5, "exit_code": 0})
        assert p.current_experiment == 1

    def test_best_val_bpb_returns_minimum(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        p.experiments = [
            {"experiment_id": 1, "val_bpb": 3.5},
            {"experiment_id": 2, "val_bpb": 3.2},
            {"experiment_id": 3, "val_bpb": 3.8},
        ]
        assert p.best_val_bpb == 3.2

    def test_best_val_bpb_none_when_no_results(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        assert p.best_val_bpb is None

    def test_best_val_bpb_skips_none_values(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        p.experiments = [
            {"experiment_id": 1, "val_bpb": None},
            {"experiment_id": 2, "val_bpb": 3.4},
        ]
        assert p.best_val_bpb == 3.4

    def test_last_val_bpb_returns_most_recent(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        p.experiments = [
            {"experiment_id": 1, "val_bpb": 3.5},
            {"experiment_id": 2, "val_bpb": 3.2},
        ]
        assert p.last_val_bpb == 3.2

    def test_last_val_bpb_skips_trailing_none(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        p.experiments = [
            {"experiment_id": 1, "val_bpb": 3.5},
            {"experiment_id": 2, "val_bpb": None},
        ]
        assert p.last_val_bpb == 3.5

    def test_last_val_bpb_none_when_no_results(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        assert p.last_val_bpb is None

    def test_new_progress_defaults(self) -> None:
        p = new_progress("abc123", "llama3", 50)
        assert p.run_id == "abc123"
        assert p.name == "llama3"
        assert p.max_experiments == 50
        assert p.status == "running"
        assert p.experiments == []


class TestWriteReadProgress:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = new_progress("abc123", "llama3", 50)
        p.experiments.append(
            {"experiment_id": 1, "val_bpb": 3.5, "exit_code": 0, "error": None}
        )
        path = tmp_path / "progress-abc123.json"
        write_progress(path, p)

        loaded = read_progress(path)
        assert loaded is not None
        assert loaded.run_id == "abc123"
        assert loaded.name == "llama3"
        assert loaded.status == "running"
        assert loaded.current_experiment == 1
        assert loaded.best_val_bpb == 3.5

    def test_read_progress_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        result = read_progress(tmp_path / "nonexistent.json")
        assert result is None

    def test_read_progress_returns_none_for_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "progress-bad.json"
        path.write_text("not valid json {{{")
        assert read_progress(path) is None

    def test_status_persisted(self, tmp_path: Path) -> None:
        p = new_progress("abc123", "llama3", 10)
        p.status = "done"
        path = tmp_path / "progress-abc123.json"
        write_progress(path, p)
        assert read_progress(path).status == "done"


class TestFindProgressFiles:
    def test_finds_progress_files(self, tmp_path: Path) -> None:
        (tmp_path / "progress-aaa.json").write_text("{}")
        (tmp_path / "progress-bbb.json").write_text("{}")
        (tmp_path / "report-aaa.md").write_text("")

        files = find_progress_files(tmp_path)
        names = [f.name for f in files]
        assert "progress-aaa.json" in names
        assert "progress-bbb.json" in names
        assert "report-aaa.md" not in names

    def test_returns_empty_for_no_matches(self, tmp_path: Path) -> None:
        assert find_progress_files(tmp_path) == []

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "progress-zzz.json").write_text("{}")
        (tmp_path / "progress-aaa.json").write_text("{}")
        files = find_progress_files(tmp_path)
        assert files[0].name == "progress-aaa.json"
        assert files[1].name == "progress-zzz.json"
