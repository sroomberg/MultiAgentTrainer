"""Tests for the runner and report modules."""

from __future__ import annotations

from multiagenttrainer.config import (
    AutoresearchConfig,
    TrainerConfig,
    TrainingConfig,
)
from multiagenttrainer.report import generate_report
from multiagenttrainer.runner import ExperimentResult, Runner


class TestRunner:
    def test_parse_val_bpb(self) -> None:
        assert Runner._parse_val_bpb("val_bpb=1.234") == 1.234
        assert Runner._parse_val_bpb("val_bpb: 0.99") == 0.99
        assert Runner._parse_val_bpb("no metric here") is None

    def test_parse_val_bpb_takes_last(self) -> None:
        output = "val_bpb=2.0\nval_bpb=1.5\nval_bpb=1.2"
        assert Runner._parse_val_bpb(output) == 1.2

    def test_run_id_is_set(self) -> None:
        runner = Runner(AutoresearchConfig(), TrainingConfig())
        assert len(runner.run_id) == 8


class TestReport:
    def test_generate_report(self) -> None:
        results = [
            ExperimentResult(
                experiment_id=1,
                exit_code=0,
                duration=60.5,
                val_bpb=1.234,
            ),
            ExperimentResult(
                experiment_id=2,
                exit_code=1,
                duration=30.0,
                error="Timed out",
            ),
        ]
        cfg = TrainerConfig()
        report = generate_report("abc123", cfg, results)

        assert "abc123" in report
        assert "1.234" in report
        assert "Timed out" in report
        assert "Best Result" in report
        assert "Experiment 1" in report

    def test_report_no_results(self) -> None:
        cfg = TrainerConfig()
        report = generate_report("empty", cfg, [])
        assert "empty" in report
        assert "Total experiments: 0" in report
