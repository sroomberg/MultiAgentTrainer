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

    def test_name_defaults_to_run_id(self) -> None:
        runner = Runner(AutoresearchConfig(), TrainingConfig())
        assert runner.name == runner.run_id

    def test_name_is_set_when_provided(self) -> None:
        runner = Runner(AutoresearchConfig(), TrainingConfig(), name="llama3")
        assert runner.name == "llama3"


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


class TestMultiMachineRunner:
    async def test_distributes_experiments_evenly(self) -> None:
        """Experiments are split evenly across machines."""
        from multiagenttrainer.config import ExecutionConfig, MachineConfig
        from multiagenttrainer.executor import Executor, RunResult

        calls: list[str] = []

        class FakeExecutor(Executor):
            def __init__(self, name: str) -> None:
                self.name = name

            async def upload(self, local_dir, remote_dir):  # type: ignore[override]
                pass

            async def run(self, cmd, cwd, timeout):  # type: ignore[override]
                calls.append(self.name)
                return RunResult(exit_code=0, stdout="val_bpb=1.0", stderr="")

        machines = [
            MachineConfig(name="m1", execution=ExecutionConfig(type="local")),
            MachineConfig(name="m2", execution=ExecutionConfig(type="local")),
        ]

        import tempfile
        import unittest.mock as mock
        from pathlib import Path

        def fake_build(mc):
            return FakeExecutor(mc.name)

        patch_target = "multiagenttrainer.runner.build_executor_for_machine"
        with (
            mock.patch(patch_target, side_effect=fake_build),
            tempfile.TemporaryDirectory() as td,
        ):
            runner = Runner(
                AutoresearchConfig(),
                TrainingConfig(max_experiments=4, output_dir=td),
                machines=machines,
            )
            Path(td, "autoresearch").mkdir(parents=True)
            results = await runner.run_experiments(Path(td))

        assert len(results) == 4
        # Each machine should have run exactly 2 experiments
        assert calls.count("m1") == 2
        assert calls.count("m2") == 2
        # Results are sorted by experiment_id
        assert [r.experiment_id for r in results] == [1, 2, 3, 4]
        # machine field is set
        assert all(r.machine in ("m1", "m2") for r in results)

    async def test_odd_experiment_count_distributed(self) -> None:
        """With 3 experiments and 2 machines, first machine gets one extra."""
        from multiagenttrainer.config import ExecutionConfig, MachineConfig
        from multiagenttrainer.executor import Executor, RunResult

        counts: dict[str, int] = {"m1": 0, "m2": 0}

        class FakeExecutor(Executor):
            def __init__(self, name: str) -> None:
                self.name = name

            async def upload(self, local_dir, remote_dir):  # type: ignore[override]
                pass

            async def run(self, cmd, cwd, timeout):  # type: ignore[override]
                counts[self.name] += 1
                return RunResult(exit_code=0, stdout="", stderr="")

        machines = [
            MachineConfig(name="m1", execution=ExecutionConfig(type="local")),
            MachineConfig(name="m2", execution=ExecutionConfig(type="local")),
        ]
        import tempfile
        import unittest.mock as mock
        from pathlib import Path

        def fake_build(mc):
            return FakeExecutor(mc.name)

        patch_target = "multiagenttrainer.runner.build_executor_for_machine"
        with (
            mock.patch(patch_target, side_effect=fake_build),
            tempfile.TemporaryDirectory() as td,
        ):
            runner = Runner(
                AutoresearchConfig(),
                TrainingConfig(max_experiments=3, output_dir=td),
                machines=machines,
            )
            Path(td, "autoresearch").mkdir(parents=True)
            await runner.run_experiments(Path(td))

        assert counts["m1"] == 2
        assert counts["m2"] == 1
