"""Execution targets: local subprocess, SSH remote host, or Docker container."""

from __future__ import annotations

import asyncio
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExecutionConfig, MachineConfig


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    error: str | None = None


class Executor(ABC):
    """Run commands and transfer files to an execution target."""

    @abstractmethod
    async def upload(self, local_dir: Path, remote_dir: str) -> None:
        """Copy *local_dir* contents to *remote_dir* on the target."""

    @abstractmethod
    async def run(self, cmd: str, cwd: str, timeout: float) -> RunResult:
        """Run *cmd* in *cwd* on the target with a *timeout* (seconds)."""


class LocalExecutor(Executor):
    """Run commands in a local subprocess — the default behaviour."""

    async def upload(self, local_dir: Path, remote_dir: str) -> None:
        # Workspace is already local; nothing to transfer.
        pass

    async def run(self, cmd: str, cwd: str, timeout: float) -> RunResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
            )
            return await _communicate(proc, timeout)
        except Exception as exc:
            return RunResult(exit_code=-1, stdout="", stderr="", error=str(exc))


class SSHExecutor(Executor):
    """Run commands on a remote host via SSH, uploading files with rsync."""

    def __init__(self, host: str, remote_base: str, ssh_key: str | None = None) -> None:
        self.host = host
        self.remote_base = remote_base.rstrip("/")
        self._ssh_opts = _ssh_opts(ssh_key)

    async def upload(self, local_dir: Path, remote_dir: str) -> None:
        src = str(local_dir).rstrip("/") + "/"
        rsync_cmd = (
            ["rsync", "-az", "--delete"]
            + (["-e", f"ssh {self._ssh_opts}"] if self._ssh_opts else ["-e", "ssh"])
            + [src, f"{self.host}:{remote_dir}/"]
        )
        proc = await asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_raw = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"rsync failed (exit {proc.returncode}): "
                f"{stderr_raw.decode('utf-8', errors='replace')}"
            )

    async def run(self, cmd: str, cwd: str, timeout: float) -> RunResult:
        remote_cmd = f"cd {shlex.quote(cwd)} && {cmd}"
        ssh_args = self._ssh_opts.split() if self._ssh_opts else []
        ssh_cmd = ["ssh", *ssh_args, self.host, remote_cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            return await _communicate(proc, timeout)
        except Exception as exc:
            return RunResult(exit_code=-1, stdout="", stderr="", error=str(exc))


class DockerExecutor(Executor):
    """Run commands inside a running Docker container."""

    def __init__(self, container: str, container_base: str) -> None:
        self.container = container
        self.container_base = container_base.rstrip("/")

    async def upload(self, local_dir: Path, remote_dir: str) -> None:
        # Ensure destination directory exists in the container.
        mkdir_proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            self.container,
            "mkdir",
            "-p",
            remote_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await mkdir_proc.communicate()

        src = str(local_dir).rstrip("/") + "/."
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "cp",
            src,
            f"{self.container}:{remote_dir}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_raw = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker cp failed (exit {proc.returncode}): "
                f"{stderr_raw.decode('utf-8', errors='replace')}"
            )

    async def run(self, cmd: str, cwd: str, timeout: float) -> RunResult:
        docker_cmd = ["docker", "exec", "-w", cwd, self.container, "sh", "-c", cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            return await _communicate(proc, timeout)
        except Exception as exc:
            return RunResult(exit_code=-1, stdout="", stderr="", error=str(exc))


async def _communicate(
    proc: asyncio.subprocess.Process, timeout: float
) -> RunResult:
    """Wait for *proc* to finish, handling timeout and output decoding."""
    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        return RunResult(exit_code=-1, stdout="", stderr="", error="Timed out")
    return RunResult(
        exit_code=proc.returncode or 0,
        stdout=stdout_raw.decode("utf-8", errors="replace"),
        stderr=stderr_raw.decode("utf-8", errors="replace"),
    )


def _ssh_opts(key_path: str | None) -> str:
    opts = "-o StrictHostKeyChecking=no -o BatchMode=yes"
    if key_path:
        opts += f" -i {key_path}"
    return opts


def build_executor(execution_cfg: ExecutionConfig) -> Executor:
    """Construct the right Executor from config."""
    t = execution_cfg.type
    if t == "ssh":
        return SSHExecutor(
            host=execution_cfg.ssh_host,  # type: ignore[arg-type]
            remote_base=execution_cfg.remote_dir,
            ssh_key=execution_cfg.ssh_key,
        )
    if t == "docker":
        return DockerExecutor(
            container=execution_cfg.container,  # type: ignore[arg-type]
            container_base=execution_cfg.container_dir,
        )
    return LocalExecutor()


def build_executor_for_machine(machine_cfg: MachineConfig) -> Executor:
    """Construct the right Executor for a named MachineConfig."""
    return build_executor(machine_cfg.execution)
