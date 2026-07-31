"""Network-blocked Modal boundary for untrusted benchmark code execution."""

from __future__ import annotations

from typing import Literal, Protocol

import modal
from pydantic import Field

from socratic_tutor.contracts import ContractModel


class SandboxExecutionRequest(ContractModel):
    """One command already prepared for isolated execution."""

    command: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(ge=1, le=60)
    max_output_characters: int = Field(ge=256, le=65_536)


class SandboxExecutionResult(ContractModel):
    """Observable result of one isolated execution attempt."""

    provider: Literal["modal"] = "modal"
    status: Literal["completed", "failed"]
    sandbox_id: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    error_message: str | None = None


class ModalSandboxConfig(ContractModel):
    """Fixed limits for short benchmark submissions, not general compute jobs."""

    app_name: str = "socratic-tutor-benchmark-sandbox"
    python_version: str = "3.12"
    cpu_request: float = Field(default=0.25, gt=0.0, le=1.0)
    cpu_limit: float = Field(default=0.5, gt=0.0, le=1.0)
    memory_request_mib: int = Field(default=256, ge=128, le=1024)
    memory_limit_mib: int = Field(default=512, ge=128, le=2048)


class ModalProcess(Protocol):
    """Minimal Modal process shape, kept small for network-free tests."""

    @property
    def stdout(self) -> ModalOutputStream: ...

    @property
    def stderr(self) -> ModalOutputStream: ...

    @property
    def object_id(self) -> str: ...

    def wait(self) -> int: ...


class ModalOutputStream(Protocol):
    """Readable process output stream."""

    def read(self) -> str: ...


class ModalSandboxClient(Protocol):
    """SDK seam that prevents unit tests from creating a remote sandbox."""

    def create(
        self,
        request: SandboxExecutionRequest,
        config: ModalSandboxConfig,
    ) -> ModalProcess: ...


class ModalSdkSandboxClient:
    """Current Modal SDK implementation with no secrets, mounts, or network access."""

    def create(
        self,
        request: SandboxExecutionRequest,
        config: ModalSandboxConfig,
    ) -> ModalProcess:
        app = modal.App.lookup(config.app_name, create_if_missing=True)
        image = modal.Image.debian_slim(python_version=config.python_version)
        return modal.Sandbox.create(
            *request.command,
            app=app,
            image=image,
            timeout=request.timeout_seconds,
            cpu=(config.cpu_request, config.cpu_limit),
            memory=(config.memory_request_mib, config.memory_limit_mib),
            block_network=True,
        )


class ModalSandboxExecutor:
    """Execute untrusted code remotely, never through a local fallback."""

    def __init__(
        self,
        *,
        config: ModalSandboxConfig | None = None,
        client: ModalSandboxClient | None = None,
    ) -> None:
        self._config = config or ModalSandboxConfig()
        self._client = client or ModalSdkSandboxClient()

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """Run one bounded command and retain a bounded observable transcript."""

        try:
            process = self._client.create(request, self._config)
            exit_code = process.wait()
            stdout = _bounded(process.stdout.read(), request.max_output_characters)
            stderr = _bounded(process.stderr.read(), request.max_output_characters)
        except Exception as error:
            return SandboxExecutionResult(
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        return SandboxExecutionResult(
            status="completed" if exit_code == 0 else "failed",
            sandbox_id=process.object_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )


def _bounded(value: str, limit: int) -> str:
    return value[:limit]
