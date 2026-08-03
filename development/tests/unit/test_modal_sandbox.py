"""Network-free tests for the Modal sandbox trust boundary."""

from socratic_tutor.sandbox import (
    ModalSandboxConfig,
    ModalSandboxExecutor,
    SandboxExecutionRequest,
)


class FakeStream:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value


class FakeProcess:
    def __init__(self, exit_code: int, *, stdout: str = "", stderr: str = "") -> None:
        self._exit_code = exit_code
        self._stdout = FakeStream(stdout)
        self._stderr = FakeStream(stderr)
        self.object_id = "sb-test-001"

    @property
    def stdout(self) -> FakeStream:
        return self._stdout

    @property
    def stderr(self) -> FakeStream:
        return self._stderr

    @property
    def returncode(self) -> int | None:
        return self._exit_code

    def wait(self) -> None:
        return None


class FakeClient:
    def __init__(self, process: FakeProcess) -> None:
        self._process = process
        self.request: SandboxExecutionRequest | None = None
        self.config: ModalSandboxConfig | None = None

    def create(self, request: SandboxExecutionRequest, config: ModalSandboxConfig) -> FakeProcess:
        self.request = request
        self.config = config
        return self._process


def test_modal_executor_uses_remote_boundary_and_bounds_visible_output() -> None:
    client = FakeClient(FakeProcess(0, stdout="x" * 300, stderr="warning"))
    result = ModalSandboxExecutor(client=client).execute(
        SandboxExecutionRequest(
            command=("python", "-I", "-c", "print('checked')"),
            timeout_seconds=15,
            max_output_characters=256,
        )
    )

    assert result.status == "completed"
    assert result.sandbox_id == "sb-test-001"
    assert len(result.stdout) == 256
    assert result.stderr == "warning"
    assert client.request is not None
    assert client.request.command[:2] == ("python", "-I")
    assert client.config is not None
    assert client.config.cpu_limit == 0.5
    assert client.config.memory_limit_mib == 512


def test_modal_executor_reports_remote_failure_without_local_fallback() -> None:
    class FailingClient:
        def create(
            self,
            request: SandboxExecutionRequest,
            config: ModalSandboxConfig,
        ) -> FakeProcess:
            del request, config
            raise RuntimeError("modal token is unavailable")

    result = ModalSandboxExecutor(client=FailingClient()).execute(
        SandboxExecutionRequest(
            command=("python", "-I", "-c", "print('never local')"),
            timeout_seconds=15,
            max_output_characters=256,
        )
    )

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert "token is unavailable" in (result.error_message or "")


def test_modal_executor_rejects_missing_exit_code_after_remote_wait() -> None:
    class MissingExitCodeProcess(FakeProcess):
        @property
        def returncode(self) -> None:
            return None

    result = ModalSandboxExecutor(client=FakeClient(MissingExitCodeProcess(0))).execute(
        SandboxExecutionRequest(
            command=("python", "-I", "-c", "print('checked')"),
            timeout_seconds=15,
            max_output_characters=256,
        )
    )

    assert result.status == "failed"
    assert result.error_type == "ModalExitCodeUnavailable"
