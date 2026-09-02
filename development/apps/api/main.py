"""FastAPI composition root for the offline tutoring demonstration."""

from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from socratic_tutor import __version__
from socratic_tutor.contracts import (
    BenchmarkReplaySnapshot,
    CreateSessionRequest,
    ExperimentSummaryArtifact,
    LiveEvaluationSnapshot,
    LiveEvaluationStatus,
    SessionSnapshot,
    StartBenchmarkReplayRequest,
    StartLiveEvaluationRequest,
    SubmitTurnRequest,
    TaskView,
)
from socratic_tutor.demo_replay import (
    BenchmarkReplayService,
    ExperimentSummaryService,
    ReplayArtifactError,
    ReplayNotStartedError,
)
from socratic_tutor.live_evaluation import (
    LiveEvaluationCaseError,
    LiveEvaluationConflictError,
    LiveEvaluationNotFoundError,
    LiveEvaluationService,
    LiveEvaluationUnavailableError,
    create_live_evaluation_service,
)
from socratic_tutor.sessions import (
    IdempotencyConflictError,
    SessionNotFoundError,
    SessionService,
)
from socratic_tutor.settings import Settings
from socratic_tutor.tasks import list_tasks
from socratic_tutor.trajectories import JsonlEventStore


class HealthResponse(BaseModel):
    """Stable health contract used by the frontend and smoke tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    service: str
    version: str


def _session_service(request: Request) -> SessionService:
    return cast(SessionService, request.app.state.session_service)


def _benchmark_replay_service(request: Request) -> BenchmarkReplayService:
    return cast(BenchmarkReplayService, request.app.state.benchmark_replay_service)


def _experiment_summary_service(request: Request) -> ExperimentSummaryService:
    return cast(ExperimentSummaryService, request.app.state.experiment_summary_service)


def _live_evaluation_service(request: Request) -> LiveEvaluationService:
    return cast(LiveEvaluationService, request.app.state.live_evaluation_service)


def create_app(
    event_log_path: Path | None = None,
    benchmark_replay_path: Path | None = None,
    experiment_summary_path: Path | None = None,
    live_evaluation_service: LiveEvaluationService | None = None,
) -> FastAPI:
    """Build an app with an injectable event log for tests and local recovery."""

    settings = Settings()
    path = event_log_path if event_log_path is not None else settings.event_log_path
    replay_path = (
        benchmark_replay_path
        if benchmark_replay_path is not None
        else settings.benchmark_replay_path
    )
    summary_path = (
        experiment_summary_path
        if experiment_summary_path is not None
        else settings.experiment_summary_path
    )
    application = FastAPI(title="Socratic Tutor API", version=__version__)
    application.state.session_service = SessionService(JsonlEventStore(path))
    application.state.benchmark_replay_service = BenchmarkReplayService(replay_path)
    application.state.experiment_summary_service = ExperimentSummaryService(summary_path)
    application.state.live_evaluation_service = (
        live_evaluation_service
        if live_evaluation_service is not None
        else create_live_evaluation_service(settings)
    )

    @application.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:  # pyright: ignore[reportUnusedFunction]
        return HealthResponse(
            status="ok",
            service="socratic-tutor-api",
            version=__version__,
        )

    @application.post(
        "/api/sessions",
        response_model=SessionSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(  # pyright: ignore[reportUnusedFunction]
        body: CreateSessionRequest, request: Request
    ) -> SessionSnapshot:
        try:
            return _session_service(request).create(body)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Task not found") from error
        except IdempotencyConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/tasks", response_model=list[TaskView])
    async def get_tasks() -> list[TaskView]:  # pyright: ignore[reportUnusedFunction]
        return [task.public_view() for task in list_tasks()]

    @application.get("/api/sessions/{session_id}", response_model=SessionSnapshot)
    async def get_session(  # pyright: ignore[reportUnusedFunction]
        session_id: str, request: Request
    ) -> SessionSnapshot:
        try:
            return _session_service(request).get(session_id)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error

    @application.post("/api/sessions/{session_id}/turns", response_model=SessionSnapshot)
    async def submit_turn(  # pyright: ignore[reportUnusedFunction]
        session_id: str,
        body: SubmitTurnRequest,
        request: Request,
    ) -> SessionSnapshot:
        try:
            return _session_service(request).submit(session_id, body)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error
        except IdempotencyConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post(
        "/api/benchmark-replays",
        response_model=BenchmarkReplaySnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    async def start_benchmark_replay(  # pyright: ignore[reportUnusedFunction]
        body: StartBenchmarkReplayRequest,
        request: Request,
    ) -> BenchmarkReplaySnapshot:
        try:
            return _benchmark_replay_service(request).start(body)
        except ReplayArtifactError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Benchmark replay is unavailable",
            ) from error

    @application.get("/api/experiment-summary", response_model=ExperimentSummaryArtifact)
    async def get_experiment_summary(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> ExperimentSummaryArtifact:
        try:
            return _experiment_summary_service(request).get()
        except ReplayArtifactError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment summary is unavailable",
            ) from error

    @application.post(
        "/api/benchmark-replays/{replay_id}/reveal",
        response_model=BenchmarkReplaySnapshot,
    )
    async def reveal_benchmark_outcome(  # pyright: ignore[reportUnusedFunction]
        replay_id: str,
        request: Request,
    ) -> BenchmarkReplaySnapshot:
        try:
            return _benchmark_replay_service(request).reveal(replay_id)
        except ReplayNotStartedError as error:
            raise HTTPException(status_code=404, detail="Benchmark replay not found") from error
        except ReplayArtifactError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Benchmark replay is unavailable",
            ) from error

    @application.get(
        "/api/live-evaluations/status",
        response_model=LiveEvaluationStatus,
    )
    async def get_live_evaluation_status(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> LiveEvaluationStatus:
        return _live_evaluation_service(request).status()

    @application.post(
        "/api/live-evaluations",
        response_model=LiveEvaluationSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    async def start_live_evaluation(  # pyright: ignore[reportUnusedFunction]
        body: StartLiveEvaluationRequest,
        request: Request,
    ) -> LiveEvaluationSnapshot:
        try:
            return await _live_evaluation_service(request).start(body)
        except LiveEvaluationUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except LiveEvaluationCaseError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except LiveEvaluationConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post(
        "/api/live-evaluations/{run_id}/reveal",
        response_model=LiveEvaluationSnapshot,
    )
    async def reveal_live_evaluation(  # pyright: ignore[reportUnusedFunction]
        run_id: str,
        request: Request,
    ) -> LiveEvaluationSnapshot:
        try:
            return await _live_evaluation_service(request).reveal(run_id)
        except LiveEvaluationUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except LiveEvaluationNotFoundError as error:
            raise HTTPException(status_code=404, detail="Live evaluation not found") from error

    return application


app = create_app()
