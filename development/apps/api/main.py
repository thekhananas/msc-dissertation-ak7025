"""FastAPI composition root for the offline tutoring demonstration."""

from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from socratic_tutor import __version__
from socratic_tutor.contracts import CreateSessionRequest, SessionSnapshot, SubmitTurnRequest
from socratic_tutor.sessions import (
    IdempotencyConflictError,
    SessionNotFoundError,
    SessionService,
)
from socratic_tutor.settings import Settings
from socratic_tutor.trajectories import JsonlEventStore


class HealthResponse(BaseModel):
    """Stable health contract used by the frontend and smoke tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    service: str
    version: str


def _session_service(request: Request) -> SessionService:
    return cast(SessionService, request.app.state.session_service)


def create_app(event_log_path: Path | None = None) -> FastAPI:
    """Build an app with an injectable event log for tests and local recovery."""

    settings = Settings()
    path = event_log_path if event_log_path is not None else settings.event_log_path
    application = FastAPI(title="Socratic Tutor API", version=__version__)
    application.state.session_service = SessionService(JsonlEventStore(path))

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
        return _session_service(request).create(body)

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

    return application


app = create_app()
