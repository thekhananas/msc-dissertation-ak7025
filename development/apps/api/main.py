"""FastAPI composition root for the development workspace."""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from socratic_tutor import __version__


class HealthResponse(BaseModel):
    """Stable health contract used by the frontend and smoke tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    service: str
    version: str


app = FastAPI(title="Socratic Tutor API", version=__version__)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return process health without touching external services."""

    return HealthResponse(
        status="ok",
        service="socratic-tutor-api",
        version=__version__,
    )
