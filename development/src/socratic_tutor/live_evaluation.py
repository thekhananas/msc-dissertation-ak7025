"""Optional live illustration of prediction before later-task outcome reveal."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import Field, SecretStr

from socratic_tutor.benchmark.cerebras import (
    CerebrasGateway,
    CerebrasGatewayConfig,
    HttpxCerebrasTransport,
)
from socratic_tutor.benchmark.evaluator.loader import (
    load_and_verify_manifest,
    load_authored_manifest,
)
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
    StudentGenerationRequest,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.benchmark.public import (
    BenchmarkSplit,
    PublicBenchmarkManifest,
    PublicChannelRequestBuilder,
    evidence_from_test_counts,
)
from socratic_tutor.benchmark.public.offline import load_initial_tracker_state
from socratic_tutor.contracts import (
    ContractModel,
    Evidence,
    EvidenceCategory,
    LiveDialogueAssessmentView,
    LiveEvaluationPhase,
    LiveEvaluationSnapshot,
    LiveEvaluationStatus,
    LiveExecutionView,
    LiveFailureStage,
    LiveFailureView,
    LiveOutcomeView,
    LivePredictionView,
    StartLiveEvaluationRequest,
)
from socratic_tutor.sandbox import (
    ModalSandboxExecutor,
    SandboxExecutor,
    SandboxTestResult,
    execute_authored_python_tests,
    extract_python_submission,
)
from socratic_tutor.settings import Settings
from socratic_tutor.tracking import update_tracker

_MODEL_ROUTE = ModelRoute(provider="cerebras", model="gpt-oss-120b")
_POLICY_THRESHOLD = 0.60
_SAMPLING = SamplingConfig(temperature=0.0, max_output_tokens=512, seed=20260911)
_CLAIM_BOUNDARY = (
    "Illustration only. This live development run is not canonical research evidence and does "
    "not measure student learning or tutoring quality."
)


class LiveEvaluationUnavailableError(RuntimeError):
    """The optional route is disabled or lacks a required local dependency."""


class LiveEvaluationCaseError(ValueError):
    """A request tried to use a case outside the single configured allowlist."""


class LiveEvaluationConflictError(ValueError):
    """An idempotency key was reused for a different request."""


class LiveEvaluationNotFoundError(KeyError):
    """The requested in-memory run does not exist in this process."""


class LiveModelResponse(ContractModel):
    """Small provider result used by the live orchestration service."""

    final_response: str = Field(min_length=1)
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_ms: int = Field(ge=0)


class LiveModelGateway(Protocol):
    """Provider boundary required by the live illustration."""

    async def generate(self, request: StudentGenerationRequest) -> LiveModelResponse: ...


class DirectCerebrasLiveGateway:
    """Use one short-lived HTTP client for each bounded demonstration request."""

    def __init__(self, api_key: SecretStr) -> None:
        self._api_key = api_key

    async def generate(self, request: StudentGenerationRequest) -> LiveModelResponse:
        async with httpx.AsyncClient() as client:
            gateway = CerebrasGateway(
                config=CerebrasGatewayConfig(
                    api_key=self._api_key,
                    reasoning_effort="low",
                    max_attempts=1,
                ),
                transport=HttpxCerebrasTransport(client),
            )
            result = await gateway.generate(request)
        metadata = result.response.provider_metadata
        return LiveModelResponse(
            final_response=result.response.final_response,
            response_hash=result.response.response_hash,
            latency_ms=metadata.latency_ms or 0,
        )


@dataclass(frozen=True, slots=True)
class LiveEvaluationAssets:
    manifest_path: Path
    benchmark_root: Path
    public_manifest: PublicBenchmarkManifest
    system_prompt: str
    system_prompt_hash: str


type Clock = Callable[[], datetime]


class LiveEvaluationService:
    """Coordinate one non-canonical live run without writing research artefacts."""

    def __init__(
        self,
        *,
        enabled: bool,
        case_id: str,
        assets: LiveEvaluationAssets | None = None,
        gateway: LiveModelGateway | None = None,
        executor: SandboxExecutor | None = None,
        unavailable_reason: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._enabled = enabled
        self._case_id = case_id
        self._assets = assets
        self._gateway = gateway
        self._executor = executor
        self._unavailable_reason = unavailable_reason
        self._clock = clock or (lambda: datetime.now(UTC))
        self._runs: dict[str, LiveEvaluationSnapshot] = {}
        self._request_identity: dict[str, tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    def status(self) -> LiveEvaluationStatus:
        available = (
            self._enabled
            and self._assets is not None
            and self._gateway is not None
            and self._executor is not None
        )
        reason = None if available else self._unavailable_reason or "Live illustration is disabled."
        return LiveEvaluationStatus(
            enabled=self._enabled,
            available=available,
            case_id=self._case_id,
            evaluation_model=f"{_MODEL_ROUTE.provider}/{_MODEL_ROUTE.model}",
            reason=reason,
            claim_boundary=_CLAIM_BOUNDARY,
        )

    async def start(self, request: StartLiveEvaluationRequest) -> LiveEvaluationSnapshot:
        """Generate visible evidence and commit two predictions exactly once."""

        self._require_available()
        if request.case_id != self._case_id:
            raise LiveEvaluationCaseError("Only the configured development case is available")
        run_id = _run_id(request.idempotency_key, request.case_id)
        async with self._lock:
            previous = self._request_identity.get(request.idempotency_key)
            identity = (request.case_id, run_id)
            if previous is not None and previous != identity:
                raise LiveEvaluationConflictError(
                    "Idempotency key was already used for another live request"
                )
            existing = self._runs.get(run_id)
            if existing is not None:
                return existing
            self._request_identity[request.idempotency_key] = identity
            snapshot = await self._start_locked(run_id)
            self._runs[run_id] = snapshot
            return snapshot

    async def reveal(self, run_id: str) -> LiveEvaluationSnapshot:
        """Request the later task only after the prediction commitment exists."""

        self._require_available()
        async with self._lock:
            try:
                committed = self._runs[run_id]
            except KeyError as error:
                raise LiveEvaluationNotFoundError(run_id) from error
            if committed.phase in {
                LiveEvaluationPhase.OUTCOME_REVEALED,
                LiveEvaluationPhase.FAILED,
            }:
                return committed
            revealed = await self._reveal_locked(committed)
            self._runs[run_id] = revealed
            return revealed

    async def _start_locked(self, run_id: str) -> LiveEvaluationSnapshot:
        assets, gateway, executor = self._dependencies()
        builder = PublicChannelRequestBuilder(assets.benchmark_root, assets.public_manifest)
        spec = self._generation_spec(run_id)
        model_calls = 0
        sandbox_calls = 0
        latency_ms = 0

        try:
            public_request = builder.build_public(case_id=self._case_id, spec=spec)
            model_calls += 1
            public = await gateway.generate(public_request)
            latency_ms += public.latency_ms
        except Exception:
            return self._failed(
                run_id=run_id,
                stage=LiveFailureStage.PUBLIC_GENERATION,
                message="The evaluation model could not produce the visible answer.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
            )

        try:
            evidence_request = builder.build_evidence(case_id=self._case_id, spec=spec)
            model_calls += 1
            evidence = await gateway.generate(evidence_request)
            latency_ms += evidence.latency_ms
        except Exception:
            return self._failed(
                run_id=run_id,
                stage=LiveFailureStage.EVIDENCE_GENERATION,
                message="The evaluation model could not produce the separate coding evidence.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
                public_response=public.final_response,
            )

        try:
            extract_python_submission(evidence.final_response)
        except ValueError:
            return self._failed(
                run_id=run_id,
                stage=LiveFailureStage.EVIDENCE_EXTRACTION,
                message="The coding evidence did not contain one executable Python block.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
                public_response=public.final_response,
                evidence_response=evidence.final_response,
            )

        case = next(item for item in assets.public_manifest.cases if item.case_id == self._case_id)
        sandbox_calls += 1
        result = await asyncio.to_thread(
            execute_authored_python_tests,
            executor,
            response=evidence.final_response,
            bundle_path=assets.benchmark_root / case.evidence_test_ref,
        )
        evidence_execution = _execution_view(result)
        if evidence_execution is None:
            return self._failed(
                run_id=run_id,
                stage=LiveFailureStage.EVIDENCE_EXECUTION,
                message="The remote coding check did not complete.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
                public_response=public.final_response,
                evidence_response=evidence.final_response,
            )

        assessment, public_evidence = _assess_public_response(self._case_id, public.final_response)
        committed_at = self._clock()
        predictions = _predictions(
            assets=assets,
            case_id=self._case_id,
            run_id=run_id,
            public_evidence=public_evidence,
            passed=evidence_execution.passed_checks,
            failed=evidence_execution.failed_checks,
            committed_at=committed_at,
        )
        commitment_hash = canonical_sha256(
            {
                "run_id": run_id,
                "case_id": self._case_id,
                "public_response_hash": public.response_hash,
                "evidence_response_hash": evidence.response_hash,
                "evidence_execution": evidence_execution,
                "predictions": predictions,
            }
        )
        return LiveEvaluationSnapshot(
            run_id=run_id,
            case_id=self._case_id,
            phase=LiveEvaluationPhase.PREDICTIONS_COMMITTED,
            evaluation_model=f"{_MODEL_ROUTE.provider}/{_MODEL_ROUTE.model}",
            public_response=public.final_response,
            public_assessment=assessment,
            evidence_response=evidence.final_response,
            evidence_execution=evidence_execution,
            predictions=predictions,
            commitment_hash=commitment_hash,
            committed_at_utc=committed_at,
            predictions_sealed_before_outcome_reveal=True,
            model_calls_made=model_calls,
            sandbox_calls_made=sandbox_calls,
            provider_latency_ms=latency_ms,
            claim_boundary=_CLAIM_BOUNDARY,
        )

    async def _reveal_locked(self, committed: LiveEvaluationSnapshot) -> LiveEvaluationSnapshot:
        assets, gateway, executor = self._dependencies()
        model_calls = committed.model_calls_made
        sandbox_calls = committed.sandbox_calls_made
        latency_ms = committed.provider_latency_ms

        try:
            criterion_request, bundle_path = _criterion_request(
                assets=assets,
                case_id=self._case_id,
                spec=self._generation_spec(committed.run_id),
            )
            model_calls += 1
            criterion = await gateway.generate(criterion_request)
            latency_ms += criterion.latency_ms
        except Exception:
            return self._failed_from_committed(
                committed,
                stage=LiveFailureStage.CRITERION_GENERATION,
                message="The evaluation model could not produce the later-task answer.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
            )

        try:
            extract_python_submission(criterion.final_response)
        except ValueError:
            return self._failed_from_committed(
                committed,
                stage=LiveFailureStage.CRITERION_EXTRACTION,
                message="The later-task answer did not contain one executable Python block.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
            )

        sandbox_calls += 1
        result = await asyncio.to_thread(
            execute_authored_python_tests,
            executor,
            response=criterion.final_response,
            bundle_path=bundle_path,
        )
        execution = _execution_view(result)
        if execution is None:
            return self._failed_from_committed(
                committed,
                stage=LiveFailureStage.CRITERION_EXECUTION,
                message="The remote later-task check did not complete.",
                model_calls=model_calls,
                sandbox_calls=sandbox_calls,
                latency_ms=latency_ms,
            )

        return _replace_snapshot(
            committed,
            {
                "phase": LiveEvaluationPhase.OUTCOME_REVEALED,
                "outcome": LiveOutcomeView(
                    response=criterion.final_response,
                    execution=execution,
                    revealed_at_utc=self._clock(),
                ),
                "model_calls_made": model_calls,
                "sandbox_calls_made": sandbox_calls,
                "provider_latency_ms": latency_ms,
            },
        )

    def _generation_spec(self, run_id: str) -> GenerationRequestSpec:
        assets, _, _ = self._dependencies()
        return GenerationRequestSpec(
            run_id=run_id,
            sample_id="live-illustration-001",
            system_prompt_version="live-evaluation-system-v1",
            system_prompt=assets.system_prompt,
            system_prompt_sha256=assets.system_prompt_hash,
            model_route=_MODEL_ROUTE,
            sampling=_SAMPLING,
        )

    def _failed_from_committed(
        self,
        committed: LiveEvaluationSnapshot,
        *,
        stage: LiveFailureStage,
        message: str,
        model_calls: int,
        sandbox_calls: int,
        latency_ms: int,
    ) -> LiveEvaluationSnapshot:
        return _replace_snapshot(
            committed,
            {
                "phase": LiveEvaluationPhase.FAILED,
                "failure": LiveFailureView(stage=stage, message=message),
                "model_calls_made": model_calls,
                "sandbox_calls_made": sandbox_calls,
                "provider_latency_ms": latency_ms,
            },
        )

    def _failed(
        self,
        *,
        run_id: str,
        stage: LiveFailureStage,
        message: str,
        model_calls: int,
        sandbox_calls: int,
        latency_ms: int,
        public_response: str | None = None,
        evidence_response: str | None = None,
    ) -> LiveEvaluationSnapshot:
        return LiveEvaluationSnapshot(
            run_id=run_id,
            case_id=self._case_id,
            phase=LiveEvaluationPhase.FAILED,
            evaluation_model=f"{_MODEL_ROUTE.provider}/{_MODEL_ROUTE.model}",
            public_response=public_response,
            evidence_response=evidence_response,
            model_calls_made=model_calls,
            sandbox_calls_made=sandbox_calls,
            provider_latency_ms=latency_ms,
            failure=LiveFailureView(stage=stage, message=message),
            claim_boundary=_CLAIM_BOUNDARY,
        )

    def _require_available(self) -> None:
        if not self.status().available:
            raise LiveEvaluationUnavailableError(self.status().reason or "Unavailable")

    def _dependencies(self) -> tuple[LiveEvaluationAssets, LiveModelGateway, SandboxExecutor]:
        if self._assets is None or self._gateway is None or self._executor is None:
            raise LiveEvaluationUnavailableError("Live illustration dependencies are unavailable")
        return self._assets, self._gateway, self._executor


def create_live_evaluation_service(settings: Settings) -> LiveEvaluationService:
    """Build the safe-default live service used by the FastAPI composition root."""

    case_id = settings.live_evaluation_case_id
    if not settings.live_evaluation_enabled:
        return LiveEvaluationService(
            enabled=False,
            case_id=case_id,
            unavailable_reason="Live illustration is disabled by default.",
        )
    if settings.cerebras_api_key is None:
        return LiveEvaluationService(
            enabled=True,
            case_id=case_id,
            unavailable_reason="The server has no Cerebras API key.",
        )
    try:
        assets = load_live_evaluation_assets(
            settings.live_evaluation_manifest_path,
            settings.live_evaluation_system_prompt_path,
            case_id,
        )
    except (OSError, ValueError):
        return LiveEvaluationService(
            enabled=True,
            case_id=case_id,
            unavailable_reason="The live development case is unavailable.",
        )
    return LiveEvaluationService(
        enabled=True,
        case_id=case_id,
        assets=assets,
        gateway=DirectCerebrasLiveGateway(settings.cerebras_api_key),
        executor=ModalSandboxExecutor(),
    )


def load_live_evaluation_assets(
    manifest_path: Path, prompt_path: Path, case_id: str
) -> LiveEvaluationAssets:
    manifest_path = manifest_path.resolve()
    authored = load_authored_manifest(manifest_path)
    public = project_public_manifest(authored)
    matches = tuple(case for case in public.cases if case.case_id == case_id)
    if len(matches) != 1 or matches[0].split is not BenchmarkSplit.DEVELOPMENT:
        raise ValueError("Live illustration requires one development case")
    prompt_bytes = prompt_path.resolve().read_bytes()
    prompt = prompt_bytes.decode("utf-8").strip()
    if not prompt:
        raise ValueError("Live illustration system prompt is empty")
    return LiveEvaluationAssets(
        manifest_path=manifest_path,
        benchmark_root=manifest_path.parent,
        public_manifest=public,
        system_prompt=prompt,
        system_prompt_hash=file_sha256(prompt.encode("utf-8")),
    )


def _criterion_request(
    *,
    assets: LiveEvaluationAssets,
    case_id: str,
    spec: GenerationRequestSpec,
) -> tuple[StudentGenerationRequest, Path]:
    authored = load_and_verify_manifest(assets.manifest_path)
    verified_public = project_public_manifest(authored)
    if verified_public.projection_hash != assets.public_manifest.projection_hash:
        raise ValueError("Live benchmark changed after predictions were committed")
    case = next(item for item in authored.cases if item.public.case_id == case_id)
    root = assets.benchmark_root.resolve()
    prompt_path = (root / case.criterion.criterion_prompt_ref).resolve()
    bundle_path = (root / case.criterion.test_bundle_ref).resolve()
    if not prompt_path.is_relative_to(root) or not bundle_path.is_relative_to(root):
        raise ValueError("Criterion artifact escapes the benchmark root")
    prompt = prompt_path.read_text(encoding="utf-8")
    payload = CriterionTaskPayload(
        benchmark_version=authored.benchmark_version,
        case_id=case_id,
        criterion_probe_id=case.criterion.criterion_probe_id,
        criterion_probe=prompt,
    )
    return create_student_generation_request(spec=spec, payload=payload), bundle_path


def _assess_public_response(
    case_id: str, response: str
) -> tuple[LiveDialogueAssessmentView, Evidence]:
    text = " ".join(response.casefold().split())
    if case_id == "dev-aliasing-001":
        correct = any(
            marker in text
            for marker in ("same list", "same object", "same reference", "refer to the same")
        )
        incorrect = any(
            marker in text
            for marker in ("independent copy", "separate copy", "separate list", "does not affect")
        ) and not any(marker in text for marker in ("not a copy", "does not create a copy"))
    elif case_id == "dev-none-falsy-001":
        correct = "is none" in text and any(marker in text for marker in ("zero", "0"))
        incorrect = any(
            marker in text
            for marker in ("zero means missing", "0 means missing", "zero is missing", "not score")
        )
    else:
        correct = False
        incorrect = False

    if correct and incorrect:
        category = EvidenceCategory.CONFLICTING
        rationale = "Phrases linked to both the expected answer and the misconception matched."
    elif correct:
        category = EvidenceCategory.CORRECT
        rationale = "A phrase linked to the expected answer matched; meaning was not verified."
    elif incorrect:
        category = EvidenceCategory.MISCONCEPTION
        rationale = "A phrase associated with the misconception matched; meaning was not verified."
    else:
        category = EvidenceCategory.UNCERTAIN
        rationale = "No recognised phrase matched; this rule leaves the answer unclassified."
    view = LiveDialogueAssessmentView(category=category, rationale=rationale)
    evidence = Evidence(
        category=category,
        confidence=(
            1.0 if category in {EvidenceCategory.CORRECT, EvidenceCategory.MISCONCEPTION} else 0.5
        ),
        rationale=rationale,
        observed_signals=("live_demo:transparent_development_rule",),
    )
    return view, evidence


def _predictions(
    *,
    assets: LiveEvaluationAssets,
    case_id: str,
    run_id: str,
    public_evidence: Evidence,
    passed: int,
    failed: int,
    committed_at: datetime,
) -> tuple[LivePredictionView, LivePredictionView]:
    initial = load_initial_tracker_state(
        assets.benchmark_root,
        assets.public_manifest,
        case_id=case_id,
    )
    dialogue = update_tracker(initial, public_evidence)
    probe_evidence = evidence_from_test_counts(
        passed=passed,
        failed=failed,
        source="live evidence probe",
    )
    informed = update_tracker(dialogue, probe_evidence)
    return (
        _prediction(
            run_id=run_id,
            condition="dialogue_only",
            label="Visible answer only",
            score=dialogue.mastery_probability,
            committed_at=committed_at,
        ),
        _prediction(
            run_id=run_id,
            condition="probe_informed",
            label="Visible answer and coding check",
            score=informed.mastery_probability,
            committed_at=committed_at,
        ),
    )


def _prediction(
    *,
    run_id: str,
    condition: Literal["dialogue_only", "probe_informed"],
    label: str,
    score: float,
    committed_at: datetime,
) -> LivePredictionView:
    content = {
        "condition": condition,
        "label": label,
        "tracker_score": score,
        "policy_threshold": _POLICY_THRESHOLD,
        "predicts_success": score >= _POLICY_THRESHOLD,
        "committed_at_utc": committed_at,
    }
    return LivePredictionView.model_validate(
        {
            **content,
            "record_hash": canonical_sha256({"run_id": run_id, **content}),
        }
    )


def _execution_view(result: SandboxTestResult) -> LiveExecutionView | None:
    parsed = result
    if (
        parsed.execution.status != "completed"
        or parsed.execution.exit_code != 0
        or parsed.outcome is None
        or parsed.outcome.status != "completed"
    ):
        return None
    return LiveExecutionView(
        passed_checks=parsed.outcome.passed,
        failed_checks=parsed.outcome.failed,
        passed_all_checks=parsed.outcome.failed == 0,
        sandbox_id=parsed.execution.sandbox_id,
    )


def _run_id(idempotency_key: str, case_id: str) -> str:
    return hashlib.sha256(f"live-evaluation-v1:{case_id}:{idempotency_key}".encode()).hexdigest()


def _replace_snapshot(
    snapshot: LiveEvaluationSnapshot,
    updates: dict[str, object],
) -> LiveEvaluationSnapshot:
    return LiveEvaluationSnapshot.model_validate({**snapshot.model_dump(mode="python"), **updates})
