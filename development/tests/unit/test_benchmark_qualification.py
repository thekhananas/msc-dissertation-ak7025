"""Development-only direct-provider qualification tests."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from socratic_tutor.benchmark.cerebras import GatewayGenerationResult
from socratic_tutor.benchmark.cli import main
from socratic_tutor.benchmark.generation import StudentGenerationRequest
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.qualification import (
    CerebrasQualificationSecrets,
    ProviderQualificationPlan,
    load_provider_qualification_plan,
    run_provider_qualification,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    create_recorded_response,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "manifest.yaml"
PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "provider-qualification-cerebras.yaml"


class FakeGateway:
    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult:
        response = create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response="I would inspect the aliases before changing the list.",
            provider_metadata=ProviderResponseMetadata(
                provider_id="cerebras",
                model_id="gpt-oss-120b",
                resolved_provider_id="cerebras",
                resolved_model_id="gpt-oss-120b",
                backend_fingerprint="fp-development-qualification",
                input_tokens=12,
                output_tokens=10,
                latency_ms=8,
            ),
            captured_at_utc=datetime(2026, 8, 23, 9, 0, tzinfo=UTC),
        )
        content = {
            "response": response,
            "attempt_count": 1,
            "retried_status_codes": (),
        }
        draft = GatewayGenerationResult.model_construct(
            _fields_set=set(content), **content, invocation_hash="0" * 64
        )
        return GatewayGenerationResult.model_validate(
            {
                **content,
                "invocation_hash": model_content_hash(draft, exclude={"invocation_hash"}),
            }
        )


def test_qualification_generates_only_development_public_records(tmp_path: Path) -> None:
    plan = load_provider_qualification_plan(PLAN)

    report = asyncio.run(
        run_provider_qualification(
            plan=plan,
            manifest_path=MANIFEST,
            output_root=tmp_path,
            run_id="development-qualification-test",
            gateway=FakeGateway(),
            clock=lambda: datetime(2026, 8, 23, 9, 0, tzinfo=UTC),
        )
    )

    assert report.gate_passed is True
    assert report.success_count == 2
    assert {result.case_id for result in report.results} == {
        "dev-aliasing-001",
        "dev-none-falsy-001",
    }
    assert all(result.backend_fingerprint for result in report.results)
    records = (tmp_path / "recorded_responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    assert all(json.loads(record)["request"]["channel"] == "public" for record in records)


def test_qualification_cli_refuses_to_run_without_explicit_local_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOCRATIC_CEREBRAS_API_KEY", raising=False)

    assert (
        main(
            [
                "provider-qualify",
                "--plan",
                str(PLAN),
                "--manifest",
                str(MANIFEST),
                "--output-root",
                str(tmp_path),
                "--run-id",
                "development-qualification-test",
            ]
        )
        == 1
    )
    assert "SOCRATIC_CEREBRAS_API_KEY is required" in capsys.readouterr().err


def test_qualification_secret_loads_from_ignored_local_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SOCRATIC_CEREBRAS_API_KEY=local-test-key\n", encoding="utf-8")

    secrets = CerebrasQualificationSecrets()

    assert secrets.cerebras_api_key is not None
    assert secrets.cerebras_api_key.get_secret_value() == "local-test-key"


def test_qualification_plan_rejects_non_cerebras_route() -> None:
    plan = load_provider_qualification_plan(PLAN)
    raw = plan.model_dump(mode="python")
    raw["model_route"] = {"provider": "other", "model": "other-model"}

    with pytest.raises(ValueError, match="direct cerebras"):
        ProviderQualificationPlan.model_validate(raw)
