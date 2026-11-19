from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.pipeline.service import (
    CommunicationIntelligenceService,
)


@pytest.fixture(scope="module")
def service() -> CommunicationIntelligenceService:
    return CommunicationIntelligenceService()


def test_service_reports_component_status(
    service: CommunicationIntelligenceService,
) -> None:
    status = service.status()

    assert set(status) == {
        "classification",
        "retrieval",
        "extraction",
        "ranking",
    }


def test_pipeline_returns_expected_structure(
    service: CommunicationIntelligenceService,
) -> None:
    result = service.analyze(
        text="Please send the report tomorrow.",
        reference_time=datetime(
            2026,
            11,
            17,
            14,
            0,
            tzinfo=ZoneInfo("Asia/Kathmandu"),
        ),
        top_k=3,
    )

    required_fields = {
        "text",
        "classification",
        "extraction",
        "context",
        "ranking",
        "recommended_actions",
        "component_errors",
        "total_latency_ms",
    }

    assert required_fields.issubset(result)
    assert result["total_latency_ms"] >= 0


def test_pipeline_handles_general_message(
    service: CommunicationIntelligenceService,
) -> None:
    result = service.analyze(
        text="Thanks for your help today.",
        top_k=2,
    )

    assert result["text"] == "Thanks for your help today."