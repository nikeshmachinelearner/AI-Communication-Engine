from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=10_000,
        examples=[
            (
                "Can you send the updated project report "
                "by tomorrow at 10 AM?"
            )
        ],
    )

    reference_time: str | None = Field(
        default=None,
        description=(
            "ISO 8601 time used to resolve relative dates."
        ),
        examples=["2026-11-17T14:00:00+05:45"],
    )

    timezone: str = Field(
        default="Asia/Kathmandu",
        examples=["Asia/Kathmandu"],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    conversation_id: str | None = None

    sender_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    conversation_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    hours_until_deadline: float | None = Field(
        default=None,
        ge=-8760,
        le=87600,
    )


class ModelStatus(BaseModel):
    classification: bool
    retrieval: bool
    extraction: bool
    ranking: bool


class HealthResponse(BaseModel):
    status: str
    models: ModelStatus


class AnalyzeResponse(BaseModel):
    request_id: str
    text: str
    classification: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    ranking: dict[str, Any] | None = None
    recommended_actions: list[str]
    component_errors: dict[str, str]
    total_latency_ms: float