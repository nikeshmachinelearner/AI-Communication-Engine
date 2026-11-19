from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from src.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.pipeline.service import (
    CommunicationIntelligenceService,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = CommunicationIntelligenceService()
    yield


app = FastAPI(
    title="AI-Native Communication Intelligence Engine",
    description=(
        "Classifies messages, retrieves historical context, "
        "extracts actionable information, and scores priority."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(
    request: Request,
    call_next,
):
    start_time = perf_counter()

    try:
        response = await call_next(request)

        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status=str(response.status_code),
        ).inc()

        return response

    except Exception:
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status="500",
        ).inc()

        raise

    finally:
        elapsed_seconds = perf_counter() - start_time

        REQUEST_LATENCY.labels(
            endpoint=request.url.path
        ).observe(elapsed_seconds)


@app.get("/")
def root() -> dict:
    return {
        "name": "AI-Native Communication Intelligence Engine",
        "version": "0.1.0",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request) -> dict:
    service = request.app.state.service
    model_status = service.status()

    all_available = all(model_status.values())
    any_available = any(model_status.values())

    if all_available:
        status = "healthy"
    elif any_available:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "models": model_status,
    }


@app.get("/ready")
def readiness(request: Request) -> dict:
    service = request.app.state.service
    model_status = service.status()

    required_components = [
        "classification",
        "extraction",
    ]

    ready = all(
        model_status[component]
        for component in required_components
    )

    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "models": model_status,
            },
        )

    return {
        "status": "ready",
        "models": model_status,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
def analyze(
    payload: AnalyzeRequest,
    request: Request,
) -> dict:
    service = request.app.state.service

    try:
        reference_time = (
            datetime.fromisoformat(payload.reference_time)
            if payload.reference_time
            else None
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "reference_time must be a valid ISO 8601 value."
            ),
        ) from error

    try:
        result = service.analyze(
            text=payload.text,
            reference_time=reference_time,
            timezone_name=payload.timezone,
            top_k=payload.top_k,
            conversation_id=payload.conversation_id,
            sender_importance=payload.sender_importance,
            conversation_relevance=(
                payload.conversation_relevance
            ),
            hours_until_deadline=(
                payload.hours_until_deadline
            ),
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Message analysis failed.",
        ) from error

    return {
        "request_id": str(uuid4()),
        **result,
    }