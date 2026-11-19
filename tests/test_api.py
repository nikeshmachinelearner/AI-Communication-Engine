from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.main import app


class FakeService:
    def status(self) -> dict:
        return {
            "classification": True,
            "retrieval": True,
            "extraction": True,
            "ranking": True,
        }

    def analyze(self, **kwargs) -> dict:
        return {
            "text": kwargs["text"],
            "classification": {
                "predicted_labels": ["task"],
                "scores": {
                    "task": 0.9,
                },
                "threshold": 0.5,
            },
            "extraction": {
                "people": [],
                "dates": [],
                "tasks": [],
                "commitments": [],
            },
            "context": {
                "query": kwargs["text"],
                "results": [],
            },
            "ranking": {
                "model": "test_ranker",
                "ranked_messages": [],
            },
            "recommended_actions": [
                "Complete the requested task."
            ],
            "component_errors": {},
            "total_latency_ms": 10.0,
        }


def test_root() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


def test_health() -> None:
    with TestClient(app) as client:
        app.state.service = FakeService()
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze() -> None:
    with TestClient(app) as client:
        app.state.service = FakeService()

        response = client.post(
            "/analyze",
            json={
                "text": "Please send the report tomorrow.",
                "top_k": 3,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["text"] == (
        "Please send the report tomorrow."
    )
    assert "request_id" in body
    assert "classification" in body
    assert "recommended_actions" in body


def test_empty_text_is_rejected() -> None:
    with TestClient(app) as client:
        app.state.service = FakeService()

        response = client.post(
            "/analyze",
            json={
                "text": "",
            },
        )

    assert response.status_code == 422


def test_invalid_top_k_is_rejected() -> None:
    with TestClient(app) as client:
        app.state.service = FakeService()

        response = client.post(
            "/analyze",
            json={
                "text": "Find related messages.",
                "top_k": 0,
            },
        )

    assert response.status_code == 422


def test_metrics_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]