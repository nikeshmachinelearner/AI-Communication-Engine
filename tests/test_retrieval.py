import pytest

from src.retrieval.evaluate import recall_at_k, reciprocal_rank
from src.retrieval.search import ContextRetriever


@pytest.fixture(scope="module")
def retriever() -> ContextRetriever:
    return ContextRetriever()


def test_search_returns_expected_structure(
    retriever: ContextRetriever,
) -> None:
    response = retriever.search(
        query="When is the project meeting?",
        top_k=3,
    )

    assert response["query"] == "When is the project meeting?"
    assert response["top_k"] == 3
    assert "latency_ms" in response
    assert "results" in response
    assert len(response["results"]) <= 3


def test_result_contains_metadata(
    retriever: ContextRetriever,
) -> None:
    response = retriever.search(
        query="What happened to the application release?",
        top_k=1,
    )

    result = response["results"][0]

    required_fields = {
        "rank",
        "similarity_score",
        "message_id",
        "conversation_id",
        "sender",
        "timestamp",
        "text",
    }

    assert required_fields.issubset(result)


def test_results_are_sorted_by_score(
    retriever: ContextRetriever,
) -> None:
    response = retriever.search(
        query="What are the password rules?",
        top_k=5,
    )

    scores = [
        result["similarity_score"]
        for result in response["results"]
    ]

    assert scores == sorted(scores, reverse=True)


def test_conversation_filter(
    retriever: ContextRetriever,
) -> None:
    response = retriever.search(
        query="What happened?",
        top_k=5,
        conversation_id="release",
    )

    assert response["results"]

    assert all(
        result["conversation_id"] == "release"
        for result in response["results"]
    )


def test_empty_query_is_rejected(
    retriever: ContextRetriever,
) -> None:
    with pytest.raises(ValueError):
        retriever.search(query="   ")


def test_recall_at_k() -> None:
    retrieved = ["msg_001", "msg_009", "msg_003"]
    relevant = ["msg_001", "msg_002", "msg_003"]

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(
        2 / 3
    )


def test_reciprocal_rank() -> None:
    retrieved = ["msg_009", "msg_005", "msg_003"]
    relevant = ["msg_003"]

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(
        1 / 3
    )