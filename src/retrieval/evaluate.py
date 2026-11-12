import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from src.retrieval.search import ContextRetriever


OUTPUT_PATH = Path("experiments/retrieval_metrics.json")

EVALUATION_QUERIES = [
    {
        "query_id": "query_001",
        "query": "When is the Project Alpha meeting?",
        "relevant_message_ids": [
            "msg_001",
            "msg_002",
            "msg_003",
        ],
    },
    {
        "query_id": "query_002",
        "query": "Where is the project review taking place?",
        "relevant_message_ids": [
            "msg_003",
        ],
    },
    {
        "query_id": "query_003",
        "query": "What changed with the marketing finances?",
        "relevant_message_ids": [
            "msg_006",
            "msg_007",
            "msg_008",
        ],
    },
    {
        "query_id": "query_004",
        "query": "Was the customer's login issue resolved?",
        "relevant_message_ids": [
            "msg_009",
            "msg_010",
            "msg_011",
        ],
    },
    {
        "query_id": "query_005",
        "query": "Why was the mobile app launch delayed?",
        "relevant_message_ids": [
            "msg_012",
            "msg_013",
        ],
    },
    {
        "query_id": "query_006",
        "query": "What is the new account password requirement?",
        "relevant_message_ids": [
            "msg_018",
            "msg_019",
            "msg_020",
        ],
    },
]


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    relevant_retrieved = retrieved_at_k.intersection(relevant_set)

    return len(relevant_retrieved) / len(relevant_set)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    relevant_set = set(relevant_ids)

    for rank, message_id in enumerate(retrieved_ids, start=1):
        if message_id in relevant_set:
            return 1.0 / rank

    return 0.0


def main() -> None:
    retriever = ContextRetriever()

    recall_at_1_scores = []
    recall_at_3_scores = []
    recall_at_5_scores = []
    reciprocal_ranks = []
    latencies_ms = []
    per_query_results = []

    evaluation_start = perf_counter()

    for item in EVALUATION_QUERIES:
        response = retriever.search(
            query=item["query"],
            top_k=5,
        )

        retrieved_ids = [
            result["message_id"]
            for result in response["results"]
        ]

        query_recall_at_1 = recall_at_k(
            retrieved_ids,
            item["relevant_message_ids"],
            1,
        )

        query_recall_at_3 = recall_at_k(
            retrieved_ids,
            item["relevant_message_ids"],
            3,
        )

        query_recall_at_5 = recall_at_k(
            retrieved_ids,
            item["relevant_message_ids"],
            5,
        )

        query_reciprocal_rank = reciprocal_rank(
            retrieved_ids,
            item["relevant_message_ids"],
        )

        recall_at_1_scores.append(query_recall_at_1)
        recall_at_3_scores.append(query_recall_at_3)
        recall_at_5_scores.append(query_recall_at_5)
        reciprocal_ranks.append(query_reciprocal_rank)
        latencies_ms.append(response["latency_ms"])

        per_query_results.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "relevant_message_ids": item[
                    "relevant_message_ids"
                ],
                "retrieved_message_ids": retrieved_ids,
                "recall_at_1": round(query_recall_at_1, 4),
                "recall_at_3": round(query_recall_at_3, 4),
                "recall_at_5": round(query_recall_at_5, 4),
                "reciprocal_rank": round(
                    query_reciprocal_rank,
                    4,
                ),
                "latency_ms": response["latency_ms"],
            }
        )

    total_time_ms = (perf_counter() - evaluation_start) * 1000

    metrics = {
        "model_name": retriever.model_name,
        "number_of_queries": len(EVALUATION_QUERIES),
        "recall_at_1": round(mean(recall_at_1_scores), 4),
        "recall_at_3": round(mean(recall_at_3_scores), 4),
        "recall_at_5": round(mean(recall_at_5_scores), 4),
        "mrr": round(mean(reciprocal_ranks), 4),
        "mean_latency_ms": round(mean(latencies_ms), 2),
        "total_evaluation_time_ms": round(total_time_ms, 2),
        "per_query_results": per_query_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"Recall@3: {metrics['recall_at_3']:.4f}")
    print(f"Recall@5: {metrics['recall_at_5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(
        "Mean search latency: "
        f"{metrics['mean_latency_ms']:.2f} ms"
    )
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()