import json
from pathlib import Path

import mlflow


EXPERIMENT_NAME = (
    "communication-intelligence-system-evaluation"
)

SUMMARY_PATH = Path(
    "experiments/system_evaluation.json"
)


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} does not exist. Run: "
            "python -m src.evaluation.run_all"
        )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_nested(
    data: dict,
    path: list[str],
    default: float = 0.0,
) -> float:
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def main() -> None:
    summary = load_summary()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="full-system-baseline"
    ):
        mlflow.log_param(
            "classification_model",
            "tfidf_logistic_regression",
        )

        mlflow.log_param(
            "retrieval_model",
            "all-MiniLM-L6-v2",
        )

        mlflow.log_param(
            "ranking_model",
            "lightgbm_lambdarank",
        )

        mlflow.log_param(
            "extraction_model",
            "spacy_rules_dateparser",
        )

        results = summary.get("results", {})

        metrics = {
            "classification_macro_f1": get_nested(
                results,
                [
                    "classification",
                    "macro_f1",
                ],
            ),
            "classification_micro_f1": get_nested(
                results,
                [
                    "classification",
                    "micro_f1",
                ],
            ),
            "retrieval_recall_at_5": get_nested(
                results,
                [
                    "retrieval",
                    "recall_at_5",
                ],
            ),
            "retrieval_mrr": get_nested(
                results,
                [
                    "retrieval",
                    "mrr",
                ],
            ),
            "ranking_ndcg_at_5": get_nested(
                results,
                [
                    "ranking",
                    "ndcg_at_5",
                ],
            ),
            "system_p95_latency_ms": get_nested(
                results,
                [
                    "latency",
                    "p95_ms",
                ],
            ),
        }

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(SUMMARY_PATH),
            artifact_path="evaluation",
        )

        for path in Path("experiments").glob("*.json"):
            if path != SUMMARY_PATH:
                mlflow.log_artifact(
                    str(path),
                    artifact_path="component_metrics",
                )

        mlflow.set_tag(
            "project",
            "AI-Native Communication Intelligence Engine",
        )

        mlflow.set_tag(
            "data_status",
            "synthetic_prototype",
        )

        mlflow.set_tag(
            "hardware",
            "MacBook Air M2",
        )

        print("MLflow run completed.")


if __name__ == "__main__":
    main()