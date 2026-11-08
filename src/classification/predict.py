import argparse
import json
from pathlib import Path

import joblib


MODEL_PATH = Path("models/classifier/tfidf_logistic.joblib")


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. "
            "Run: python -m src.classification.train_baseline"
        )

    return joblib.load(MODEL_PATH)


def predict_message(text: str, threshold: float | None = None) -> dict:
    artifact = load_model()

    pipeline = artifact["pipeline"]
    labels = artifact["labels"]

    if threshold is None:
        threshold = artifact["threshold"]

    probabilities = pipeline.predict_proba([text])[0]

    scores = {
        label: round(float(probability), 4)
        for label, probability in zip(labels, probabilities)
    }

    predicted_labels = [
        label
        for label, probability in scores.items()
        if probability >= threshold
    ]

    if not predicted_labels:
        predicted_labels = [max(scores, key=scores.get)]

    return {
        "text": text,
        "predicted_labels": predicted_labels,
        "scores": scores,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify an incoming communication message."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Message that should be classified.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional multi-label probability threshold.",
    )

    arguments = parser.parse_args()

    result = predict_message(
        text=arguments.text,
        threshold=arguments.threshold,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()