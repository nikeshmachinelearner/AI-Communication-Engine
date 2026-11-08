import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed/messages.csv")
MODEL_PATH = Path("models/classifier/tfidf_logistic.joblib")
METRICS_PATH = Path("experiments/baseline_metrics.json")

LABEL_COLUMNS = [
    "question",
    "task",
    "meeting_request",
    "important_information",
    "general_conversation",
    "urgent",
]

RANDOM_STATE = 42


def load_data() -> tuple[pd.Series, pd.DataFrame]:
    dataframe = pd.read_csv(DATA_PATH)

    required_columns = {"text", *LABEL_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    return dataframe["text"], dataframe[LABEL_COLUMNS]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
        ]
    )


def main() -> None:
    texts, labels = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=LABEL_COLUMNS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "tfidf_logistic_regression",
        "training_examples": len(x_train),
        "test_examples": len(x_test),
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "classification_report": report,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "pipeline": pipeline,
        "labels": LABEL_COLUMNS,
        "threshold": 0.5,
    }

    joblib.dump(model_artifact, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()