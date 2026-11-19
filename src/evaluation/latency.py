import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from src.pipeline.service import (
    CommunicationIntelligenceService,
)


OUTPUT_PATH = Path(
    "experiments/latency_metrics.json"
)

TEST_MESSAGES = [
    "Please send the report tomorrow.",
    "Are we still meeting this afternoon?",
    "Urgent: the production server is down.",
    "The client approved the final design.",
    "Thanks for your help today.",
    "I will review the budget before Friday.",
    "Can you schedule a meeting with Maya?",
    "The application release moved to Monday.",
]


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    return float(
        np.percentile(values, percentile_value)
    )


def main() -> None:
    service = CommunicationIntelligenceService()

    # Warm-up calls prevent initialization overhead from dominating
    # the measured requests.
    for text in TEST_MESSAGES[:3]:
        service.analyze(
            text=text,
            top_k=3,
        )

    measurements_ms = []

    for _ in range(10):
        for text in TEST_MESSAGES:
            start_time = perf_counter()

            service.analyze(
                text=text,
                top_k=3,
            )

            elapsed_ms = (
                perf_counter() - start_time
            ) * 1000

            measurements_ms.append(elapsed_ms)

    metrics = {
        "request_count": len(measurements_ms),
        "mean_ms": round(
            mean(measurements_ms),
            2,
        ),
        "median_ms": round(
            median(measurements_ms),
            2,
        ),
        "p90_ms": round(
            percentile(measurements_ms, 90),
            2,
        ),
        "p95_ms": round(
            percentile(measurements_ms, 95),
            2,
        ),
        "p99_ms": round(
            percentile(measurements_ms, 99),
            2,
        ),
        "minimum_ms": round(
            min(measurements_ms),
            2,
        ),
        "maximum_ms": round(
            max(measurements_ms),
            2,
        ),
        "target_p95_ms": 300,
    }

    metrics["p95_target_met"] = (
        metrics["p95_ms"]
        < metrics["target_p95_ms"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()