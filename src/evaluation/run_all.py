import json
import subprocess
import sys
from pathlib import Path
from typing import Any


RESULT_PATHS = {
    "classification": Path(
        "experiments/baseline_metrics.json"
    ),
    "retrieval": Path(
        "experiments/retrieval_metrics.json"
    ),
    "ranking": Path(
        "experiments/ranking_training_metrics.json"
    ),
    "latency": Path(
        "experiments/latency_metrics.json"
    ),
}

SUMMARY_PATH = Path(
    "experiments/system_evaluation.json"
)


def run_command(module_name: str) -> None:
    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    print(f"Running: {' '.join(command)}")

    subprocess.run(
        command,
        check=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
        }

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    evaluation_modules = [
        "src.retrieval.evaluate",
        "src.ranking.train",
        "src.evaluation.latency",
    ]

    execution_errors = {}

    for module_name in evaluation_modules:
        try:
            run_command(module_name)
        except subprocess.CalledProcessError as error:
            execution_errors[module_name] = str(error)

    results = {
        component: load_json(path)
        for component, path in RESULT_PATHS.items()
    }

    summary = {
        "results": results,
        "execution_errors": execution_errors,
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(f"\nSaved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()