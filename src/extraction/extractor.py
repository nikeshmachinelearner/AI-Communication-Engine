import argparse
import json
from datetime import datetime
from time import perf_counter

from src.extraction.action_extractor import (
    extract_commitments,
    extract_tasks,
)
from src.extraction.date_extractor import extract_dates
from src.extraction.entity_extractor import EntityExtractor
from src.extraction.schemas import ExtractionResult


class InformationExtractor:
    def __init__(self) -> None:
        self.entity_extractor = EntityExtractor()

    def extract(
        self,
        text: str,
        reference_time: datetime | None = None,
        timezone_name: str = "Asia/Kathmandu",
    ) -> dict:
        text = text.strip()

        if not text:
            raise ValueError("Text cannot be empty.")

        start_time = perf_counter()

        entity_result = self.entity_extractor.extract(text)
        doc = entity_result["doc"]

        dates = extract_dates(
            text=text,
            reference_time=reference_time,
            timezone_name=timezone_name,
        )

        tasks = extract_tasks(doc)
        commitments = extract_commitments(doc)

        normalized_deadline = (
            dates[0].normalized if dates else None
        )

        for task in tasks:
            task.deadline = normalized_deadline

        result = ExtractionResult(
            text=text,
            people=entity_result["people"],
            organizations=entity_result["organizations"],
            locations=entity_result["locations"],
            dates=dates,
            tasks=tasks,
            commitments=commitments,
            entities=entity_result["entities"],
        )

        output = result.model_dump()
        output["latency_ms"] = round(
            (perf_counter() - start_time) * 1000,
            2,
        )

        return output


def parse_reference_time(
    value: str | None,
) -> datetime | None:
    if value is None:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "reference-time must use ISO 8601 format, for example "
            "2026-11-12T14:00:00+05:45"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract tasks, dates, people, locations, "
            "and commitments from a message."
        )
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Message to analyze.",
    )

    parser.add_argument(
        "--reference-time",
        default=None,
        help=(
            "ISO 8601 message timestamp used to resolve relative dates."
        ),
    )

    parser.add_argument(
        "--timezone",
        default="Asia/Kathmandu",
        help="IANA timezone name.",
    )

    arguments = parser.parse_args()

    extractor = InformationExtractor()

    result = extractor.extract(
        text=arguments.text,
        reference_time=parse_reference_time(
            arguments.reference_time
        ),
        timezone_name=arguments.timezone,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()