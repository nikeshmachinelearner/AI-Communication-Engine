from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.extraction.date_extractor import extract_dates
from src.extraction.extractor import InformationExtractor


REFERENCE_TIME = datetime(
    2026,
    11,
    12,
    14,
    0,
    tzinfo=ZoneInfo("Asia/Kathmandu"),
)


@pytest.fixture(scope="module")
def extractor() -> InformationExtractor:
    return InformationExtractor()


def test_empty_text_is_rejected(
    extractor: InformationExtractor,
) -> None:
    with pytest.raises(ValueError):
        extractor.extract("   ")


def test_extracts_task(
    extractor: InformationExtractor,
) -> None:
    result = extractor.extract(
        "Please send the revised project report.",
        reference_time=REFERENCE_TIME,
    )

    assert result["tasks"]
    assert result["tasks"][0]["action"] == "send"


def test_extracts_commitment(
    extractor: InformationExtractor,
) -> None:
    result = extractor.extract(
        "I will review the budget tomorrow.",
        reference_time=REFERENCE_TIME,
    )

    assert result["commitments"]
    assert result["commitments"][0]["owner"] == "I"


def test_extracts_business_location(
    extractor: InformationExtractor,
) -> None:
    result = extractor.extract(
        "Please meet Maya in conference room B.",
        reference_time=REFERENCE_TIME,
    )

    normalized_locations = {
        location.lower()
        for location in result["locations"]
    }

    assert "conference room b" in normalized_locations


def test_extracts_person(
    extractor: InformationExtractor,
) -> None:
    result = extractor.extract(
        "Please send the updated report to Maya Sharma before the review.",
        reference_time=REFERENCE_TIME,
    )

    normalized_people = {
        person.lower()
        for person in result["people"]
    }

    assert any("maya" in person for person in normalized_people)


def test_relative_date_uses_reference_time() -> None:
    results = extract_dates(
        "Send it tomorrow at 10 AM.",
        reference_time=REFERENCE_TIME,
    )

    assert results

    normalized_values = [
        result.normalized
        for result in results
        if result.normalized is not None
    ]

    assert any(
        value.startswith("2026-11-13")
        for value in normalized_values
    )


def test_output_contains_required_fields(
    extractor: InformationExtractor,
) -> None:
    result = extractor.extract(
        "Can you send the report tomorrow?",
        reference_time=REFERENCE_TIME,
    )

    required_fields = {
        "text",
        "people",
        "organizations",
        "locations",
        "dates",
        "tasks",
        "commitments",
        "entities",
        "latency_ms",
    }

    assert required_fields.issubset(result)