from datetime import datetime
from zoneinfo import ZoneInfo

from dateparser.search import search_dates

from src.extraction.schemas import ExtractedDate


DEFAULT_TIMEZONE = "Asia/Kathmandu"


def extract_dates(
    text: str,
    reference_time: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> list[ExtractedDate]:
    timezone = ZoneInfo(timezone_name)

    if reference_time is None:
        reference_time = datetime.now(timezone)
    elif reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone)
    else:
        reference_time = reference_time.astimezone(timezone)

    settings = {
        "RELATIVE_BASE": reference_time,
        "TIMEZONE": timezone_name,
        "TO_TIMEZONE": timezone_name,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
    }

    matches = search_dates(
        text,
        languages=["en"],
        settings=settings,
    )

    if not matches:
        return []

    extracted_dates = []
    seen = set()

    for matched_text, parsed_datetime in matches:
        cleaned_text = matched_text.strip()

        if not cleaned_text:
            continue

        key = (
            cleaned_text.lower(),
            parsed_datetime.isoformat(),
        )

        if key in seen:
            continue

        start = text.lower().find(cleaned_text.lower())
        end = start + len(cleaned_text) if start >= 0 else None

        extracted_dates.append(
            ExtractedDate(
                text=cleaned_text,
                normalized=parsed_datetime.isoformat(),
                start=start if start >= 0 else None,
                end=end,
            )
        )

        seen.add(key)

    return extracted_dates