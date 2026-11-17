
import math
import re
from datetime import datetime


FEATURE_COLUMNS = [
    "message_age_hours",
    "has_question",
    "has_task",
    "has_meeting",
    "has_deadline",
    "is_urgent",
    "is_important",
    "has_commitment",
    "sender_importance",
    "conversation_relevance",
    "deadline_proximity",
    "message_length",
]


QUESTION_PATTERN = re.compile(
    r"\?|"
    r"\b(?:what|when|where|who|why|how|can you|could you|"
    r"would you|are you|do you|did you)\b",
    flags=re.IGNORECASE,
)

TASK_PATTERN = re.compile(
    r"\b(?:please|kindly|send|submit|review|update|prepare|"
    r"complete|finish|investigate|fix|call|contact|schedule|"
    r"remember to|need you to|we need to)\b",
    flags=re.IGNORECASE,
)

MEETING_PATTERN = re.compile(
    r"\b(?:meeting|meet|schedule a call|book a call|"
    r"appointment|discussion|conference|video call)\b",
    flags=re.IGNORECASE,
)

URGENT_PATTERN = re.compile(
    r"\b(?:urgent|immediately|asap|critical|emergency|"
    r"right now|high priority|production is down|"
    r"server is down|system is down|blocked)\b",
    flags=re.IGNORECASE,
)

IMPORTANT_PATTERN = re.compile(
    r"\b(?:important|deadline|approved|rejected|postponed|"
    r"cancelled|security|breach|outage|client|customer|"
    r"production|release)\b",
    flags=re.IGNORECASE,
)

DEADLINE_PATTERN = re.compile(
    r"\b(?:today|tomorrow|tonight|this morning|"
    r"this afternoon|this evening|before\s+\w+|"
    r"by\s+\w+|within\s+\d+\s+(?:minutes?|hours?|days?)|"
    r"\d{1,2}(?::\d{2})?\s*(?:AM|PM))\b",
    flags=re.IGNORECASE,
)

COMMITMENT_PATTERN = re.compile(
    r"\b(?:I will|I\'?[Ll]|we will|we\'ll|I can|we can|"
    r"I promise|I commit)\b",
    flags=re.IGNORECASE,
)


def regex_flag(pattern: re.Pattern, text: str) -> int:
    return int(pattern.search(text) is not None)


def calculate_deadline_proximity(
    hours_until_deadline: float | None,
) -> float:
    """
    Converts time until deadline into a 0-1 urgency feature.

    Overdue or immediate deadlines receive 1.0.
    Deadlines farther away receive gradually smaller values.
    """
    if hours_until_deadline is None:
        return 0.0
    if hours_until_deadline <= 0:
        return 1.0

    return 1.0 / (1.0 + math.log1p(hours_until_deadline))


def extract_features(
    message: dict,
    reference_time: datetime | None = None,
) -> dict:
    text = str(message["text"]).strip()

    if not text:
        raise ValueError("Message text cannot be empty.")

    if reference_time is None:
        reference_time = datetime.now().astimezone()

    message_time_value = message.get("timestamp")

    if message_time_value:
        message_time = datetime.fromisoformat(message_time_value)

        if message_time.tzinfo is None:
            message_time = message_time.replace(
                tzinfo=reference_time.tzinfo
            )

        age = reference_time - message_time
        message_age_hours = max(
            age.total_seconds() / 3600,
            0.0
        )
    else:
        message_age_hours = 0.0

    hours_until_deadline = message.get("hours_until_deadline")

    if hours_until_deadline is not None:
        hours_until_deadline = float(hours_until_deadline)

    features = {
        "message_age_hours": float(message_age_hours),
        "has_question": regex_flag(QUESTION_PATTERN, text),
        "has_task": regex_flag(TASK_PATTERN, text),
        "has_meeting": regex_flag(MEETING_PATTERN, text),
        "has_deadline": regex_flag(DEADLINE_PATTERN, text),
        "is_urgent": regex_flag(URGENT_PATTERN, text),
        "is_important": regex_flag(IMPORTANT_PATTERN, text),
        "has_commitment": regex_flag(COMMITMENT_PATTERN, text),
        "sender_importance": float(
            message.get("sender_importance", 0.5)
        ),
        "conversation_relevance": float(
            message.get("conversation_relevance", 0.5)
        ),
        "deadline_proximity": calculate_deadline_proximity(
            hours_until_deadline
        ),
        "message_length": float(len(text.split())),
    }

    return features

