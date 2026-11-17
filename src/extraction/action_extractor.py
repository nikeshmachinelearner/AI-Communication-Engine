import re

from spacy.tokens import Doc, Span, Token

from src.extraction.schemas import (
    ExtractedCommitment,
    ExtractedTask,
)


TASK_PREFIX_PATTERNS = [
    re.compile(r"^\s*please\b", re.IGNORECASE),
    re.compile(r"^\s*kindly\b", re.IGNORECASE),
    re.compile(
        r"^\s*(?:can|could|would|will)\s+you\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*you\s+need\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*remember\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*do\s+not\s+forget\s+to\b", re.IGNORECASE),
    re.compile(r"\bwe\s+need\s+to\b", re.IGNORECASE),
    re.compile(r"\bneed\s+you\s+to\b", re.IGNORECASE),
]

COMMITMENT_PATTERNS = [
    re.compile(
        r"\bI(?:'ll|\s+will|\s+can|\s+shall)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe(?:'ll|\s+will|\s+can|\s+shall)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+promise\s+to\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+commit\s+to\s+(.+)",
        re.IGNORECASE,
    ),
]


def clean_span(text: str) -> str:
    return text.strip().rstrip(".,;:!?")


def find_action_verb(sentence: Span) -> Token | None:
    preferred_dependencies = {
        "ROOT",
        "xcomp",
        "ccomp",
        "conj",
    }

    for token in sentence:
        if token.pos_ == "VERB" and token.dep_ in preferred_dependencies:
            return token

    for token in sentence:
        if token.pos_ == "VERB":
            return token

    return None


def extract_assignee(sentence: Span) -> str | None:
    lowered = sentence.text.lower()

    if re.search(
        r"\b(?:can|could|would|will)\s+you\b",
        lowered,
    ):
        return "you"

    if re.search(r"\byou\s+need\s+to\b", lowered):
        return "you"

    if re.search(r"\bI\s+(?:will|can|shall)\b", sentence.text):
        return "I"

    if re.search(r"\bwe\s+(?:will|can|shall|need\s+to)\b", lowered):
        return "we"

    for token in sentence:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            return token.text

    return None


def extract_tasks(doc: Doc) -> list[ExtractedTask]:
    tasks = []
    seen = set()

    for sentence in doc.sents:
        sentence_text = clean_span(sentence.text)

        if not sentence_text:
            continue

        is_task = any(
            pattern.search(sentence_text)
            for pattern in TASK_PREFIX_PATTERNS
        )

        first_token = next(
            (
                token
                for token in sentence
                if not token.is_space and not token.is_punct
            ),
            None,
        )

        is_imperative = (
            first_token is not None
            and first_token.pos_ == "VERB"
            and first_token.tag_ in {"VB", "VBP"}
        )

        if not is_task and not is_imperative:
            continue

        key = sentence_text.lower()

        if key in seen:
            continue

        action_verb = find_action_verb(sentence)

        tasks.append(
            ExtractedTask(
                text=sentence_text,
                action=(
                    action_verb.lemma_.lower()
                    if action_verb is not None
                    else None
                ),
                assignee=extract_assignee(sentence),
                start=sentence.start_char,
                end=sentence.end_char,
            )
        )

        seen.add(key)

    return tasks


def extract_commitments(
    doc: Doc,
) -> list[ExtractedCommitment]:
    commitments = []
    seen = set()

    for sentence in doc.sents:
        sentence_text = clean_span(sentence.text)

        for pattern in COMMITMENT_PATTERNS:
            match = pattern.search(sentence_text)

            if match is None:
                continue

            commitment_text = clean_span(match.group(0))
            key = commitment_text.lower()

            if key in seen:
                continue

            owner = (
                "I"
                if commitment_text.lower().startswith("i")
                else "we"
            )

            action_verb = find_action_verb(sentence)

            commitments.append(
                ExtractedCommitment(
                    text=commitment_text,
                    owner=owner,
                    action=(
                        action_verb.lemma_.lower()
                        if action_verb is not None
                        else None
                    ),
                )
            )

            seen.add(key)

    return commitments