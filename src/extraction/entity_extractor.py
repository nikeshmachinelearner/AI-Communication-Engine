import re

import spacy
from spacy.language import Language

from src.extraction.schemas import ExtractedEntity


LOCATION_PATTERNS = [
    re.compile(
        r"\bconference room\s+[A-Za-z0-9-]+\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bmeeting room\s+[A-Za-z0-9-]+\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\broom\s+[A-Za-z0-9-]+\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:on|via)\s+(Microsoft Teams|Teams|Zoom|Google Meet)\b",
        flags=re.IGNORECASE,
    ),
]

RECIPIENT_PREPOSITIONS = {"to", "for", "with", "from", "cc"}
NON_PERSON_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "zoom",
    "teams",
    "microsoft",
    "google",
    "q1",
    "q2",
    "q3",
    "q4",
}


class EntityExtractor:
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
    ) -> None:
        self.model_name = model_name
        self.nlp = self._load_model(model_name)
        self._add_business_entity_patterns()

    @staticmethod
    def _load_model(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError as error:
            raise RuntimeError(
                f"spaCy model '{model_name}' is unavailable. "
                "Run: python -m spacy download en_core_web_sm"
            ) from error

    def _add_business_entity_patterns(self) -> None:
        if "entity_ruler" in self.nlp.pipe_names:
            return

        ruler = self.nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={
                "overwrite_ents": False,
                "validate": True,
                "phrase_matcher_attr": "LOWER",
            },
        )

        patterns = [
            {
                "label": "LOCATION",
                "pattern": [
                    {"LOWER": "conference"},
                    {"LOWER": "room"},
                    {"TEXT": {"REGEX": "^[A-Za-z0-9-]+$"}},
                ],
            },
            {
                "label": "LOCATION",
                "pattern": [
                    {"LOWER": "meeting"},
                    {"LOWER": "room"},
                    {"TEXT": {"REGEX": "^[A-Za-z0-9-]+$"}},
                ],
            },
            {
                "label": "LOCATION",
                "pattern": [
                    {"LOWER": "room"},
                    {"TEXT": {"REGEX": "^[A-Za-z0-9-]+$"}},
                ],
            },
            {
                "label": "LOCATION",
                "pattern": [{"LOWER": "microsoft"}, {"LOWER": "teams"}],
            },
            {
                "label": "LOCATION",
                "pattern": [{"LOWER": "google"}, {"LOWER": "meet"}],
            },
            {
                "label": "LOCATION",
                "pattern": [{"LOWER": "zoom"}],
            },
        ]

        ruler.add_patterns(patterns)

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        output = []
        seen = set()

        for value in values:
            cleaned = value.strip()
            key = cleaned.lower()

            if cleaned and key not in seen:
                output.append(cleaned)
                seen.add(key)

        return output

    def extract(self, text: str) -> dict:
        doc = self.nlp(text)

        people = []
        organizations = []
        locations = []
        entities = []
        location_spans = set()

        for entity in doc.ents:
            entities.append(
                ExtractedEntity(
                    text=entity.text,
                    label=entity.label_,
                    start=entity.start_char,
                    end=entity.end_char,
                )
            )

            if entity.label_ == "PERSON":
                people.append(entity.text)

            elif entity.label_ == "ORG":
                organizations.append(entity.text)

            elif entity.label_ in {
                "GPE",
                "LOC",
                "FAC",
                "LOCATION",
            }:
                locations.append(entity.text)
                location_spans.add((entity.start, entity.end))

        for pattern in LOCATION_PATTERNS:
            for match in pattern.finditer(text):
                matched_location = match.group(0)

                matched_location = re.sub(
                    r"^(on|via)\s+",
                    "",
                    matched_location,
                    flags=re.IGNORECASE,
                )

                locations.append(matched_location)

        # Fallback: capture capitalized recipient names after prepositions ("to Maya")
        # when en_core_web_sm tags them as NOUN or misses PERSON in short imperatives.
        for i, token in enumerate(doc):
            if (
                i > 0
                and doc[i - 1].lower_ in RECIPIENT_PREPOSITIONS
                and token.is_title
                and token.is_alpha
                and token.lower_ not in NON_PERSON_WORDS
                and not any(start <= token.i < end for start, end in location_spans)
            ):
                end_i = i
                while (
                    end_i + 1 < len(doc)
                    and doc[end_i + 1].is_title
                    and doc[end_i + 1].is_alpha
                    and doc[end_i + 1].lower_ not in NON_PERSON_WORDS
                ):
                    end_i += 1

                person_span = doc[i : end_i + 1]
                people.append(person_span.text)
                entities.append(
                    ExtractedEntity(
                        text=person_span.text,
                        label="PERSON",
                        start=person_span.start_char,
                        end=person_span.end_char,
                    )
                )

        return {
            "doc": doc,
            "people": self._deduplicate(people),
            "organizations": self._deduplicate(organizations),
            "locations": self._deduplicate(locations),
            "entities": entities,
        }