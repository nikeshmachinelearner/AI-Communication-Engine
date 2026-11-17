from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from src.ranking.features import FEATURE_COLUMNS, extract_features

OUTPUT_PATH = Path("data/processed/ranking_dataset.csv")

REFERENCE_TIME = datetime(
    2026,
    11,
    17,
    14,
    0,
    tzinfo=ZoneInfo("Asia/Kathmandu"),
)

GROUP_TEMPLATES = [
    [
        {
            "text": "Urgent: the production server is down. Investigate immediately.",
            "sender_importance": 1.0,
            "conversation_relevance": 1.0,
            "hours_until_deadline": 0,
            "relevance": 4,
        },
        {
            "text": "Please submit the updated project report by tomorrow at 10 AM.",
            "sender_importance": 0.9,
            "conversation_relevance": 0.9,
            "hours_until_deadline": 20,
            "relevance": 4,
        },
        {
            "text": "Can we schedule a project meeting next week?",
            "sender_importance": 0.7,
            "conversation_relevance": 0.8,
            "hours_until_deadline": None,
            "relevance": 2,
        },
        {
            "text": "Hope you had a relaxing weekend.",
            "sender_importance": 0.5,
            "conversation_relevance": 0.2,
            "hours_until_deadline": None,
            "relevance": 0,
        },
        {
            "text": "The weekly marketing newsletter is now available.",
            "sender_importance": 0.1,
            "conversation_relevance": 0.1,
            "hours_until_deadline": None,
            "relevance": 0,
        },
    ],
    [
        {
            "text": "The security team detected a critical account breach.",
            "sender_importance": 1.0,
            "conversation_relevance": 1.0,
            "hours_until_deadline": 0,
            "relevance": 4,
        },
        {
            "text": "Please reset your account password today.",
            "sender_importance": 0.9,
            "conversation_relevance": 0.9,
            "hours_until_deadline": 5,
            "relevance": 4,
        },
        {
            "text": "The client approved the final project design.",
            "sender_importance": 0.9,
            "conversation_relevance": 0.9,
            "hours_until_deadline": None,
            "relevance": 3,
        },
        {
            "text": "Would you like to join a social event next month?",
            "sender_importance": 0.4,
            "conversation_relevance": 0.2,
            "hours_until_deadline": None,
            "relevance": 1,
        },
        {
            "text": "Thanks for the presentation.",
            "sender_importance": 0.6,
            "conversation_relevance": 0.3,
            "hours_until_deadline": None,
            "relevance": 0,
        },
    ],
    [
        {
            "text": "The customer cannot make payments and needs help now.",
            "sender_importance": 1.0,
            "conversation_relevance": 1.0,
            "hours_until_deadline": 0,
            "relevance": 4,
        },
        {
            "text": "Please review the budget before Friday.",
            "sender_importance": 0.9,
            "conversation_relevance": 0.8,
            "hours_until_deadline": 44,
            "relevance": 3,
        },
        {
            "text": "The mobile application release was postponed to Monday.",
            "sender_importance": 0.8,
            "conversation_relevance": 0.9,
            "hours_until_deadline": None,
            "relevance": 3,
        },
        {
            "text": "Can we have a call sometime next month?",
            "sender_importance": 0.5,
            "conversation_relevance": 0.5,
            "hours_until_deadline": None,
            "relevance": 1,
        },
        {
            "text": "Have a great evening.",
            "sender_importance": 0.4,
            "conversation_relevance": 0.2,
            "hours_until_deadline": None,
            "relevance": 0,
        },
    ],
]


def main() -> None:
    rows = []
    for group_id, group in enumerate(GROUP_TEMPLATES):
        for item in group:
            # Pass 'item' dictionary directly as the 'message' argument
            features = extract_features(
                message=item,
                reference_time=REFERENCE_TIME,
            )
            row = {
                "group_id": group_id,
                **features,
                "relevance": item["relevance"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset successfully created and saved to {OUTPUT_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    main()