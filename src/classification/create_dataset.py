from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("data/processed/messages.csv")

LABEL_COLUMNS = [
    "question",
    "task",
    "meeting_request",
    "important_information",
    "general_conversation",
    "urgent",
]


def build_dataset() -> pd.DataFrame:
    records = [
        {
            "text": "What time does the presentation start?",
            "question": 1,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Do you know where the project documents are stored?",
            "question": 1,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Can you explain the latest sales numbers?",
            "question": 1,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "When will the new application be released?",
            "question": 1,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Please update the project report.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Send the revised budget to the finance team.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Review the document before Friday.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Remember to submit your expense claim.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Can we meet tomorrow afternoon?",
            "question": 1,
            "task": 0,
            "meeting_request": 1,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Let us schedule a call for Monday morning.",
            "question": 0,
            "task": 0,
            "meeting_request": 1,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Are you available for a budget meeting at 3 PM?",
            "question": 1,
            "task": 0,
            "meeting_request": 1,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Please book a thirty-minute project review this week.",
            "question": 0,
            "task": 1,
            "meeting_request": 1,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "The production release has been postponed until Monday.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "The client approved the final design.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "The office will be closed next Friday.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "The password policy changes next month.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "I hope you had a nice weekend.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 1,
            "urgent": 0,
        },
        {
            "text": "Thanks for your help today.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 1,
            "urgent": 0,
        },
        {
            "text": "That movie was really enjoyable.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 1,
            "urgent": 0,
        },
        {
            "text": "Have a great evening.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 1,
            "urgent": 0,
        },
        {
            "text": "Urgent, the production server is unavailable.",
            "question": 0,
            "task": 0,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 1,
        },
        {
            "text": "Please call me immediately.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 1,
        },
        {
            "text": "We need the corrected report within the next hour.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 1,
        },
        {
            "text": "The customer cannot access their account and needs help now.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 1,
        },
        {
            "text": "Can you send me the updated report by tomorrow morning?",
            "question": 1,
            "task": 1,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "We need to discuss the budget before Friday.",
            "question": 0,
            "task": 1,
            "meeting_request": 1,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "Are we still meeting tomorrow?",
            "question": 1,
            "task": 0,
            "meeting_request": 1,
            "important_information": 0,
            "general_conversation": 0,
            "urgent": 0,
        },
        {
            "text": "The deadline is tomorrow, so please finish this today.",
            "question": 0,
            "task": 1,
            "meeting_request": 0,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 1,
        },
        {
            "text": "Hi, how are you doing?",
            "question": 1,
            "task": 0,
            "meeting_request": 0,
            "important_information": 0,
            "general_conversation": 1,
            "urgent": 0,
        },
        {
            "text": "Please join the emergency meeting in ten minutes.",
            "question": 0,
            "task": 1,
            "meeting_request": 1,
            "important_information": 1,
            "general_conversation": 0,
            "urgent": 1,
        },
    ]

    dataframe = pd.DataFrame(records)

    assert dataframe["text"].notna().all()
    assert set(dataframe[LABEL_COLUMNS].stack().unique()).issubset({0, 1})
    assert (dataframe[LABEL_COLUMNS].sum(axis=1) >= 1).all()

    return dataframe


def main() -> None:
    dataframe = build_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(dataframe)} messages to {OUTPUT_PATH}")
    print("\nLabel counts:")
    print(dataframe[LABEL_COLUMNS].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()

    