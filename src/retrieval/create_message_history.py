import json
from pathlib import Path


OUTPUT_PATH = Path("data/processed/message_history.json")


def build_message_history() -> list[dict]:
    messages = [
        {
            "message_id": "msg_001",
            "conversation_id": "project_alpha",
            "sender": "Aarav",
            "timestamp": "2026-11-03T09:00:00",
            "text": "Let's hold the Project Alpha review on Tuesday at 2 PM.",
        },
        {
            "message_id": "msg_002",
            "conversation_id": "project_alpha",
            "sender": "Nikesh",
            "timestamp": "2026-11-03T09:05:00",
            "text": "Tuesday at 2 PM works for me.",
        },
        {
            "message_id": "msg_003",
            "conversation_id": "project_alpha",
            "sender": "Aarav",
            "timestamp": "2026-11-03T09:10:00",
            "text": "The project review will take place in conference room B.",
        },
        {
            "message_id": "msg_004",
            "conversation_id": "project_alpha",
            "sender": "Maya",
            "timestamp": "2026-11-03T10:00:00",
            "text": "Please prepare the updated Project Alpha report before Tuesday.",
        },
        {
            "message_id": "msg_005",
            "conversation_id": "project_alpha",
            "sender": "Nikesh",
            "timestamp": "2026-11-03T10:10:00",
            "text": "I will finish the report and send it before the meeting.",
        },
        {
            "message_id": "msg_006",
            "conversation_id": "budget",
            "sender": "Maya",
            "timestamp": "2026-11-04T11:00:00",
            "text": "The finance team reduced the marketing budget by ten percent.",
        },
        {
            "message_id": "msg_007",
            "conversation_id": "budget",
            "sender": "Aarav",
            "timestamp": "2026-11-04T11:05:00",
            "text": "We should discuss the revised budget before Friday.",
        },
        {
            "message_id": "msg_008",
            "conversation_id": "budget",
            "sender": "Nikesh",
            "timestamp": "2026-11-04T11:10:00",
            "text": "I can join a budget discussion on Thursday afternoon.",
        },
        {
            "message_id": "msg_009",
            "conversation_id": "client_portal",
            "sender": "Sofia",
            "timestamp": "2026-11-04T13:00:00",
            "text": "The client cannot log in to the customer portal.",
        },
        {
            "message_id": "msg_010",
            "conversation_id": "client_portal",
            "sender": "Nikesh",
            "timestamp": "2026-11-04T13:08:00",
            "text": "I reset the client's password and asked them to try again.",
        },
        {
            "message_id": "msg_011",
            "conversation_id": "client_portal",
            "sender": "Sofia",
            "timestamp": "2026-11-04T13:20:00",
            "text": "The customer confirmed that account access is working now.",
        },
        {
            "message_id": "msg_012",
            "conversation_id": "release",
            "sender": "Maya",
            "timestamp": "2026-11-05T08:30:00",
            "text": "The mobile application release has been postponed to Monday.",
        },
        {
            "message_id": "msg_013",
            "conversation_id": "release",
            "sender": "Aarav",
            "timestamp": "2026-11-05T08:35:00",
            "text": "The release was delayed because two critical bugs remain open.",
        },
        {
            "message_id": "msg_014",
            "conversation_id": "release",
            "sender": "Nikesh",
            "timestamp": "2026-11-05T08:40:00",
            "text": "I will test the payment bug before the new release date.",
        },
        {
            "message_id": "msg_015",
            "conversation_id": "general",
            "sender": "Sofia",
            "timestamp": "2026-11-05T12:00:00",
            "text": "I hope everyone had a relaxing weekend.",
        },
        {
            "message_id": "msg_016",
            "conversation_id": "general",
            "sender": "Aarav",
            "timestamp": "2026-11-05T12:05:00",
            "text": "The football match yesterday was exciting.",
        },
        {
            "message_id": "msg_017",
            "conversation_id": "general",
            "sender": "Nikesh",
            "timestamp": "2026-11-05T12:10:00",
            "text": "Thanks for helping me with the presentation.",
        },
        {
            "message_id": "msg_018",
            "conversation_id": "security",
            "sender": "Maya",
            "timestamp": "2026-11-06T09:00:00",
            "text": "All employees must change their passwords before November 15.",
        },
        {
            "message_id": "msg_019",
            "conversation_id": "security",
            "sender": "Sofia",
            "timestamp": "2026-11-06T09:10:00",
            "text": "The new password must contain at least twelve characters.",
        },
        {
            "message_id": "msg_020",
            "conversation_id": "security",
            "sender": "Nikesh",
            "timestamp": "2026-11-06T09:20:00",
            "text": "I have updated my account password according to the new policy.",
        },
    ]
    return messages


def validate_messages(messages: list[dict]) -> None:
    required_fields = {
        "message_id",
        "conversation_id",
        "sender",
        "timestamp",
        "text",
    }

    message_ids = set()

    for message in messages:
        missing_fields = required_fields.difference(message)

        if missing_fields:
            raise ValueError(
                f"Message is missing fields: {sorted(missing_fields)}"
            )

        if message["message_id"] in message_ids:
            raise ValueError(
                f"Duplicate message ID: {message['message_id']}"
            )

        if not message["text"].strip():
            raise ValueError(
                f"Empty text in message: {message['message_id']}"
            )

        message_ids.add(message["message_id"])


def main() -> None:
    messages = build_message_history()
    validate_messages(messages)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(messages, file, indent=2)

    print(f"Saved {len(messages)} messages to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()