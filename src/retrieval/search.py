import argparse
import json
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("models/embeddings/messages.faiss")
METADATA_PATH = Path("models/embeddings/metadata.json")
INDEX_INFO_PATH = Path("models/embeddings/index_info.json")


class ContextRetriever:
    def __init__(self) -> None:
        self._validate_files()

        with INDEX_INFO_PATH.open("r", encoding="utf-8") as file:
            index_info = json.load(file)

        self.model_name = index_info["model_name"]
        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(str(INDEX_PATH))

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            self.messages = json.load(file)

        if self.index.ntotal != len(self.messages):
            raise ValueError(
                "FAISS index and metadata contain different numbers "
                "of messages."
            )

    @staticmethod
    def _validate_files() -> None:
        required_paths = [
            INDEX_PATH,
            METADATA_PATH,
            INDEX_INFO_PATH,
        ]

        missing_paths = [
            str(path) for path in required_paths if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                "Missing retrieval artifacts: "
                f"{missing_paths}. Run "
                "python -m src.retrieval.build_index"
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        query = query.strip()

        if not query:
            raise ValueError("The query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start_time = perf_counter()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        # Retrieve extra candidates when metadata filtering is requested.
        candidate_count = (
            len(self.messages)
            if conversation_id
            else min(top_k, len(self.messages))
        )

        scores, indices = self.index.search(
            query_embedding,
            candidate_count,
        )

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position == -1:
                continue

            message = self.messages[int(index_position)]

            if (
                conversation_id is not None
                and message["conversation_id"] != conversation_id
            ):
                continue

            score = float(score)

            if minimum_score is not None and score < minimum_score:
                continue

            results.append(
                {
                    "rank": len(results) + 1,
                    "similarity_score": round(score, 4),
                    **message,
                }
            )

            if len(results) == top_k:
                break

        latency_ms = (perf_counter() - start_time) * 1000

        return {
            "query": query,
            "top_k": top_k,
            "conversation_filter": conversation_id,
            "latency_ms": round(latency_ms, 2),
            "results": results,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve historical messages by semantic similarity."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Incoming message or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of messages to return.",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help="Optional minimum cosine-similarity score.",
    )

    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Optionally restrict results to one conversation.",
    )

    arguments = parser.parse_args()

    retriever = ContextRetriever()

    response = retriever.search(
        query=arguments.query,
        top_k=arguments.top_k,
        minimum_score=arguments.minimum_score,
        conversation_id=arguments.conversation_id,
    )

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()