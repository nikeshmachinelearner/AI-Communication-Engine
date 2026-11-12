import json
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/processed/message_history.json")
INDEX_PATH = Path("models/embeddings/messages.faiss")
METADATA_PATH = Path("models/embeddings/metadata.json")
INDEX_INFO_PATH = Path("models/embeddings/index_info.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_messages() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run "
            "python -m src.retrieval.create_message_history"
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    messages = load_messages()
    texts = [message["text"] for message in messages]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    start_time = perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_time_ms = (perf_counter() - start_time) * 1000

    embeddings = np.asarray(embeddings, dtype="float32")

    embedding_dimension = embeddings.shape[1]

    # Because embeddings are normalized, inner product equals cosine similarity.
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(messages, file, indent=2)

    index_info = {
        "model_name": MODEL_NAME,
        "index_type": "IndexFlatIP",
        "similarity": "cosine_similarity_via_normalized_inner_product",
        "message_count": len(messages),
        "embedding_dimension": embedding_dimension,
        "embedding_time_ms": round(embedding_time_ms, 2),
    }

    with INDEX_INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(index_info, file, indent=2)

    print(f"Indexed messages: {index.ntotal}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Index information saved to: {INDEX_INFO_PATH}")


if __name__ == "__main__":
    main()