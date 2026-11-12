## Module 1: Intelligent Message Classification

Module 1 uses multi-label classification because a communication message can
represent multiple intentions. For example, a message may contain both a task
and a meeting request.

### Labels

- Question
- Task
- Meeting request
- Important information
- General conversation
- Urgent

### Baseline

The initial baseline uses:

- TF-IDF word and bigram features
- One-vs-rest logistic regression
- Class balancing
- A configurable prediction threshold

# Module 2: Context Retrieval Engine
 
The context retrieval engine searches historical messages using semantic
similarity.
 
### Baseline architecture
 
1. Historical messages are converted into dense vectors using
`sentence-transformers/all-MiniLM-L6-v2`.
2. Embeddings are normalized to unit length.
3. The vectors are stored in a FAISS `IndexFlatIP` index.
4. Incoming messages are embedded using the same model.
5. FAISS returns the top-K messages using normalized inner-product search,
which is equivalent to cosine similarity.
 
### Build the retrieval index
 
```bash
python -m src.retrieval.create_message_history
python -m src.retrieval.build_index
Search historical context
Shell
python -m src.retrieval.search \
--query "Are we still meeting tomorrow?" \
--top-k 5
Evaluate retrieval
Shell
python -m src.retrieval.evaluate

The evaluation reports:

Recall@1
Recall@3
Recall@5
Mean Reciprocal Rank
Mean query latency
Current limitations
The current corpus is a small synthetic dataset.
Semantic similarity does not yet consider recency.
Metadata filtering happens after vector search.
No keyword-based retrieval is included yet.
No cross-encoder reranker is included yet.

---

# 15. Git commit

Check the files:

```bash
git status


Commit:

Shell
git add .
git commit -m "Build semantic context retrieval baseline with FAISS"

Push:

Shell
git push
16. What this first Module 2 milestone proves

You now have:

Historical message storage
Message metadata
Sentence Transformer embeddings
A persistent FAISS index
Cosine-similarity search
Top-K retrieval
Conversation filtering
Minimum-score filtering
Recall@K evaluation
MRR evaluation
Latency measurement
Automated retrieval tests
Module 2 development roadmap

The complete Module 2 should be developed in these milestones:

Milestone 2.1: Dense retrieval

What we built above:

Plain Text
Sentence Transformer + FAISS
Milestone 2.2: Keyword baseline

Build:

Plain Text
TF-IDF or BM25 retrieval

This gives us a baseline that is good at:

Exact names
Message IDs
Project names
Specific dates
Rare technical terminology
Milestone 2.3: Hybrid search

Combine:

Plain Text
Semantic score + keyword score

Example:

Plain Text
hybrid_score =
0.70 × semantic_score
+
0.30 × keyword_score

The weights must be selected using validation data, not guessed from test results.

Milestone 2.4: Recency and metadata

Add:

Plain Text
final_score =
semantic relevance
+ keyword relevance
+ recency weight
+ conversation relevance
Milestone 2.5: Reranking

Retrieve perhaps 20 candidates using FAISS, then rerank them with a Cross-Encoder:

Plain Text
Query
↓
FAISS retrieves top 20
↓
Cross-Encoder scores query-message pairs
↓
Return best 5

Sentence Transformers documents this retrieve-and-rerank pattern, with a bi-encoder used for efficient candidate retrieval and a Cross-Encoder used for more precise reranking.

Milestone 2.6: Production integration

Then expose retrieval through FastAPI:

HTTP
POST /retrieve

Example request:

JSON
{
"query": "Are we still meeting tomorrow?",
"top_k": 5,
"conversation_id": "project_alpha"
}

The immediate goal is to complete Milestone 2.1 and verify that these three commands succeed:

Shell
python -m src.retrieval.build_index
python -m src.retrieval.evaluate
pytest -v

Only after those work should you add BM25 and hybrid retrieval. This preserves a measurable baseline and makes the later improvement defensible in your GitHub portfolio.

### Run the baseline

```bash
python -m src.classification.create_dataset
python -m src.classification.train_baseline
python -m src.classification.predict \
  --text "Please send the updated report by tomorrow."
pytest -v