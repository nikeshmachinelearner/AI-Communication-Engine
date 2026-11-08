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

### Run the baseline

```bash
python -m src.classification.create_dataset
python -m src.classification.train_baseline
python -m src.classification.predict \
  --text "Please send the updated report by tomorrow."
pytest -v