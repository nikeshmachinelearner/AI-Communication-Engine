import json
from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd

from src.ranking.features import FEATURE_COLUMNS

print("--> train.py script has started!")

DATA_PATH = Path("data/processed/ranking_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ranking_lightgbm_ranker.joblib"
EXPERIMENTS_DIR = Path("experiments")


def main() -> None:
    print("--> Inside main() function")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run 'python -m src.ranking.create_dataset' first."
        )

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows.")

    # 2. Separate features, target, and group structure
    X = df[FEATURE_COLUMNS]
    y = df["relevance"]
    
    group_counts = df.groupby("group_id", sort=False).size().to_list()

    # 3. Create LightGBM Dataset
    train_data = lgb.Dataset(X, label=y, group=group_counts)

    # 4. Set LightGBM ranking parameters
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "verbose": -1,
    }

    print("Training LightGBM ranker...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=50,
        valid_sets=[train_data],
        valid_names=["train"],
    )

    # 5. Save model using joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # 6. Save experiment metrics / metadata
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "num_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "num_rows": len(df),
    }
    with open(EXPERIMENTS_DIR / "ranking_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()