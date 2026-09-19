"""
Baseline comparison for Telecom QoE Prediction.

The README names "Mean MOS prediction" as the baseline in its training
approach table. This module scores that baseline against the project's
LightGBM regressor on one shared train/test split, using the project's
own generator, feature pipeline, and metric computation.
"""

import json

from sklearn.dummy import DummyRegressor

from .config import PROJECT_ROOT
from .data_generator import QoEDataGenerator
from .features import FeatureEngineer
from .models import BaseModel, LightGBMQoERegressor

NON_FEATURE_COLUMNS = ["session_id", "timestamp"]
TARGET_COLUMN = "mos_score"

README_SEED = 42
README_N_SAMPLES = 10_000

RESULTS_PATH = PROJECT_ROOT / "evidence" / "baseline_metrics.json"


def run_baseline_comparison(seed: int, n_samples: int) -> dict:
    """Score the README's mean baseline against the project's model.

    Generates data with the project generator, runs the project feature
    pipeline, makes one train/test split, and trains both the mean-MOS
    baseline and the LightGBM regressor on that same split.

    Args:
        seed: Random seed for data generation and the train/test split.
        n_samples: Number of synthetic sessions to generate.

    Returns:
        Dictionary with the seed, sample size, split sizes, and a
        metrics block for each of the baseline and the model.
    """
    raw = QoEDataGenerator(seed=seed, n_samples=n_samples).generate()
    features = FeatureEngineer().pipeline(raw)
    features = features.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in features.columns])

    model = LightGBMQoERegressor()
    X_train, X_test, y_train, y_test = model.prepare_data(
        features, target_col=TARGET_COLUMN, random_state=seed
    )

    baseline = BaseModel()
    baseline.model = DummyRegressor(strategy="mean")
    baseline.model.fit(X_train, y_train)
    baseline.is_trained = True
    baseline_metrics = baseline.evaluate(X_test, y_test, task_type="regression")

    model.train(X_train, y_train)
    model_metrics = model.evaluate(X_test, y_test, task_type="regression")

    return {
        "seed": seed,
        "n_samples": n_samples,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "baseline": {"name": "mean_mos_prediction", "metrics": baseline_metrics},
        "model": {"name": "lightgbm_qoe_regressor", "metrics": model_metrics},
    }


def main() -> None:
    """Run the baseline comparison at the README's scale and seed, and write results."""
    results = run_baseline_comparison(seed=README_SEED, n_samples=README_N_SAMPLES)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Baseline comparison written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
