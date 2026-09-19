"""Tests for model training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qoe_prediction.data_generator import QoEDataGenerator
from qoe_prediction.features import FeatureEngineer
from qoe_prediction.models import BaseModel, LightGBMQoERegressor

TARGET = "mos_score"
NON_FEATURES = ["session_id", "timestamp"]

# MOS is clipped to [1, 5] by the generator. LightGBM predictions are sums
# of leaf values learned from training targets, so a small overshoot past
# the clip bounds is possible; anything outside this wider band signals a
# broken target or a unit mismatch in the feature pipeline.
MOS_PREDICTION_BAND = (0.5, 5.5)

# The README reports RMSE 0.45 on 10,000 sessions. The ceiling here is
# higher because the test trains on 3,000 sessions to stay fast; a rise
# above it means the signal in the generator or the feature pipeline has
# broken.
RMSE_CEILING = 0.55


@pytest.fixture(scope="module")
def features():
    """Seeded data run through the project's own feature pipeline."""
    raw = QoEDataGenerator(seed=42, n_samples=3000).generate()
    features = FeatureEngineer().pipeline(raw)
    features = features.drop(columns=[c for c in NON_FEATURES if c in features.columns])
    return features


@pytest.fixture(scope="module")
def trained(features):
    """The model as the project uses it: prepare_data, then train."""
    model = LightGBMQoERegressor()
    X_train, _, y_train, _ = model.prepare_data(features, target_col=TARGET)
    model.train(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def split(features):
    return LightGBMQoERegressor().prepare_data(features, target_col=TARGET)


class TestTraining:
    def test_untrained_model_refuses_to_predict(self, split):
        _, X_test, _, _ = split
        with pytest.raises(ValueError):
            LightGBMQoERegressor().predict(X_test)

    def test_base_model_has_no_training(self, split):
        X_train, _, y_train, _ = split
        with pytest.raises(NotImplementedError):
            BaseModel().train(X_train, y_train)

    def test_training_marks_the_model_trained(self, trained):
        assert trained.is_trained

    def test_training_is_reproducible(self, split, trained):
        X_train, X_test, y_train, _ = split
        again = LightGBMQoERegressor()
        again.train(X_train, y_train)
        np.testing.assert_allclose(
            trained.predict(X_test), again.predict(X_test), rtol=0, atol=1e-7
        )


class TestEvaluation:
    def test_predictions_have_valid_shape_and_range(self, split, trained):
        _, X_test, _, _ = split
        predictions = trained.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert predictions.min() >= MOS_PREDICTION_BAND[0]
        assert predictions.max() <= MOS_PREDICTION_BAND[1]

    def test_metrics_are_complete(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test)
        assert set(metrics) == {"mse", "rmse", "mae", "r2"}

    def test_rmse_stays_below_the_ceiling(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test)
        assert metrics["rmse"] <= RMSE_CEILING, f"RMSE rose to {metrics['rmse']:.4f}"

    def test_model_beats_predicting_the_training_mean(self, split, trained):
        _, X_test, y_train, y_test = split
        mean_prediction = np.full(len(y_test), y_train.mean())
        mean_rmse = np.sqrt(mean_squared_error(y_test, mean_prediction))
        metrics = trained.evaluate(X_test, y_test)
        assert metrics["rmse"] < mean_rmse

    def test_feature_importance_covers_every_feature(self, split, trained):
        X_train, _, _, _ = split
        importance = trained.get_feature_importance()
        assert sorted(importance["feature"]) == sorted(X_train.columns)
        assert (importance["importance"] >= 0).all()
        assert importance["importance"].sum() > 0


class TestPersistence:
    def test_saved_model_predicts_the_same(self, split, trained, tmp_path):
        _, X_test, _, _ = split
        path = tmp_path / "qoe.pkl"
        trained.save(path)
        restored = LightGBMQoERegressor()
        restored.load(path)
        np.testing.assert_array_equal(trained.predict(X_test), restored.predict(X_test))

    def test_untrained_model_refuses_to_save(self, tmp_path):
        with pytest.raises(ValueError):
            LightGBMQoERegressor().save(tmp_path / "qoe.pkl")
