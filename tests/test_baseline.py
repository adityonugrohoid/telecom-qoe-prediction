"""Tests for the mean-MOS baseline comparison."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qoe_prediction.baseline import run_baseline_comparison

EXPECTED_METRIC_KEYS = {"mse", "rmse", "mae", "r2"}


@pytest.fixture(scope="module")
def results():
    return run_baseline_comparison(seed=42, n_samples=1000)


class TestBaselineComparison:
    def test_returns_seed_and_sample_size(self, results):
        assert results["seed"] == 42
        assert results["n_samples"] == 1000

    def test_split_sizes_sum_to_the_sample_size(self, results):
        assert results["train_size"] + results["test_size"] == results["n_samples"]

    def test_baseline_block_has_expected_keys(self, results):
        assert set(results["baseline"]["metrics"]) == EXPECTED_METRIC_KEYS

    def test_model_block_has_expected_keys(self, results):
        assert set(results["model"]["metrics"]) == EXPECTED_METRIC_KEYS

    def test_model_is_no_worse_than_the_baseline(self, results):
        assert results["model"]["metrics"]["rmse"] <= results["baseline"]["metrics"]["rmse"]
