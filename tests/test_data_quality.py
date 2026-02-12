"""Tests for data quality and validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qoe_prediction.data_generator import QoEDataGenerator


@pytest.fixture
def sample_data():
    generator = QoEDataGenerator(seed=42, n_samples=1000)
    return generator.generate()


class TestDataQuality:
    def test_no_missing_values(self, sample_data):
        critical_cols = ["session_id", "sinr_db", "throughput_mbps", "latency_ms", "mos_score"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values in {col}"

    def test_data_types(self, sample_data):
        assert pd.api.types.is_numeric_dtype(sample_data["sinr_db"])
        assert pd.api.types.is_numeric_dtype(sample_data["throughput_mbps"])
        assert pd.api.types.is_numeric_dtype(sample_data["mos_score"])

    def test_value_ranges(self, sample_data):
        assert sample_data["sinr_db"].min() >= -5
        assert sample_data["sinr_db"].max() <= 25
        assert sample_data["throughput_mbps"].min() > 0
        assert sample_data["mos_score"].min() >= 1
        assert sample_data["mos_score"].max() <= 5
        assert sample_data["latency_ms"].min() >= 10
        assert sample_data["packet_loss_pct"].min() >= 0
        assert sample_data["packet_loss_pct"].max() <= 5

    def test_categorical_values(self, sample_data):
        assert set(sample_data["network_type"].unique()).issubset({"4G", "5G"})
        assert set(sample_data["device_class"].unique()).issubset({"low", "mid", "high"})
        assert set(sample_data["app_type"].unique()).issubset(
            {"video_streaming", "browsing", "gaming", "social", "voip"}
        )

    def test_sample_size(self, sample_data):
        assert len(sample_data) == 1000

    def test_mos_distribution(self, sample_data):
        # MOS should have reasonable mean (around 2.5-4.0)
        mean_mos = sample_data["mos_score"].mean()
        assert 1.5 < mean_mos < 4.5, f"Mean MOS {mean_mos:.2f} outside expected range"


class TestDataGenerator:
    def test_generator_reproducibility(self):
        gen1 = QoEDataGenerator(seed=42, n_samples=100)
        gen2 = QoEDataGenerator(seed=42, n_samples=100)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_sinr_generation(self):
        gen = QoEDataGenerator(seed=42, n_samples=100)
        sinr = gen.generate_sinr(1000)
        assert len(sinr) == 1000
        assert sinr.min() >= -5
        assert sinr.max() <= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
