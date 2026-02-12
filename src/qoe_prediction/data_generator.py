"""
Domain-informed synthetic data generator for Telecom QoE Prediction.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """Base class for generating synthetic telecom data."""

    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_sinr(
        self, n: int, base_sinr_db: float = 10.0, noise_std: float = 5.0
    ) -> np.ndarray:
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)

    def sinr_to_throughput(
        self, sinr_db: np.ndarray, network_type: np.ndarray, noise_factor: float = 0.2
    ) -> np.ndarray:
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_factor = np.log2(1 + sinr_linear)
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        return np.clip(throughput, 0.1, max_throughput)

    def generate_congestion_pattern(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        return np.clip(congestion, 0, 1)

    def congestion_to_latency(
        self, congestion: np.ndarray, base_latency_ms: float = 20
    ) -> np.ndarray:
        latency = base_latency_ms * (1 + 5 * congestion**2)
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        return np.clip(latency, 10, 300)

    def compute_qoe_mos(
        self,
        throughput_mbps: np.ndarray,
        latency_ms: np.ndarray,
        packet_loss_pct: np.ndarray,
        app_type: np.ndarray,
    ) -> np.ndarray:
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        loss_penalty = packet_loss_pct / 2
        mos = mos_throughput - latency_penalty - loss_penalty
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        return np.clip(mos, 1, 5)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


class QoEDataGenerator(TelecomDataGenerator):
    """Generates synthetic session-level data for QoE (MOS) prediction."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 10_000,
        app_types: Optional[list] = None,
        app_weights: Optional[list] = None,
        device_classes: Optional[list] = None,
        device_weights: Optional[list] = None,
    ):
        super().__init__(seed=seed, n_samples=n_samples)
        self.app_types = app_types or [
            "video_streaming",
            "browsing",
            "gaming",
            "social",
            "voip",
        ]
        self.app_weights = app_weights or [0.25, 0.30, 0.15, 0.15, 0.15]
        self.device_classes = device_classes or ["low", "mid", "high"]
        self.device_weights = device_weights or [0.2, 0.5, 0.3]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate session-level QoE dataset.

        Returns
        -------
        pd.DataFrame
            One row per session with network KPIs and the target MOS score.
        """
        n = self.n_samples

        # --- identifiers ---
        session_ids = [f"SES_{i:07d}" for i in range(n)]

        # --- timestamps: random instants over 30 days ---
        start = pd.Timestamp("2024-01-01")
        random_seconds = self.rng.integers(0, 30 * 24 * 3600, size=n)
        timestamps = pd.DatetimeIndex(
            [start + pd.Timedelta(seconds=int(s)) for s in random_seconds]
        )

        # --- categorical features ---
        network_type = self.rng.choice(["4G", "5G"], size=n, p=[0.6, 0.4])
        device_class = self.rng.choice(
            self.device_classes,
            size=n,
            p=self.device_weights,
        )
        app_type = self.rng.choice(
            self.app_types,
            size=n,
            p=self.app_weights,
        )

        # --- SINR (base varies by device quality) ---
        device_sinr_base = {"low": 7.0, "mid": 10.0, "high": 13.0}
        base_sinr = np.array([device_sinr_base[d] for d in device_class])
        sinr_db = self.rng.normal(base_sinr, 4.0)
        sinr_db = np.clip(sinr_db, -5, 25)

        # --- throughput from SINR, capped by device class ---
        throughput_mbps = self.sinr_to_throughput(sinr_db, network_type)
        device_max_tp = {"low": 20.0, "mid": 50.0, "high": 150.0}
        max_tp = np.array([device_max_tp[d] for d in device_class])
        throughput_mbps = np.minimum(throughput_mbps, max_tp)

        # --- congestion & latency ---
        congestion_level = self.generate_congestion_pattern(timestamps)
        latency_ms = self.congestion_to_latency(congestion_level)

        # --- packet loss ---
        packet_loss_pct = self.rng.exponential(0.5, n)
        packet_loss_pct = np.clip(packet_loss_pct, 0, 5)

        # --- session duration (gamma) ---
        session_duration_min = self.rng.gamma(shape=3, scale=5, size=n)
        session_duration_min = np.clip(session_duration_min, 1, 120)

        # --- data volume ---
        # throughput (Mbps) * duration (min) * 60 / 8 -> MB, plus noise
        data_volume_mb = (
            throughput_mbps * session_duration_min * (60 / 8) * self.rng.uniform(0.5, 1.0, n)
        )

        # --- target: MOS score ---
        mos_score = self.compute_qoe_mos(
            throughput_mbps,
            latency_ms,
            packet_loss_pct,
            app_type,
        )

        df = pd.DataFrame(
            {
                "session_id": session_ids,
                "timestamp": timestamps,
                "network_type": network_type,
                "device_class": device_class,
                "app_type": app_type,
                "sinr_db": sinr_db,
                "throughput_mbps": throughput_mbps,
                "latency_ms": latency_ms,
                "packet_loss_pct": packet_loss_pct,
                "congestion_level": congestion_level,
                "session_duration_min": session_duration_min,
                "data_volume_mb": data_volume_mb,
                "mos_score": mos_score,
            }
        )

        print(
            f"Generated {len(df):,} sessions  |  "
            f"MOS mean={df['mos_score'].mean():.2f}  "
            f"std={df['mos_score'].std():.2f}"
        )
        return df


def main() -> None:
    """Generate QoE prediction dataset using project configuration."""
    config = DATA_GEN_CONFIG
    use_case = config["use_case_params"]

    generator = QoEDataGenerator(
        seed=config["random_seed"],
        n_samples=config["n_samples"],
        app_types=use_case["app_types"],
        app_weights=use_case["app_weights"],
        device_classes=use_case["device_classes"],
        device_weights=use_case["device_weights"],
    )
    df = generator.generate()
    generator.save(df, "qoe_prediction_raw")


if __name__ == "__main__":
    main()
