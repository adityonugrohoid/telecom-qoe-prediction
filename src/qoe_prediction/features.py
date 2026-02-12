"""
Feature engineering pipeline for Telecom QoE Prediction.

This module handles all feature transformations and engineering,
keeping the logic separate from data generation and modeling.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import FEATURE_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR


class FeatureEngineer:
    """
    Feature engineering pipeline for telecom QoE prediction.
    """

    def __init__(self, config: dict = None):
        """
        Initialize feature engineer.

        Args:
            config: Optional feature config dictionary
        """
        self.config = config or FEATURE_CONFIG

    def create_temporal_features(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Create time-based features from timestamp.

        Args:
            df: Input dataframe
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract time components
        df["hour"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_peak_hour"] = (
            ((df["hour"] >= 9) & (df["hour"] <= 11)) | ((df["hour"] >= 18) & (df["hour"] <= 21))
        ).astype(int)

        return df

    def create_rolling_aggregates(
        self, df: pd.DataFrame, group_col: str, value_cols: List[str], windows: List[int] = [7, 30]
    ) -> pd.DataFrame:
        """
        Create rolling window aggregations.

        Useful for: user behavior trends, network performance over time

        Args:
            df: Input dataframe (must be sorted by time)
            group_col: Column to group by (e.g., "user_id", "cell_id")
            value_cols: Columns to aggregate
            windows: Window sizes in days

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        df = df.sort_values("timestamp")

        for col in value_cols:
            for window in windows:
                df[f"{col}_rolling_{window}d_mean"] = (
                    df.groupby(group_col)[col]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df[f"{col}_rolling_{window}d_std"] = (
                    df.groupby(group_col)[col]
                    .rolling(window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create QoE-specific interaction features.

        Combines network metrics and application characteristics to capture
        quality-of-experience drivers.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with QoE interaction features
        """
        df = df.copy()

        # Network quality index: composite of signal quality and throughput
        if "sinr_db" in df.columns and "throughput_mbps" in df.columns:
            df["network_quality_index"] = 0.5 * (df["sinr_db"] / 25) + 0.5 * (
                df["throughput_mbps"] / 100
            )

        # Service degradation: penalizes high latency and packet loss
        if "latency_ms" in df.columns and "packet_loss_pct" in df.columns:
            df["service_degradation"] = df["latency_ms"] / 100 + df["packet_loss_pct"] * 2

        # Throughput per user (proxy for per-session throughput)
        if "throughput_mbps" in df.columns:
            df["throughput_per_user_mbps"] = df["throughput_mbps"]

        # Bandwidth utilization: how much of theoretical capacity is consumed
        if all(
            c in df.columns for c in ["data_volume_mb", "throughput_mbps", "session_duration_min"]
        ):
            df["bandwidth_utilization"] = df["data_volume_mb"] / (
                df["throughput_mbps"] * df["session_duration_min"] * 60 / 8 + 0.01
            )

        # Application sensitivity score: map app types to latency sensitivity
        if "app_type" in df.columns:
            sensitivity_map = {
                "gaming": 0.9,
                "voip": 0.85,
                "video_streaming": 0.8,
                "browsing": 0.3,
                "social": 0.4,
            }
            df["app_sensitivity_score"] = df["app_type"].map(sensitivity_map)

        return df

    def encode_categorical(
        self, df: pd.DataFrame, categorical_cols: List[str] = None, method: str = "onehot"
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Encode categorical features.

        Args:
            df: Input dataframe
            categorical_cols: List of categorical columns
            method: Encoding method ("onehot" or "label")

        Returns:
            Encoded dataframe and encoding mapping
        """
        df = df.copy()
        categorical_cols = categorical_cols or self.config.get("categorical_features", [])
        encoding_map = {}

        for col in categorical_cols:
            if col not in df.columns:
                continue

            if method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                encoding_map[col] = list(dummies.columns)
                df = df.drop(columns=[col])

            elif method == "label":
                # Create mapping
                unique_vals = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
                encoding_map[col] = mapping

        return df, encoding_map

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            df: Input dataframe
            strategy: Imputation strategy ("mean", "median", "drop")

        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def pipeline(
        self,
        df: pd.DataFrame,
        create_temporal: bool = True,
        create_interactions: bool = True,
        encode_cats: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.

        Args:
            df: Input raw dataframe
            create_temporal: Whether to create temporal features
            create_interactions: Whether to create interaction features
            encode_cats: Whether to encode categorical features

        Returns:
            Fully engineered feature dataframe
        """
        print("Running feature engineering pipeline...")

        if create_temporal and "timestamp" in df.columns:
            print("  - Creating temporal features")
            df = self.create_temporal_features(df)

        if create_interactions:
            print("  - Creating interaction features")
            df = self.create_interaction_features(df)

        if encode_cats:
            print("  - Encoding categorical features")
            df, _ = self.encode_categorical(df)

        print("  - Handling missing values")
        df = self.handle_missing_values(df)

        print(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def save_features(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save engineered features to processed data directory.

        Args:
            df: DataFrame to save
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


def main():
    """Run the QoE prediction feature engineering workflow."""
    # Load raw data
    raw_data_path = RAW_DATA_DIR / "synthetic_data.parquet"
    if not raw_data_path.exists():
        print(f"Error: {raw_data_path} not found. Run data_generator.py first.")
        return

    df = pd.read_parquet(raw_data_path)
    print(f"Loaded {len(df):,} rows from {raw_data_path}")

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.pipeline(df)

    # Save
    engineer.save_features(df_features, "engineered_features")
    print("\nFeature engineering complete!")


if __name__ == "__main__":
    main()
