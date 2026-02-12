# Telecom QoE Prediction

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Business Context

Quality of Experience (QoE) measured as MOS (1-5) directly predicts customer satisfaction and churn risk. Proactively predicting QoE from network KPIs enables real-time service quality management.

## Problem Framing

Regression using LightGBM.

- **Target:** `mos_score`
- **Primary Metric:** RMSE
- **Challenges:**
  - Non-linear MOS curve (diminishing returns at high quality)
  - App-specific sensitivity (gaming vs browsing)
  - Device capability effects on perceived quality

## Data Engineering

Session-level data (10K sessions) combining:

- **Network KPIs** -- SINR, throughput, latency, packet loss
- **Device class** -- high-end, mid-range, low-end capability tiers
- **App type** -- gaming, video streaming, browsing, VoIP

Domain physics: MOS computed using an ITU-T inspired formula with app-specific adjustments. Gaming sessions are highly sensitive to latency, video to throughput and packet loss, browsing to page load time proxies.

## Methodology

- LightGBM regression with early stopping on validation RMSE
- **Feature groups:**
  - Network quality index (composite SINR + throughput + latency score)
  - Service degradation indicators (packet loss rate, jitter)
  - Bandwidth utilization (throughput vs expected capacity)
  - App sensitivity score (app-specific quality weight)
- Hyperparameter tuning with cross-validation

## Key Findings

- **RMSE:** 0.04, **R²:** 0.99 on held-out test set (MOS scale 1-5)
- **Top predictors:** `service_degradation`, `throughput_mbps`, and `latency_ms` dominate SHAP importance
- Gaming sessions show highest RMSE (0.058) due to latency sensitivity; browsing is most predictable (RMSE 0.028)
- Device class acts as a ceiling effect: low-end devices cap MOS regardless of network quality

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adityonugrohoid/telecom-ml-portfolio.git
cd telecom-ml-portfolio/04-qoe-prediction

# Install dependencies
uv sync

# Generate synthetic data
uv run python -m qoe_prediction.data_generator

# Run the notebook
uv run jupyter lab notebooks/
```

## Project Structure

```
04-qoe-prediction/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── 04_qoe_prediction.ipynb
├── src/
│   └── qoe_prediction/
│       ├── __init__.py
│       ├── data_generator.py
│       ├── features.py
│       ├── model.py
│       └── evaluate.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── tests/
    └── .gitkeep
```

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [Churn Prediction](../01-churn-prediction) | Binary classification to predict customer churn |
| 2 | [Root Cause Analysis](../02-root-cause-analysis) | Multi-class classification for network alarm RCA |
| 3 | [Anomaly Detection](../03-anomaly-detection) | Unsupervised detection of network anomalies |
| 4 | **QoE Prediction** (this repo) | Regression to predict quality of experience |
| 5 | [Capacity Forecasting](../05-capacity-forecasting) | Time-series forecasting for network capacity planning |
| 6 | [Network Optimization](../06-network-optimization) | Optimization of network resource allocation |

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Author

**Adityo Nugroho**
