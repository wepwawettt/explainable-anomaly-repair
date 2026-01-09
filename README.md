# Type-Aware Anomaly Detection and Safe Repair for Time Series

This project presents a **type-aware, drift-sensitive anomaly detection and repair pipeline** for univariate time series.  
Unlike naive anomaly correction methods, the proposed approach distinguishes between **spike anomalies** and **concept drift**, and applies **confidence-gated repairs** to guarantee safety.

---

## 🚀 Key Features

- Robust STL-based decomposition (trend / seasonal / residual)
- MAD-based spike anomaly detection with high recall
- Slope-aware drift detection using sustained trend changes
- Type-aware anomaly classification:
  - Spike
  - Spike-in-drift
  - Moderate deviation
- Confidence-weighted repair mechanism
- Repairs applied **only to safe spike anomalies**
- Drift regions explicitly excluded from repair
- Explainable anomaly decisions:
  - z-score
  - confidence
  - drift-awareness per anomaly
- Safety guarantees:
  - Zero negative-impact repairs observed
  - Conservative repair policy inside drift

---

## 🧠 Method Overview

1. STL decomposition is applied to isolate residual anomalies.
2. Spike anomalies are detected using a robust **MAD-based z-score**.
3. Concept drift is identified via **sustained slope changes** in the trend.
4. Each anomaly is explained and classified based on:
   - Magnitude
   - Temporal context
   - Drift awareness
5. Repairs are applied **only when safe**, using:
   - Trend + seasonal reconstruction
   - Confidence-weighted blending
6. Drift regions are explicitly protected from forced corrections.

---

## 📊 Quantitative Results (Synthetic Injection)

| Metric                    | Value   |
|---------------------------|---------|
| Precision                 | 0.319   |
| Recall                    | 0.759   |
| RMSE (before repair)      | 305.43  |
| RMSE (after repair)       | 177.12  |
| RMSE (drift excluded)     | 78.07   |
| Avg repair improvement    | 97.8%   |
| Repair safety rate        | 100%    |
| Spike repair coverage     | 100%    |

> Detection precision is intentionally relaxed to maximize recall.  
> The repair module remains **conservative and confidence-gated**, ensuring **zero negative impact**.

---

## 🔬 Notes

- The system prioritizes **repair safety over aggressive correction**.
- Drift detection acts as a **hard protection boundary**.
- All repairs are fully explainable and auditable.

---

## 📄 License

For research and academic use.
