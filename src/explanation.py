import numpy as np

def explain_anomaly(idx, residual, scores, series):
    explanation = {}

    z = scores[idx]
    explanation["residual"] = float(residual[idx])
    explanation["z_score"] = float(z)

    if idx > 0:
        prev = series[idx - 1]
        curr = series[idx]
        explanation["pct_change"] = float(100 * (curr - prev) / prev)
    else:
        explanation["pct_change"] = 0.0

    # Tip sınıflaması
    if abs(z) >= 6:
        explanation["type"] = "extreme_spike"
    else:
        explanation["type"] = "moderate_deviation"

    # Güven puanı (0–1)
    explanation["confidence"] = min(1.0, abs(z) / 10.0)

    return explanation
