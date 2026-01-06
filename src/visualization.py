import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(series, repaired, anomalies, explanations):
    plt.figure(figsize=(14,6))
    plt.plot(series, label="Raw", alpha=0.6)
    plt.plot(repaired, label="Repaired", linewidth=2)

    ext = [i for i, e in explanations.items() if e["type"] == "extreme_spike"]
    mod = [i for i, e in explanations.items() if e["type"] == "moderate_deviation"]

    plt.scatter(ext, series[ext], color="red", s=20, label="Extreme spikes")
    plt.scatter(mod, series[mod], color="orange", s=20, label="Moderate deviations")

    plt.legend()
    plt.title("Raw vs Repaired Time Series (Type-aware Repair)")
    plt.show()
