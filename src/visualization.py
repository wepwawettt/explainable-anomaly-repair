import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(series, corrupted, repaired, explanations, drift_flags):
    plt.figure(figsize=(15,6))

    # 🔵 Drift bölgeleri (arka plan)
    in_drift = False
    start = 0
    for i in range(len(drift_flags)):
        if drift_flags[i] and not in_drift:
            start = i
            in_drift = True
        elif not drift_flags[i] and in_drift:
            plt.axvspan(start, i, color="blue", alpha=0.08)
            in_drift = False
    if in_drift:
        plt.axvspan(start, len(drift_flags), color="blue", alpha=0.08)

    # 📉 Corrupted (arka plan, silik)
    plt.plot(corrupted, label="Corrupted", alpha=0.15, linewidth=1, zorder=1)

    # 📈 Original (referans)
    plt.plot(series, label="Original", linewidth=1.5, color="green", zorder=3)
    # Repaired (drift dışında)
    masked_repaired = repaired.copy()
    masked_repaired[drift_flags] = np.nan

    plt.plot(masked_repaired, label="Repaired (stable)", linewidth=2, zorder=3)

    # 🛠 Repaired → SADECE DEĞİŞEN NOKTALAR
    eps = 1e-6
    repaired_idx = np.where(np.abs(repaired - corrupted) > eps)[0]


    plt.scatter(
    repaired_idx,
    repaired[repaired_idx],
    color="orange",
    edgecolor="black",
    linewidth=0.5,
    s=35,
    label="Repaired points",
    zorder=7
)


    # 🔴 Spike anomalileri
    normal_spikes = []
    drift_spikes = []

    for i, e in explanations.items():
        if e["type"] == "spike":
            if e.get("in_drift", False):
                drift_spikes.append(i)
            else:
                normal_spikes.append(i)

    plt.scatter(
        normal_spikes,
        corrupted[normal_spikes],
        color="red",
        s=25,
        label="Spike anomalies",
        zorder=6
    )

    plt.scatter(
        drift_spikes,
        corrupted[drift_spikes],
        color="darkorange",
        s=30,
        label="Spike in drift",
        zorder=6
    )

    plt.legend()
    plt.title("Type-aware Anomaly Detection and Repair")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
