from load_data import load_series
from decomposition import stl_decompose
from detection import detect_anomalies
from explanation import explain_anomaly
from repair import repair_series
from visualization import plot_comparison

import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load data
series = load_series("data/yahoo.csv")

# 2️⃣ STL decomposition
trend, seasonal, resid = stl_decompose(series, period=5)

plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(series)
plt.title("Adj Close")

plt.subplot(4,1,2)
plt.plot(trend)
plt.title("Trend")

plt.subplot(4,1,3)
plt.plot(seasonal)
plt.title("Seasonal")

plt.subplot(4,1,4)
plt.plot(resid)
plt.title("Residual")

plt.tight_layout()
plt.show()

# 3️⃣ Detect anomalies
anomalies, scores = detect_anomalies(resid)
print("Toplam anomali sayısı:", anomalies.sum())

# 4️⃣ Explain anomalies
idxs = np.where(anomalies)[0]

print("İlk 5 anomali açıklaması:")
explanations = {}
for i in idxs:
    explanations[i] = explain_anomaly(i, resid, scores, series)

for i in idxs[:5]:
    print(i, explanations[i])

# 5️⃣ Repair (TYPE-AWARE)
repaired = repair_series(series, trend, seasonal, anomalies, explanations)

# 6️⃣ Visualization
plot_comparison(series, repaired, anomalies, explanations)
print(
    "Extreme spikes:",
    sum(1 for e in explanations.values() if e["type"] == "extreme_spike")
)
print(
    "Moderate deviations:",
    sum(1 for e in explanations.values() if e["type"] == "moderate_deviation")
)
