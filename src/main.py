from statsmodels.tsa.seasonal import STL
import numpy as np
from load_data import load_series
from visualization import plot_comparison


def stl_decompose(series, period):
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

def detect_anomalies(residual, k=5.0):
    residual = np.asarray(residual)

    median = np.median(residual)
    mad = np.median(np.abs(residual - median))

    # Güvenli MAD
    if mad < 1e-9:
        anomalies = np.zeros_like(residual, dtype=bool)
        scores = np.zeros_like(residual)
        return anomalies, scores

    modified_z = 0.6745 * (residual - median) / mad
    anomalies = np.abs(modified_z) > k

    return anomalies, modified_z

def detect_drift_from_trend(trend, window=25, z_thresh=3.0, min_run=15):
    """
    Drift detection using sustained change in trend slope.
    - trend: STL trend component
    - window: rolling window for slope smoothing
    - z_thresh: threshold on robust z-score of slope
    - min_run: minimum consecutive points to call it drift
    """
    trend = np.asarray(trend, dtype=float)

    # slope (first derivative)
    slope = np.diff(trend, prepend=trend[0])

    # robust z-score for slope
    med = np.median(slope)
    mad = np.median(np.abs(slope - med))
    if mad < 1e-9:
        return np.zeros_like(trend, dtype=bool)

    z = 0.6745 * (slope - med) / mad

    # smooth using rolling mean of |z|
    absz = np.abs(z)
    kernel = np.ones(window) / window
    smooth = np.convolve(absz, kernel, mode="same")

    raw_flags = smooth > z_thresh

    # enforce "sustained" drift via min_run
    drift_flags = np.zeros_like(raw_flags, dtype=bool)
    run = 0
    for i in range(len(raw_flags)):
        if raw_flags[i]:
            run += 1
        else:
            if run >= min_run:
                drift_flags[i-run:i] = True
            run = 0
    if run >= min_run:
        drift_flags[len(raw_flags)-run:] = True

    return drift_flags

def expand_drift_region(drift_flags, radius=5):
    expanded = drift_flags.copy()
    n = len(drift_flags)

    for i in range(n):
        if drift_flags[i]:
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
            expanded[start:end] = True

    return expanded


def explain_anomaly(idx, residual, scores, series, drift_flags):
    z = float(scores[idx])
    in_drift = bool(drift_flags[idx])

    confidence = compute_repair_confidence(z, in_drift)

    explanation = {
        "residual": float(residual[idx]),
        "z_score": z,
        "confidence": confidence,
        "in_drift": in_drift,
    }

    if idx > 0 and series[idx-1] != 0:
        explanation["pct_change"] = float(
            100 * (series[idx] - series[idx-1]) / series[idx-1]
        )
    else:
        explanation["pct_change"] = 0.0

    # anomaly type
    if abs(z) >= 6 and in_drift:
        explanation["type"] = "spike_in_drift"
    elif abs(z) >= 6:
        explanation["type"] = "spike"
    else:
        explanation["type"] = "moderate_deviation"

    return explanation

def interpolate_from_neighbors(x, i, good_mask):
    n = len(x)

    # sol
    l = i - 1
    while l >= 0 and not good_mask[l]:
        l -= 1

    # sağ
    r = i + 1
    while r < n and not good_mask[r]:
        r += 1

    if l >= 0 and r < n:
        # linear interpolation
        return x[l] + (x[r] - x[l]) * (i - l) / (r - l)
    elif l >= 0:
        return x[l]
    elif r < n:
        return x[r]
    else:
        return x[i]  # fallback
def inject_drift(series, start, length, slope):
    """
    Adds a gradual drift to the series.
    start  : drift start index
    length : drift duration
    slope  : per-step change
    """
    series = series.copy()
    end = min(len(series), start + length)

    for i in range(start, end):
        series[i] += slope * (i - start)

    return series
def local_robust_repair(x, i, anomalies, half_window=7):
    n = len(x)
    l = max(0, i-half_window)
    r = min(n, i+half_window+1)

    window_idx = [j for j in range(l, r) if not anomalies[j] and j != i]
    if len(window_idx) >= 3:
        return float(np.median(x[window_idx]))
    return x[i]
  
def repair_series(
    corrupted,
    trend,
    seasonal,
    anomalies,
    explanations,
    drift_flags
):
    repaired = corrupted.copy()

    for i in np.where(anomalies)[0]:
        info = explanations[i]
        t = info["type"]
        alpha = info["confidence"]

        # 🚫 Drift içinde ASLA çizgisel repair yapma
        if t == "spike_in_drift":
            continue

        if t == "spike":
            candidate = trend[i] + seasonal[i]
            repaired[i] = alpha * candidate + (1 - alpha) * corrupted[i]

        # moderate_deviation -> dokunma

    return repaired

def inject_anomalies(
    series,
    spike_rate=0.01,
    spike_scale=5.0,
    random_state=42
):
    """
    Seriye sentetik ani spike anomalileri ekler.

    Returns:
        corrupted_series : anomaly eklenmiş seri
        anomaly_indices  : gerçek anomaly indexleri
    """
    rng = np.random.default_rng(random_state)

    series = np.asarray(series, dtype=float)
    corrupted = series.copy()

    n = len(series)
    num_spikes = int(n * spike_rate)

    anomaly_indices = rng.choice(n, size=num_spikes, replace=False)

    std = np.std(series)

    for idx in anomaly_indices:
        corrupted[idx] += spike_scale * std * rng.choice([-1, 1])

    return corrupted, anomaly_indices
def compute_repair_confidence(z_score, in_drift):
    """
    Returns confidence in [0,1] for repair decision
    """
    # z-score based confidence
    conf = min(1.0, abs(z_score) / 10.0)

    # drift içinde güven düşür
    if in_drift:
        conf *= 0.4

    return float(conf)
def repair_impact_report(original, corrupted, repaired, explanations):
    improvements = []
    negative = 0

    for i, e in explanations.items():
        if e["type"] != "spike":

            continue

        err_before = abs(original[i] - corrupted[i])
        err_after  = abs(original[i] - repaired[i])

        if err_before > 1e-6:
            improvement = 100 * (err_before - err_after) / err_before
            improvements.append(improvement)
            if improvement < 0:
                negative += 1

    if not improvements:
        return

    total_repaired = len(improvements)
    safety_ratio = 1.0 - (negative / total_repaired)

    print("\n--- Repair Impact Report ---")
    print(f"Total repaired points : {total_repaired}")
    print(f"Avg improvement (%)   : {np.mean(improvements):.2f}")
    print(f"Median improvement    : {np.median(improvements):.2f}")
    print(f"Negative impact cases : {negative}")
    print(f"Repair safety rate    : {100 * safety_ratio:.1f}%")


series = load_series("data/yahoo.csv")

# ===============================
# PIPELINE (FINAL)
# ===============================

# 1️⃣ spike inject
corrupted_series, true_anomalies = inject_anomalies(
    series,
    spike_rate=0.02,
    spike_scale=6.0
)

# 2️⃣ drift inject
corrupted_series = inject_drift(
    corrupted_series,
    start=900,
    length=200,
    slope=2.5
)

# 2️⃣ STL decomposition
trend, seasonal, resid = stl_decompose(corrupted_series, period=5)

# 3️⃣ Detect anomalies
anomalies, scores = detect_anomalies(resid)
drift_flags = detect_drift_from_trend(trend, window=25, z_thresh=3.0, min_run=15)
expanded_drift_flags = expand_drift_region(drift_flags, radius=30)


# 4️⃣ Explain detected anomalies
explanations = {}
for idx in np.where(anomalies)[0]:
    explanations[idx] = explain_anomaly(
    idx,
    resid,
    scores,
    corrupted_series,
    expanded_drift_flags
)


# 5️⃣ Repair series
repaired = repair_series(
    corrupted_series,
    trend,
    seasonal,
    anomalies,
    explanations,
    expanded_drift_flags
)


repair_impact_report(
    series,
    corrupted_series,
    repaired,
    explanations
)
# 6️⃣ Detection evaluation
# sadece spike evaluation (drift hariç)
detected_spikes = {
    i for i in np.where(anomalies)[0]
    if explanations[i]["type"] == "spike"
}

true_spikes = set(true_anomalies)

tp = len(detected_spikes & true_spikes)
fp = len(detected_spikes - true_spikes)
fn = len(true_spikes - detected_spikes)


precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

rmse_before = np.sqrt(np.mean((series - corrupted_series) ** 2))
rmse_after  = np.sqrt(np.mean((series - repaired) ** 2))


print(f"RMSE before repair: {rmse_before:.4f}")
print(f"RMSE after repair : {rmse_after:.4f}")
# Drift-excluded RMSE
mask = ~expanded_drift_flags

rmse_spike_only = np.sqrt(
    np.mean((series[mask] - repaired[mask]) ** 2)
)

print(f"RMSE (drift excluded): {rmse_spike_only:.4f}")

# ===============================
# NUMERICAL ANALYSIS (NEW)
# ===============================

total_anomalies = int(np.sum(anomalies))
def count_drift_segments(drift_flags):
    count = 0
    in_drift = False
    for f in drift_flags:
        if f and not in_drift:
            count += 1
            in_drift = True
        elif not f:
            in_drift = False
    return count
def export_repair_log(original, corrupted, repaired, explanations):
    logs = []

    for i, e in explanations.items():
        if e["type"] != "spike":
            continue
        if abs(original[i] - corrupted[i]) < 1e-6:
            continue   # gerçek spike değil → rapora girme


        log = {
            "index": i,
            "type": e["type"],
            "z_score": round(e["z_score"], 3),
            "confidence": round(e["confidence"], 3),
            "in_drift": e["in_drift"],
            "error_before": abs(original[i] - corrupted[i]),
            "error_after": abs(original[i] - repaired[i]),
        }

        if log["error_before"] > 1e-6:
            log["improvement_pct"] = round(
                100 * (log["error_before"] - log["error_after"]) / log["error_before"],
                2
            )
        else:
            log["improvement_pct"] = 0.0

        logs.append(log)

    return logs
repair_logs = export_repair_log(
    series,
    corrupted_series,
    repaired,
    explanations
)
total_spikes = sum(1 for e in explanations.values() if e["type"] == "spike")
repaired_spikes = len([
    i for i, e in explanations.items()
    if e["type"] == "spike" and repaired[i] != corrupted_series[i]
])

print(f"Repair coverage (spikes): {100 * repaired_spikes / total_spikes:.1f}%")

print("\nSample repair log:")
for r in repair_logs[:5]:
    print(r)

def estimate_slope(x):
    t = np.arange(len(x))
    return np.polyfit(t, x, 1)[0]
print("\n--- Drift Slope Comparison ---")

in_drift = False
start = 0

for i, f in enumerate(expanded_drift_flags):
    if f and not in_drift:
        start = i
        in_drift = True
    elif not f and in_drift:
        end = i
        orig_slope = estimate_slope(series[start:end])
        rep_slope = estimate_slope(repaired[start:end])
        cor_slope = estimate_slope(corrupted_series[start:end])


        print(f"Drift [{start}-{end}] | original slope: {orig_slope:.4f}, repaired slope: {rep_slope:.4f}")
        in_drift = False

# drift en sonda bitiyorsa
if in_drift:
    end = len(series)
    orig_slope = estimate_slope(series[start:end])
    rep_slope = estimate_slope(repaired[start:end])
    cor_slope = estimate_slope(corrupted_series[start:end])


    print(
  f"Drift [{start}-{end}] | "
  f"original: {orig_slope:.4f}, "
  f"corrupted: {cor_slope:.4f}, "
  f"repaired: {rep_slope:.4f}"
)



drift_count = count_drift_segments(expanded_drift_flags)

spike_count = 0
for info in explanations.values():
    if info.get("type") == "spike":
        spike_count += 1

repaired_points = spike_count  # sadece spike'lar onarılıyor

print("\n--- Anomaly Breakdown ---")
print(f"Total detected anomalies : {total_anomalies}")
print(f"Spike anomalies          : {spike_count}")
print(f"Drift anomalies          : {drift_count}")
print(f"Repaired points          : {repaired_points}")

if total_anomalies > 0:
    print(f"Spike ratio              : {spike_count / total_anomalies:.2f}")
    print(f"Drift ratio              : {drift_count / total_anomalies:.2f}")

conf_values = [
    e["confidence"] for e in explanations.values()
    if e["type"] in ("spike", "spike_in_drift")
]

print(f"Avg repair confidence: {np.mean(conf_values):.3f}")
print(f"Min repair confidence: {np.min(conf_values):.3f}")
print(f"Max repair confidence: {np.max(conf_values):.3f}")


# ===============================
# VISUALIZATION
# ===============================
plot_comparison(
    series,
    corrupted_series,
    repaired,
    explanations,
    expanded_drift_flags
)
