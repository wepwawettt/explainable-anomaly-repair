def repair_series(series, trend, seasonal, anomalies, explanations):
    repaired = series.copy()

    for i in range(len(series)):
        if anomalies[i]:
            info = explanations.get(i, {})
            if info.get("type") == "extreme_spike":
                repaired[i] = trend[i] + seasonal[i]
            # moderate_deviation -> DOKUNMA

    return repaired
