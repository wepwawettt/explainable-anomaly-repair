import numpy as np

def detect_anomalies(residual, k=3.5):
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
