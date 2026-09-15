"""Metrics for TMI data fit and magnetic-susceptibility recovery."""
from __future__ import annotations
import numpy as np

def _finite_pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape: raise ValueError("Arrays must have matching shapes.")
    if reference.size == 0 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Arrays must be nonempty and finite.")
    return reference, prediction

def calculate_tmi_fit_metrics(observed_tmi_nt: np.ndarray, predicted_tmi_nt: np.ndarray) -> dict[str, float]:
    """Return signed TMI residual metrics in nT."""
    observed, predicted = _finite_pair(observed_tmi_nt, predicted_tmi_nt)
    residual = predicted - observed
    norm = np.linalg.norm(observed)
    correlation = float("nan") if np.std(observed) == 0 or np.std(predicted) == 0 else float(np.corrcoef(observed.ravel(), predicted.ravel())[0,1])
    return {"tmi_mae_nt": float(np.mean(np.abs(residual))), "tmi_rmse_nt": float(np.sqrt(np.mean(residual**2))),
            "tmi_relative_l2": float(np.linalg.norm(residual)/norm) if norm else float("nan"), "tmi_correlation": correlation}

def calculate_susceptibility_metrics(true_si: np.ndarray, recovered_si: np.ndarray, threshold_si: float = 0.0) -> dict[str, float | int]:
    """Return susceptibility error and occupied-body overlap metrics."""
    true, recovered = _finite_pair(true_si, recovered_si)
    if threshold_si < 0: raise ValueError("threshold_si cannot be negative.")
    residual = recovered - true; true_mask = np.abs(true) > threshold_si; recovered_mask = np.abs(recovered) > threshold_si
    intersection = np.count_nonzero(true_mask & recovered_mask); union = np.count_nonzero(true_mask | recovered_mask)
    return {"susceptibility_mae_si": float(np.mean(np.abs(residual))), "susceptibility_rmse_si": float(np.sqrt(np.mean(residual**2))),
            "susceptibility_relative_l2": float(np.linalg.norm(residual)/np.linalg.norm(true)) if np.linalg.norm(true) else float("nan"),
            "support_iou": float(intersection/union) if union else 1.0, "true_occupied_cells": int(np.count_nonzero(true_mask)),
            "recovered_occupied_cells": int(np.count_nonzero(recovered_mask))}
