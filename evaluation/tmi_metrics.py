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


def calculate_support_metrics(true_mask: np.ndarray, predicted_mask: np.ndarray) -> dict[str, float | int | None]:
    """Binary support scores. Both empty is perfect; undefined volume ratio is null."""
    truth = np.asarray(true_mask, dtype=bool)
    predicted = np.asarray(predicted_mask, dtype=bool)
    if truth.shape != predicted.shape or not truth.size:
        raise ValueError("Support masks must have the same nonempty shape.")
    n_true = int(np.count_nonzero(truth))
    n_pred = int(np.count_nonzero(predicted))
    tp = int(np.count_nonzero(truth & predicted))
    union = n_true + n_pred - tp
    both_empty = n_true == n_pred == 0
    return {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (n_true + n_pred) if n_true + n_pred else 1.0,
        "precision": tp / n_pred if n_pred else (1.0 if both_empty else 0.0),
        "recall": tp / n_true if n_true else (1.0 if both_empty else 0.0),
        "true_occupied_cells": n_true,
        "predicted_occupied_cells": n_pred,
        "predicted_to_true_volume_ratio": n_pred / n_true if n_true else (1.0 if both_empty else None),
    }


def calculate_e02_support_metrics(
    true_si: np.ndarray, predicted_si: np.ndarray, *,
    geological_mask: np.ndarray | None = None,
    prediction_threshold_si: float = 0.00005,
    diagnostic_threshold_si: float = 0.001,
) -> dict[str, dict[str, float | int | None]]:
    """Keep geological support separate from symmetric SI-threshold support.

    Prediction and diagnostic thresholds use >=; exact-zero synthetic truth
    uses >0 when no generated body mask is supplied.
    """
    truth, prediction = _finite_pair(true_si, predicted_si)
    if prediction_threshold_si <= 0 or diagnostic_threshold_si <= 0:
        raise ValueError("Support thresholds must be positive SI values.")
    body = np.asarray(geological_mask, dtype=bool) if geological_mask is not None else truth > 0
    if body.shape != truth.shape:
        raise ValueError("Geological mask shape does not match susceptibility.")
    return {
        "geological_support": calculate_support_metrics(body, prediction >= prediction_threshold_si),
        "thresholded_susceptibility_support": calculate_support_metrics(
            truth >= diagnostic_threshold_si, prediction >= diagnostic_threshold_si
        ),
    }
