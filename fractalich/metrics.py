"""
fractalich.metrics
==================
Python-level convenience wrapper (delegates to C++ backend).

All functions accept numpy arrays.
"""

from . import fractalich_ext as _ext
import numpy as np


def ffs(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Fractal Fidelity Score (FFS).

    Measures how accurately a fractal reconstruction retains the original signal.

    Args:
        original (ndarray): 1-D original signal (float64).
        reconstructed (ndarray): 1-D fractal-reconstructed signal (float64).

    Returns:
        float: FFS in (-inf, 1]. 1.0 means perfect reconstruction.

    Example:
        >>> import numpy as np
        >>> from fractalich import metrics
        >>> x = np.random.rand(1000)
        >>> metrics.ffs(x, x)   # 1.0
        1.0
    """
    return _ext.metrics.ffs(np.asarray(original, dtype=np.float64),
                             np.asarray(reconstructed, dtype=np.float64))


def msc(substructures_orig, substructures_rec) -> float:
    """Multiscale Structural Consistency (MSC).

    Args:
        substructures_orig (list of ndarray): Original signal at each scale.
        substructures_rec  (list of ndarray): Reconstructed signal at each scale.

    Returns:
        float: Mean SSIM across all scales.
    """
    orig = [np.asarray(s, dtype=np.float64) for s in substructures_orig]
    rec  = [np.asarray(s, dtype=np.float64) for s in substructures_rec]
    return _ext.metrics.msc(orig, rec)


def fre(measured: np.ndarray, predicted: np.ndarray) -> float:
    """Fractal Residual Error (FRE).

    Args:
        measured  (ndarray): Empirically measured fractal dimensions.
        predicted (ndarray): Model-predicted fractal dimensions.

    Returns:
        float: Total absolute residual error.
    """
    return _ext.metrics.fre(np.asarray(measured, dtype=np.float64),
                             np.asarray(predicted, dtype=np.float64))


def ffri(feature: np.ndarray, labels: np.ndarray, num_bins: int = 50) -> float:
    """Fractal Feature Relevance Index (FFRI).

    Args:
        feature  (ndarray): 1-D fractal feature vector.
        labels   (ndarray): 1-D integer class labels.
        num_bins (int):     Histogram resolution for MI estimation.

    Returns:
        float: FFRI score.
    """
    return _ext.metrics.ffri(np.asarray(feature, dtype=np.float64),
                              np.asarray(labels, dtype=np.int32),
                              num_bins)


def ssri(rfd_values: np.ndarray) -> float:
    """Self-Similarity Retention Index (SSRI).

    Args:
        rfd_values (ndarray): Relative Fractal Dimension values at each scale.

    Returns:
        float: SSRI score.
    """
    return _ext.metrics.ssri(np.asarray(rfd_values, dtype=np.float64))


def fri(match_scores: np.ndarray, theta: float = 0.5) -> float:
    """Fractal Recurrence Index (FRI).

    Args:
        match_scores (ndarray): FractalMatch scores at each time step.
        theta        (float):   Recurrence threshold.

    Returns:
        float: FRI in [0, 1].
    """
    return _ext.metrics.fri(np.asarray(match_scores, dtype=np.float64), theta)
