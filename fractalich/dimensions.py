"""
fractalich.dimensions
=====================
Python-level convenience wrapper for fractal dimension estimators.
"""

from . import fractalich_ext as _ext
import numpy as np


def box_counting(signal, min_box=2, max_box=64, num_scales=10):
    """Box-Counting Dimension. Use: Image texture, shape complexity."""
    return _ext.dimensions.box_counting(np.asarray(signal, np.float64), min_box, max_box, num_scales)

def hausdorff(points, D=2, min_box=2, max_box=64, num_scales=10):
    """Hausdorff Dimension (numerical). Use: Irregular geometries."""
    return _ext.dimensions.hausdorff(np.asarray(points, np.float64), D, min_box, max_box, num_scales)

def correlation_dim(data, embedding_dim=3, tau=1):
    """Correlation Dimension. Use: Chaos, EEG/heart-rate."""
    return _ext.dimensions.correlation_dim(np.asarray(data, np.float64), embedding_dim, tau)

def information_dim(signal, num_bins=64):
    """Information Dimension. Use: Network traffic, signal irregularity."""
    return _ext.dimensions.information_dim(np.asarray(signal, np.float64), num_bins)

def packing_dim(signal, min_box=2, max_box=64, num_scales=10):
    """Packing Dimension. Use: Sparse/porous materials."""
    return _ext.dimensions.packing_dim(np.asarray(signal, np.float64), min_box, max_box, num_scales)

def minkowski_bouligand(signal, min_box=2, max_box=64, num_scales=10):
    """Minkowski-Bouligand Dimension. Use: Digital pathology, coastlines."""
    return _ext.dimensions.minkowski_bouligand(np.asarray(signal, np.float64), min_box, max_box, num_scales)

def lyapunov_dim(exponents):
    """Lyapunov Dimension (Kaplan-Yorke). Use: Nonlinear dynamical systems."""
    return _ext.dimensions.lyapunov_dim(np.asarray(exponents, np.float64))

def renyi_dim(signal, q=2.0, num_bins=64):
    """Renyi Dimension D_q. Use: Financial time series, multifractals."""
    return _ext.dimensions.renyi_dim(np.asarray(signal, np.float64), q, num_bins)

def higuchi(signal, k_max=10):
    """Higuchi Fractal Dimension. Use: EEG/ECG biomedical analysis."""
    return _ext.dimensions.higuchi(np.asarray(signal, np.float64), k_max)

def katz(signal):
    """Katz Fractal Dimension. Use: Real-time EMG monitoring."""
    return _ext.dimensions.katz(np.asarray(signal, np.float64))

def petrosian(signal):
    """Petrosian Fractal Dimension. Use: Brain activity, mental disorder diagnosis."""
    return _ext.dimensions.petrosian(np.asarray(signal, np.float64))

def dfa(signal, scales=None):
    """DFA Hurst exponent. Use: Heart rate variability, climate."""
    return _ext.dimensions.dfa(np.asarray(signal, np.float64), scales or [])

def multifractal_spectrum(signal, q_min=-5.0, q_max=5.0, num_q=21, num_scales=10):
    """Multifractal Spectrum f(alpha). Use: Financial markets, earth sciences.
    
    Returns:
        dict with 'alpha' and 'f' keys (lists).
    """
    return _ext.dimensions.multifractal_spectrum(
        np.asarray(signal, np.float64), q_min, q_max, num_q, num_scales)

def wavelet_dim(signal, num_levels=6):
    """Wavelet-Based Fractal Dimension (Haar). Use: Image compression, remote sensing."""
    return _ext.dimensions.wavelet_dim(np.asarray(signal, np.float64), num_levels)

def lacunarity(signal, box_size=8):
    """Lacunarity. Use: Soil analysis, tumor heterogeneity."""
    return _ext.dimensions.lacunarity(np.asarray(signal, np.float64), box_size)

def hurst(signal):
    """Hurst Exponent (R/S method). Use: Stock trends, hydrology."""
    return _ext.dimensions.hurst(np.asarray(signal, np.float64))

def temporal_fractal_dim(signal, dt=1.0):
    """Temporal Fractal Dimension. Use: Weather, traffic time series."""
    return _ext.dimensions.temporal_fractal_dim(np.asarray(signal, np.float64), dt)
