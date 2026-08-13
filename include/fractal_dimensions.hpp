#pragma once
/**
 * Fractalich - Fractal-Based Data Analysis Library
 * Fractal Dimension Estimators
 * Version: 0.1.0
 *
 * Implements all 17 fractal dimension measures.
 */

#include <vector>
#include <complex>
#include <functional>

namespace fractalich {
namespace dimensions {

/**
 * 1. Box-Counting Dimension
 * Covers the set with boxes of decreasing size epsilon.
 * signal: 1D signal (for 1D box-counting).
 * min_box, max_box, num_scales: range of box sizes.
 * Returns: estimated fractal dimension.
 */
double box_counting(const std::vector<double>& signal,
                    int min_box = 2, int max_box = 64, int num_scales = 10);

/**
 * 2. Hausdorff Dimension (numerical estimate via box-counting on a point cloud)
 * points: N x D matrix flattened row-major (D dimensions).
 * D: point cloud dimensionality.
 */
double hausdorff(const std::vector<double>& points, int D = 2,
                 int min_box = 2, int max_box = 64, int num_scales = 10);

/**
 * 3. Correlation Dimension
 * Based on pairwise distances within a point cloud (Grassberger-Procaccia).
 * data: time series (will be embedded).
 * embedding_dim: embedding dimension m.
 * tau: time delay.
 */
double correlation_dim(const std::vector<double>& data,
                       int embedding_dim = 3, int tau = 1);

/**
 * 4. Information Dimension
 * Measures complexity via Shannon entropy of occupancy.
 * signal: 1D signal.
 * num_bins: histogram resolution.
 */
double information_dim(const std::vector<double>& signal,
                       int num_bins = 64);

/**
 * 5. Packing Dimension (approximation via dual box-packing scheme)
 */
double packing_dim(const std::vector<double>& signal,
                   int min_box = 2, int max_box = 64, int num_scales = 10);

/**
 * 6. Minkowski-Bouligand Dimension (equivalent to box-counting)
 */
double minkowski_bouligand(const std::vector<double>& signal,
                           int min_box = 2, int max_box = 64, int num_scales = 10);

/**
 * 7. Lyapunov Dimension (Kaplan-Yorke formula from sorted Lyapunov exponents)
 * lyapunov_exponents: sorted descending list of Lyapunov exponents.
 * Returns: Lyapunov dimension.
 */
double lyapunov_dim(const std::vector<double>& lyapunov_exponents);

/**
 * 8. Renyi Dimension D_q
 * Generalizes information dimension with parameter q.
 * q: order parameter (q=1 → information dim, q=2 → correlation dim).
 * signal: 1D signal.
 * num_bins: histogram resolution.
 */
double renyi_dim(const std::vector<double>& signal, double q,
                 int num_bins = 64);

/**
 * 9. Higuchi's Fractal Dimension
 * Uses curve lengths over different segment sizes.
 * signal: 1D time series.
 * k_max: maximum lag (typically 8–32).
 */
double higuchi(const std::vector<double>& signal, int k_max = 10);

/**
 * 10. Katz's Fractal Dimension
 * Waveform complexity estimator.
 * signal: 1D time series.
 */
double katz(const std::vector<double>& signal);

/**
 * 11. Petrosian Fractal Dimension
 * Signal-based; computed using zero crossings.
 * signal: 1D time series.
 */
double petrosian(const std::vector<double>& signal);

/**
 * 12. Detrended Fluctuation Analysis (DFA)
 * Detects long-range correlations; returns the Hurst-like DFA exponent.
 * signal: 1D time series.
 * scales: box sizes for DFA (e.g. {4,8,16,32,64}).
 */
double dfa(const std::vector<double>& signal,
           const std::vector<int>& scales = {});

/**
 * 13. Multifractal Spectrum f(alpha)
 * Returns (alpha_values, f_values) pairs describing the spectrum.
 * signal: 1D signal.
 * q_range: [q_min, q_max, num_q] to scan.
 * num_scales: number of box scales.
 */
struct MultifractalSpectrum { std::vector<double> alpha; std::vector<double> f; };
MultifractalSpectrum multifractal_spectrum(const std::vector<double>& signal,
                                           double q_min = -5.0, double q_max = 5.0,
                                           int num_q = 21, int num_scales = 10);

/**
 * 14. Wavelet-Based Fractal Dimension
 * Computes scaling via Haar wavelet energy across levels.
 * signal: 1D signal (length must be power-of-2, or will be zero-padded).
 * num_levels: number of wavelet decomposition levels.
 */
double wavelet_dim(const std::vector<double>& signal, int num_levels = 6);

/**
 * 15. Lacunarity
 * Quantifies texture gaps and spatial heterogeneity.
 * signal: 1D signal treated as occupancy.
 * box_size: sliding window size.
 */
double lacunarity(const std::vector<double>& signal, int box_size = 8);

/**
 * 16. Hurst Exponent (R/S method)
 * Indicates memory/persistence in a time series.
 * signal: 1D time series.
 * Returns: H in [0, 1]; H > 0.5 = persistent, H < 0.5 = anti-persistent.
 */
double hurst(const std::vector<double>& signal);

/**
 * 17. Temporal Fractal Dimension
 * Measures complexity across time; box-counting on the time-domain signal graph.
 * signal: 1D time series.
 * dt: time step between samples.
 */
double temporal_fractal_dim(const std::vector<double>& signal, double dt = 1.0);

} // namespace dimensions
} // namespace fractalich
