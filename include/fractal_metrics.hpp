#pragma once
/**
 * Fractalich - Fractal-Based Data Analysis Library
 * Evaluation Metrics
 * Version: 0.1.0
 */

#include <vector>
#include <functional>
#include <stdexcept>

namespace fractalich {
namespace metrics {

/**
 * Fractal Fidelity Score (FFS)
 * FFS = 1 - ||Xf - Xo||_2 / ||Xo||_2
 * Measures how accurately a fractal reconstruction retains the original signal.
 * Returns: score in [0, 1], where 1 = perfect reconstruction.
 */
double ffs(const std::vector<double>& original,
           const std::vector<double>& reconstructed);

/**
 * Multiscale Structural Consistency (MSC)
 * MSC = (1/n) * sum_i SSIM(Si, S'i)
 * Measures structural similarity across multiple scales.
 * substructures_orig and substructures_rec: vectors of substructure signals at each scale.
 * Returns: mean SSIM across all scales [0, 1].
 */
double msc(const std::vector<std::vector<double>>& substructures_orig,
           const std::vector<std::vector<double>>& substructures_rec);

/**
 * Fractal Residual Error (FRE)
 * FRE = sum_i |Di - D'i|
 * Quantifies prediction error of fractal models vs. empirical fractal properties.
 * measured: empirically measured fractal dimensions.
 * predicted: model-predicted fractal dimensions.
 * Returns: total absolute residual error.
 */
double fre(const std::vector<double>& measured,
           const std::vector<double>& predicted);

/**
 * Fractal Feature Relevance Index (FFRI)
 * FFRI = MI(f, y) / H(f)
 * Ranks features by their fractal relevance in supervised tasks.
 * feature: fractal feature vector.
 * labels:  class labels (integer-coded).
 * num_bins: number of bins for histogram-based entropy/MI estimation.
 * Returns: FFRI score; higher = more relevant.
 */
double ffri(const std::vector<double>& feature,
            const std::vector<int>&    labels,
            int                        num_bins = 50);

/**
 * Self-Similarity Retention Index (SSRI)
 * SSRI = 1 - Var(RFD(s)) / mean(RFD(s))
 * Tracks preservation of self-similarity across scales.
 * rfd_values: Relative Fractal Dimension at each scale s.
 * Returns: SSRI score; higher = better self-similarity preservation.
 */
double ssri(const std::vector<double>& rfd_values);

/**
 * Fractal Recurrence Index (FRI)
 * FRI = (1/T) * sum_{t=1}^{T} 1{ FractalMatch(t, t+tau) > theta }
 * Measures how frequently fractal patterns repeat over time or space.
 * match_scores: FractalMatch scores at each time step.
 * tau:          lag offset (for documentation; already baked into match_scores).
 * theta:        recurrence threshold.
 * Returns: FRI in [0, 1].
 */
double fri(const std::vector<double>& match_scores, double theta);

// ─── Internal helper ──────────────────────────────────────────────────────────
namespace detail {
    double l2_norm(const std::vector<double>& v);
    double ssim_1d(const std::vector<double>& a, const std::vector<double>& b);
    double entropy(const std::vector<double>& values, int num_bins);
    double mutual_information(const std::vector<double>& values,
                              const std::vector<int>&    labels,
                              int num_bins);
}

} // namespace metrics
} // namespace fractalich
