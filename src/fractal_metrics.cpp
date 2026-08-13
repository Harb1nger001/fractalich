/**
 * Fractalich - fractal_metrics.cpp
 * Implementation of all 6 fractal evaluation metrics.
 */

#include "fractal_metrics.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace fractalich {
namespace metrics {
namespace detail {

double l2_norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// 1-D Structural Similarity Index (simplified: luminance + contrast + structure)
double ssim_1d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty())
        throw std::invalid_argument("ssim_1d: vectors must have equal non-zero length");

    const double N    = static_cast<double>(a.size());
    const double C1   = 0.01 * 0.01;  // stabilisation constants
    const double C2   = 0.03 * 0.03;

    double mu_a = 0.0, mu_b = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { mu_a += a[i]; mu_b += b[i]; }
    mu_a /= N; mu_b /= N;

    double sig_a = 0.0, sig_b = 0.0, sig_ab = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double da = a[i] - mu_a, db = b[i] - mu_b;
        sig_a  += da * da;
        sig_b  += db * db;
        sig_ab += da * db;
    }
    sig_a  /= (N - 1.0); sig_b /= (N - 1.0); sig_ab /= (N - 1.0);

    double numerator   = (2.0 * mu_a * mu_b + C1) * (2.0 * sig_ab + C2);
    double denominator = (mu_a * mu_a + mu_b * mu_b + C1) * (sig_a + sig_b + C2);
    return numerator / denominator;
}

double entropy(const std::vector<double>& values, int num_bins) {
    if (values.empty()) return 0.0;
    double vmin = *std::min_element(values.begin(), values.end());
    double vmax = *std::max_element(values.begin(), values.end());
    double range = vmax - vmin;
    if (range == 0.0) return 0.0;

    std::vector<int> hist(num_bins, 0);
    for (double v : values) {
        int bin = static_cast<int>((v - vmin) / range * (num_bins - 1));
        hist[std::clamp(bin, 0, num_bins - 1)]++;
    }
    double H = 0.0;
    double N = static_cast<double>(values.size());
    for (int c : hist) {
        if (c == 0) continue;
        double p = c / N;
        H -= p * std::log2(p);
    }
    return H;
}

double mutual_information(const std::vector<double>& values,
                          const std::vector<int>&    labels,
                          int num_bins) {
    if (values.size() != labels.size() || values.empty()) return 0.0;

    double vmin = *std::min_element(values.begin(), values.end());
    double vmax = *std::max_element(values.begin(), values.end());
    double range = vmax - vmin;
    if (range == 0.0) return 0.0;

    // Collect unique labels
    std::vector<int> ulabels = labels;
    std::sort(ulabels.begin(), ulabels.end());
    ulabels.erase(std::unique(ulabels.begin(), ulabels.end()), ulabels.end());
    int num_classes = static_cast<int>(ulabels.size());

    double N = static_cast<double>(values.size());

    // Joint histogram: [bin][class]
    std::vector<std::vector<int>> joint(num_bins, std::vector<int>(num_classes, 0));
    std::vector<int> hist_f(num_bins, 0);
    std::vector<int> hist_y(num_classes, 0);

    for (std::size_t i = 0; i < values.size(); ++i) {
        int bin = static_cast<int>((values[i] - vmin) / range * (num_bins - 1));
        bin = std::clamp(bin, 0, num_bins - 1);
        int cls = static_cast<int>(std::find(ulabels.begin(), ulabels.end(), labels[i]) - ulabels.begin());
        joint[bin][cls]++;
        hist_f[bin]++;
        hist_y[cls]++;
    }

    double MI = 0.0;
    for (int b = 0; b < num_bins; ++b) {
        for (int c = 0; c < num_classes; ++c) {
            if (joint[b][c] == 0) continue;
            double pxy = joint[b][c] / N;
            double px  = hist_f[b] / N;
            double py  = hist_y[c] / N;
            MI += pxy * std::log2(pxy / (px * py));
        }
    }
    return std::max(MI, 0.0);
}

} // namespace detail

// ─── FFS ─────────────────────────────────────────────────────────────────────
double ffs(const std::vector<double>& original,
           const std::vector<double>& reconstructed) {
    if (original.size() != reconstructed.size() || original.empty())
        throw std::invalid_argument("ffs: original and reconstructed must have equal non-zero length");

    double norm_orig = detail::l2_norm(original);
    if (norm_orig == 0.0) throw std::invalid_argument("ffs: original signal is zero vector");

    std::vector<double> diff(original.size());
    for (std::size_t i = 0; i < original.size(); ++i)
        diff[i] = reconstructed[i] - original[i];

    return 1.0 - (detail::l2_norm(diff) / norm_orig);
}

// ─── MSC ─────────────────────────────────────────────────────────────────────
double msc(const std::vector<std::vector<double>>& substructures_orig,
           const std::vector<std::vector<double>>& substructures_rec) {
    if (substructures_orig.size() != substructures_rec.size() || substructures_orig.empty())
        throw std::invalid_argument("msc: substructure vectors must have equal non-zero size");

    double sum = 0.0;
    for (std::size_t i = 0; i < substructures_orig.size(); ++i)
        sum += detail::ssim_1d(substructures_orig[i], substructures_rec[i]);

    return sum / static_cast<double>(substructures_orig.size());
}

// ─── FRE ─────────────────────────────────────────────────────────────────────
double fre(const std::vector<double>& measured,
           const std::vector<double>& predicted) {
    if (measured.size() != predicted.size() || measured.empty())
        throw std::invalid_argument("fre: measured and predicted must have equal non-zero length");

    double error = 0.0;
    for (std::size_t i = 0; i < measured.size(); ++i)
        error += std::abs(measured[i] - predicted[i]);
    return error;
}

// ─── FFRI ────────────────────────────────────────────────────────────────────
double ffri(const std::vector<double>& feature,
            const std::vector<int>&    labels,
            int                        num_bins) {
    if (feature.empty()) throw std::invalid_argument("ffri: feature vector is empty");

    double H_f = detail::entropy(feature, num_bins);
    if (H_f == 0.0) return 0.0;

    double MI = detail::mutual_information(feature, labels, num_bins);
    return MI / H_f;
}

// ─── SSRI ────────────────────────────────────────────────────────────────────
double ssri(const std::vector<double>& rfd_values) {
    if (rfd_values.empty()) throw std::invalid_argument("ssri: rfd_values is empty");

    double N   = static_cast<double>(rfd_values.size());
    double mean = 0.0;
    for (double v : rfd_values) mean += v;
    mean /= N;

    if (mean == 0.0) return 0.0;

    double var = 0.0;
    for (double v : rfd_values) var += (v - mean) * (v - mean);
    var /= N;

    return 1.0 - (var / mean);
}

// ─── FRI ─────────────────────────────────────────────────────────────────────
double fri(const std::vector<double>& match_scores, double theta) {
    if (match_scores.empty()) throw std::invalid_argument("fri: match_scores is empty");

    double count = 0.0;
    for (double s : match_scores) if (s > theta) count += 1.0;
    return count / static_cast<double>(match_scores.size());
}

} // namespace metrics
} // namespace fractalich
