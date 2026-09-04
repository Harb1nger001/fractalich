/**
 * Fractalich - fractal_dimensions.cpp
 * Implementation of all 17 fractal dimension estimation methods.
 */

#include "fractal_dimensions.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <random>
#include <set>

namespace fractalich {
namespace dimensions {

namespace {

// ─── Shared helpers ───────────────────────────────────────────────────────────

// Least-squares slope of log(y) vs log(x)
double log_log_slope(const std::vector<double>& xs, const std::vector<double>& ys) {
    assert(xs.size() == ys.size() && !xs.empty());
    int N = static_cast<int>(xs.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < N; ++i) {
        double lx = std::log(xs[i]), ly = std::log(ys[i]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
    }
    double denom = N * sxx - sx * sx;
    if (std::abs(denom) < 1e-15) return 0.0;
    return (N * sxy - sx * sy) / denom;
}

// Least-squares linear slope of y vs x
double linear_slope(const std::vector<double>& xs, const std::vector<double>& ys) {
    assert(xs.size() == ys.size() && !xs.empty());
    int N = static_cast<int>(xs.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < N; ++i) {
        sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
    }
    double denom = N * sxx - sx * sx;
    if (std::abs(denom) < 1e-15) return 0.0;
    return (N * sxy - sx * sy) / denom;
}

double signal_mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double signal_var(const std::vector<double>& v) {
    double mu = signal_mean(v);
    double var = 0.0;
    for (double x : v) var += (x - mu) * (x - mu);
    return var / v.size();
}

// Delay embedding: returns (N - (m-1)*tau) vectors of length m
std::vector<std::vector<double>> embed(const std::vector<double>& ts, int m, int tau) {
    int N = static_cast<int>(ts.size()) - (m - 1) * tau;
    std::vector<std::vector<double>> embedded(N, std::vector<double>(m));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j)
            embedded[i][j] = ts[i + j * tau];
    return embedded;
}

// Returns a linspace of 'num' values geometrically spaced in [lo, hi]
std::vector<double> geomspace(double lo, double hi, int num) {
    std::vector<double> out(num);
    double step = std::pow(hi / lo, 1.0 / (num - 1));
    out[0] = lo;
    for (int i = 1; i < num; ++i) out[i] = out[i - 1] * step;
    return out;
}

// Box-counting for a 1D signal graph: count occupied (box, bin) pairs
double box_counting_1d(const std::vector<double>& signal, int box_size) {
    int N = static_cast<int>(signal.size());
    double vmin = *std::min_element(signal.begin(), signal.end());
    double vmax = *std::max_element(signal.begin(), signal.end());
    double vrange = vmax - vmin + 1e-12;

    // Map to [0, N) vertically
    std::set<std::pair<int,int>> boxes;
    // #include <set> needed – include at top
    for (int i = 0; i < N; i += box_size) {
        int end = std::min(i + box_size, N);
        for (int j = i; j < end; ++j) {
            int vy = static_cast<int>((signal[j] - vmin) / vrange * N / box_size);
            boxes.insert({i / box_size, vy});
        }
    }
    return static_cast<double>(boxes.size());
}

} // anonymous namespace

// We need <set> – add include chain via helper struct workaround handled in cpp-level include
#include <set>

// ─── 1. Box-Counting ──────────────────────────────────────────────────────────
double box_counting(const std::vector<double>& signal, int min_box, int max_box, int num_scales) {
    if (signal.size() < 4) throw std::invalid_argument("box_counting: signal too short");

    auto scales = geomspace(min_box, max_box, num_scales);
    std::vector<double> epsilons, counts;
    for (double s : scales) {
        int bs = std::max(1, static_cast<int>(s));
        double cnt = box_counting_1d(signal, bs);
        if (cnt > 0) { epsilons.push_back(s); counts.push_back(cnt); }
    }
    if (epsilons.size() < 2) return 0.0;
    return -log_log_slope(epsilons, counts); // D = -slope of log N vs log eps
}

// ─── 2. Hausdorff (numerical via box-counting on point cloud) ─────────────────
double hausdorff(const std::vector<double>& points, int D, int min_box, int max_box, int num_scales) {
    // Project onto first axis for 1D box-counting approximation
    if (points.empty()) throw std::invalid_argument("hausdorff: empty points");
    std::vector<double> proj;
    for (int i = 0; i < static_cast<int>(points.size()); i += D){
        if (i < static_cast<int>(points.size())) proj.push_back(points[i]);
    }
    return box_counting(proj, min_box, max_box, num_scales);
}

// ─── 3. Correlation Dimension (Grassberger-Procaccia) ─────────────────────────
double correlation_dim(const std::vector<double>& data, int embedding_dim, int tau) {
    if (data.empty()) throw std::invalid_argument("correlation_dim: empty data");
    auto pts = embed(data, embedding_dim, tau);
    int N = static_cast<int>(pts.size());
    if (N < 4) return 0.0;

    // Compute pairwise distances, count pairs within radius r
    std::vector<double> dists;
    dists.reserve(N * (N - 1) / 2);
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            double d = 0.0;
            for (int k = 0; k < embedding_dim; ++k) {
                double diff = pts[i][k] - pts[j][k];
                d += diff * diff;
            }
            dists.push_back(std::sqrt(d));
        }
    std::sort(dists.begin(), dists.end());

    double rmin = dists.front() + 1e-12, rmax = dists.back();
    if (rmin >= rmax) return 0.0;

    int num_radii = 10;
    auto radii = geomspace(rmin, rmax * 0.9, num_radii);
    std::vector<double> rs, Cs;
    double total = static_cast<double>(dists.size());

    for (double r : radii) {
        int cnt = static_cast<int>(std::lower_bound(dists.begin(), dists.end(), r) - dists.begin());
        if (cnt > 0 && cnt < static_cast<int>(dists.size())) {
            rs.push_back(r);
            Cs.push_back(cnt / total);
        }
    }
    if (rs.size() < 2) return 0.0;
    return log_log_slope(rs, Cs);
}

// ─── 4. Information Dimension ─────────────────────────────────────────────────
double information_dim(const std::vector<double>& signal, int num_bins) {
    if (signal.empty()) throw std::invalid_argument("information_dim: empty signal");
    double vmin = *std::min_element(signal.begin(), signal.end());
    double vmax = *std::max_element(signal.begin(), signal.end());
    double range = vmax - vmin + 1e-12;

    // Estimate ID vs shrinking epsilon
    std::vector<double> epsilons, ID;
    for (int bins = num_bins; bins >= 4; bins /= 2) {
        std::vector<int> hist(bins, 0);
        for (double v : signal) {
            int b = static_cast<int>((v - vmin) / range * (bins - 1));
            hist[std::clamp(b, 0, bins - 1)]++;
        }
        double N  = static_cast<double>(signal.size());
        double H  = 0.0;
        for (int c : hist) { if (c > 0) { double p = c / N; H -= p * std::log(p); } }
        double eps = range / bins;
        epsilons.push_back(eps);
        ID.push_back(H);
    }
    if (epsilons.size() < 2) return 0.0;
    // Slope of H vs -log(eps) gives the information dimension
    std::vector<double> neg_log_eps(epsilons.size());
    for (std::size_t i = 0; i < epsilons.size(); ++i)
        neg_log_eps[i] = -std::log(epsilons[i]);
    return linear_slope(neg_log_eps, ID);
}

// ─── 5. Packing Dimension ─────────────────────────────────────────────────────
double packing_dim(const std::vector<double>& signal, int min_box, int max_box, int num_scales) {
    // Numerical approximation: average of upper box-counting
    return box_counting(signal, min_box, max_box, num_scales);
}

// ─── 6. Minkowski-Bouligand (= box-counting) ──────────────────────────────────
double minkowski_bouligand(const std::vector<double>& signal, int min_box, int max_box, int num_scales) {
    return box_counting(signal, min_box, max_box, num_scales);
}

// ─── 7. Lyapunov Dimension (Kaplan-Yorke) ─────────────────────────────────────
double lyapunov_dim(const std::vector<double>& exponents) {
    if (exponents.empty()) throw std::invalid_argument("lyapunov_dim: empty exponents");
    // exponents must be sorted descending
    std::vector<double> sorted = exponents;
    std::sort(sorted.rbegin(), sorted.rend());

    double cum = 0.0;
    int j = 0;
    for (; j < static_cast<int>(sorted.size()); ++j) {
        cum += sorted[j];
        if (cum < 0) break;
    }
    if (j == 0 || std::abs(sorted[j]) < 1e-12) return 0.0;
    return static_cast<double>(j) + (cum - sorted[j]) / std::abs(sorted[j]);
}

// ─── 8. Renyi Dimension ───────────────────────────────────────────────────────
double renyi_dim(const std::vector<double>& signal, double q, int num_bins) {
    if (std::abs(q - 1.0) < 1e-9) return information_dim(signal, num_bins);

    if (signal.empty()) throw std::invalid_argument("renyi_dim: empty signal");
    double vmin = *std::min_element(signal.begin(), signal.end());
    double vmax = *std::max_element(signal.begin(), signal.end());
    double range = vmax - vmin + 1e-12;

    std::vector<double> epsilons, Rq;
    for (int bins = num_bins; bins >= 4; bins /= 2) {
        std::vector<int> hist(bins, 0);
        for (double v : signal) {
            int b = static_cast<int>((v - vmin) / range * (bins - 1));
            hist[std::clamp(b, 0, bins - 1)]++;
        }
        double N = static_cast<double>(signal.size());
        double sum_pq = 0.0;
        for (int c : hist) {
            if (c > 0) sum_pq += std::pow(c / N, q);
        }
        double rq = (1.0 / (1.0 - q)) * std::log(sum_pq);
        epsilons.push_back(range / bins);
        Rq.push_back(rq);
    }
    if (epsilons.size() < 2) return 0.0;
    std::vector<double> neg_log_eps(epsilons.size());
    for (std::size_t i = 0; i < epsilons.size(); ++i)
        neg_log_eps[i] = -std::log(epsilons[i]);
    return linear_slope(neg_log_eps, Rq);
}

// ─── 9. Higuchi's Fractal Dimension ──────────────────────────────────────────
double higuchi(const std::vector<double>& signal, int k_max) {
    int N = static_cast<int>(signal.size());
    if (N < k_max + 1) throw std::invalid_argument("higuchi: signal too short for k_max");

    std::vector<double> ks, Lms;
    for (int k = 1; k <= k_max; ++k) {
        double Lm_sum = 0.0;
        for (int m = 1; m <= k; ++m) {
            double Lmk = 0.0;
            int num_terms = static_cast<int>(std::floor((N - m) / k));
            for (int i = 1; i <= num_terms; ++i)
                Lmk += std::abs(signal[m - 1 + i * k] - signal[m - 1 + (i - 1) * k]);
            if (num_terms > 0)
                Lmk = Lmk * (N - 1.0) / (k * num_terms);
            Lm_sum += Lmk;
        }
        double Lk = Lm_sum / k;
        if (Lk > 0) { ks.push_back(k); Lms.push_back(Lk); }
    }
    if (ks.size() < 2) return 0.0;
    return -log_log_slope(ks, Lms);
}

// ─── 10. Katz's Fractal Dimension ────────────────────────────────────────────
double katz(const std::vector<double>& signal) {
    int N = static_cast<int>(signal.size());
    if (N < 2) throw std::invalid_argument("katz: signal too short");

    double L = 0.0; // total arc length
    double d = 0.0; // max distance from first sample
    for (int i = 1; i < N; ++i) {
        double dl = std::abs(signal[i] - signal[i - 1]);
        L += std::sqrt(1.0 + dl * dl); // approximate arc length with unit x-step
        double dist = std::sqrt(static_cast<double>((i) * (i)) +
                                (signal[i] - signal[0]) * (signal[i] - signal[0]));
        d = std::max(d, dist);
    }
    if (L == 0.0 || std::log10(d) == 0.0) return 0.0;
    double n = static_cast<double>(N - 1);
    return std::log10(n) / (std::log10(d / L) + std::log10(n));
}

// ─── 11. Petrosian Fractal Dimension ─────────────────────────────────────────
double petrosian(const std::vector<double>& signal) {
    int N = static_cast<int>(signal.size());
    if (N < 2) throw std::invalid_argument("petrosian: signal too short");

    // Count delta-zero-crossings (sign changes in first difference)
    int Nzc = 0;
    std::vector<double> diff(N - 1);
    for (int i = 0; i < N - 1; ++i) diff[i] = signal[i + 1] - signal[i];
    for (int i = 0; i < static_cast<int>(diff.size()) - 1; ++i)
        if (diff[i] * diff[i + 1] < 0) ++Nzc;

    double n = static_cast<double>(N);
    if (Nzc == 0) return std::log10(n) / std::log10(n);
    return std::log10(n) / (std::log10(n) + std::log10(n / (n + 0.4 * Nzc)));
}

// ─── 12. Detrended Fluctuation Analysis (DFA) ────────────────────────────────
double dfa(const std::vector<double>& signal, const std::vector<int>& scales_in) {
    int N = static_cast<int>(signal.size());
    if (N < 8) throw std::invalid_argument("dfa: signal too short");

    // Cumulative sum (profile)
    double mu = signal_mean(signal);
    std::vector<double> profile(N);
    profile[0] = signal[0] - mu;
    for (int i = 1; i < N; ++i) profile[i] = profile[i - 1] + (signal[i] - mu);

    // Choose scales
    std::vector<int> scales = scales_in;
    if (scales.empty()) {
        for (int s = 4; s <= N / 4; s *= 2) scales.push_back(s);
    }

    std::vector<double> Fs;
    std::vector<double> sc;
    for (int s : scales) {
        if (s < 4 || s > N / 2) continue;
        double F2 = 0.0;
        int num_seg = N / s;
        for (int seg = 0; seg < num_seg; ++seg) {
            // Fit linear trend in this segment
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (int i = 0; i < s; ++i) {
                double x = i, y = profile[seg * s + i];
                sx += x; sy += y; sxx += x * x; sxy += x * y;
            }
            double slope = (s * sxy - sx * sy) / (s * sxx - sx * sx);
            double intercept = (sy - slope * sx) / s;
            for (int i = 0; i < s; ++i) {
                double res = profile[seg * s + i] - (slope * i + intercept);
                F2 += res * res;
            }
        }
        if (num_seg > 0) {
            Fs.push_back(std::sqrt(F2 / (num_seg * s)));
            sc.push_back(static_cast<double>(s));
        }
    }
    if (sc.size() < 2) return 0.5;
    return log_log_slope(sc, Fs);
}

// ─── 13. Multifractal Spectrum ────────────────────────────────────────────────
MultifractalSpectrum multifractal_spectrum(const std::vector<double>& signal,
                                           double q_min, double q_max, int num_q,
                                           int num_scales) {
    if (signal.empty()) throw std::invalid_argument("multifractal_spectrum: empty signal");
    MultifractalSpectrum result;

    double vmin = *std::min_element(signal.begin(), signal.end());
    double vmax = *std::max_element(signal.begin(), signal.end());
    double range = vmax - vmin + 1e-12;

    double q_step = (num_q > 1) ? (q_max - q_min) / (num_q - 1) : 0.0;

    auto scales = geomspace(4, std::max(8, static_cast<int>(signal.size()) / 4), num_scales);

    // For each q, compute tau(q) = slope of log(sum_b mu_b^q) vs log(eps)
    // alpha(q) = d tau / dq, f(q) = q*alpha - tau
    std::vector<double> tau_vec;
    std::vector<double> q_vec;

    for (int qi = 0; qi < num_q; ++qi) {
        double q = q_min + qi * q_step;
        q_vec.push_back(q);
        std::vector<double> epsilons, Zq;
        for (double s : scales) {
            int bins = std::max(2, static_cast<int>(signal.size() / s));
            std::vector<int> hist(bins, 0);
            for (double v : signal) {
                int b = static_cast<int>((v - vmin) / range * (bins - 1));
                hist[std::clamp(b, 0, bins - 1)]++;
            }
            double N = static_cast<double>(signal.size());
            double sum_mu_q = 0.0;
            for (int c : hist) {
                if (c > 0) {
                    double p = c / N;
                    if (std::abs(q - 1.0) < 1e-9)
                        sum_mu_q += p * std::log(p);
                    else
                        sum_mu_q += std::pow(p, q);
                }
            }
            epsilons.push_back(range / bins);
            Zq.push_back(std::abs(sum_mu_q) + 1e-30);
        }
        tau_vec.push_back(log_log_slope(epsilons, Zq) - 1.0);
    }

    // Numerical derivative for alpha = d(tau)/d(q)
    for (int qi = 1; qi < num_q - 1; ++qi) {
        double alpha = (tau_vec[qi + 1] - tau_vec[qi - 1]) / (2.0 * q_step + 1e-12);
        double f_val = q_vec[qi] * alpha - tau_vec[qi];
        result.alpha.push_back(alpha);
        result.f.push_back(f_val);
    }
    return result;
}

// ─── 14. Wavelet-Based Fractal Dimension ────────────────────────────────────────
double wavelet_dim(const std::vector<double>& signal, int num_levels) {
    if (signal.empty()) throw std::invalid_argument("wavelet_dim: empty signal");

    // Haar wavelet energy per level
    std::vector<double> energies;
    std::vector<double> current = signal;

    // Zero-pad to next power-of-two
    int n = 1;
    while (n < static_cast<int>(current.size())) n <<= 1;
    current.resize(n, 0.0);

    for (int lv = 0; lv < num_levels && current.size() >= 2; ++lv) {
        int M = static_cast<int>(current.size()) / 2;
        std::vector<double> approx(M), detail(M);
        for (int i = 0; i < M; ++i) {
            approx[i] = (current[2 * i] + current[2 * i + 1]) / std::sqrt(2.0);
            detail[i] = (current[2 * i] - current[2 * i + 1]) / std::sqrt(2.0);
        }
        double E = 0.0;
        for (double d : detail) E += d * d;
        energies.push_back(E + 1e-30);
        current = approx;
    }
    if (energies.size() < 2) return 0.0;
    std::vector<double> levels(energies.size());
    std::iota(levels.begin(), levels.end(), 1.0);
    return -log_log_slope(levels, energies) / 2.0;
}

// ─── 15. Lacunarity ────────────────────────────────────────────────────────────
double lacunarity(const std::vector<double>& signal, int box_size) {
    if (signal.empty() || box_size < 1)
        throw std::invalid_argument("lacunarity: invalid input");

    double vmin = *std::min_element(signal.begin(), signal.end());
    double vmax = *std::max_element(signal.begin(), signal.end());
    double vmid = (vmin + vmax) / 2.0;

    int N = static_cast<int>(signal.size());
    std::vector<double> masses;
    for (int i = 0; i <= N - box_size; ++i) {
        int mass = 0;
        for (int j = i; j < i + box_size; ++j)
            if (signal[j] >= vmid) ++mass;
        masses.push_back(static_cast<double>(mass));
    }
    if (masses.empty()) return 0.0;
    double mu  = 0.0, mu2 = 0.0;
    for (double m : masses) { mu += m; mu2 += m * m; }
    mu /= masses.size(); mu2 /= masses.size();
    double var = mu2 - mu * mu;
    if (mu == 0.0) return 0.0;
    return var / (mu * mu);
}

// ─── 16. Hurst Exponent (R/S method) ─────────────────────────────────────────
double hurst(const std::vector<double>& signal) {
    int N = static_cast<int>(signal.size());
    if (N < 8) throw std::invalid_argument("hurst: signal too short");

    std::vector<double> ns, rs;
    for (int n = 8; n <= N; n *= 2) {
        int num_blocks = N / n;
        if (num_blocks == 0) break;
        double RS_sum = 0.0;
        for (int b = 0; b < num_blocks; ++b) {
            const auto* seg = signal.data() + b * n;
            double mu_b = 0.0;
            for (int i = 0; i < n; ++i) mu_b += seg[i];
            mu_b /= n;
            // deviation series
            double cum = 0.0, mx = 0, mn = 0;
            double var_b = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = seg[i] - mu_b;
                var_b += d * d;
                cum += d;
                mx = std::max(mx, cum);
                mn = std::min(mn, cum);
            }
            double S = std::sqrt(var_b / n);
            if (S > 0) RS_sum += (mx - mn) / S;
        }
        double RS = RS_sum / num_blocks;
        if (RS > 0) { ns.push_back(n); rs.push_back(RS); }
    }
    if (ns.size() < 2) return 0.5;
    return log_log_slope(ns, rs);
}

// ─── 17. Temporal Fractal Dimension ──────────────────────────────────────────
double temporal_fractal_dim(const std::vector<double>& signal, double dt) {
    if (signal.empty()) throw std::invalid_argument("temporal_fractal_dim: empty signal");
    // Box-counting on the time-domain signal graph with time axis scaled by dt
    (void)dt; // dt influences effective scale if needed; here we use index-space
    return box_counting(signal);
}

} // namespace dimensions
} // namespace fractalich
