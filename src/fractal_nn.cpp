/**
 * Fractalich - fractal_nn.cpp
 * Neural Network architectures based on fractal/dendritic principles.
 * FDNN, DDAN, MSFCN, SNN-FT
 */

#include "fractal_nn.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>

namespace fractalich {
namespace nn {

// ─── Utilities ────────────────────────────────────────────────────────────────
static double dot(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static Matrix zeros(int rows, int cols) {
    return Matrix(rows, Vector(cols, 0.0));
}

static Matrix rand_matrix(int rows, int cols, std::mt19937& rng, double scale = 0.1) {
    std::normal_distribution<double> dist(0.0, scale);
    Matrix M = zeros(rows, cols);
    for (auto& row : M) for (auto& v : row) v = dist(rng);
    return M;
}

static Vector rand_vector(int n, std::mt19937& rng, double scale = 0.0) {
    std::normal_distribution<double> dist(0.0, 0.05);
    Vector v(n, scale);
    for (auto& x : v) x += dist(rng);
    return v;
}

// ─── Common ops ───────────────────────────────────────────────────────────────
Matrix linear(const Matrix& W, const Vector& x, const Vector& b) {
    int rows = static_cast<int>(W.size());
    Vector y(rows);
    for (int i = 0; i < rows; ++i) {
        y[i] = dot(W[i], x) + b[i];
    }
    return {y}; // 1 x output_dim matrix → caller unpacks [0]
}

Vector relu(const Vector& x) {
    Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::max(0.0, x[i]);
    return y;
}

Vector tanh_act(const Vector& x) {
    Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::tanh(x[i]);
    return y;
}

Vector sigmoid(const Vector& x) {
    Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = 1.0 / (1.0 + std::exp(-x[i]));
    return y;
}

Vector softmax(const Vector& x) {
    double mxv = *std::max_element(x.begin(), x.end());
    Vector y(x.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { y[i] = std::exp(x[i] - mxv); sum += y[i]; }
    for (auto& v : y) v /= sum;
    return y;
}

Vector scale(const Vector& x, double s) {
    // Nearest-neighbor resampling at scale s
    int N_in  = static_cast<int>(x.size());
    int N_out = std::max(1, static_cast<int>(N_in / s));
    Vector y(N_out);
    for (int i = 0; i < N_out; ++i) {
        int src = std::min(static_cast<int>(i * s), N_in - 1);
        y[i] = x[src];
    }
    return y;
}

static Vector matvec(const Matrix& M, const Vector& v) {
    int rows = static_cast<int>(M.size());
    Vector out(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < v.size(); ++j)
            out[i] += M[i][j] * v[j];
    }
    return out;
}

// ─── 1. FDNN ──────────────────────────────────────────────────────────────────
FDNNState fdnn_init(const FDNNConfig& cfg, unsigned seed) {
    std::mt19937 rng(seed);
    FDNNState state;
    for (int b = 0; b < cfg.num_branches; ++b) {
        state.branch_W.push_back(rand_matrix(cfg.branch_dim, cfg.input_dim, rng));
        state.branch_b.push_back(rand_vector(cfg.branch_dim, rng));
    }
    state.gate_g  = rand_vector(cfg.num_branches, rng, 1.0);
    state.soma_b  = rand_vector(cfg.output_dim, rng);
    state.soma_W  = rand_matrix(cfg.output_dim, cfg.branch_dim, rng);
    return state;
}

static Vector fdnn_branch(const Vector& x, const Matrix& W, const Vector& b,
                          double s, const FDNNConfig& cfg) {
    Vector scaled = scale(x, s);
    // Pad/crop to input_dim
    scaled.resize(cfg.input_dim, 0.0);
    Vector pre = matvec(W, scaled);
    for (std::size_t i = 0; i < pre.size(); ++i) pre[i] += b[i];
    return relu(pre);
}

Vector fdnn_forward(const Vector& x, const FDNNState& state, const FDNNConfig& cfg) {
    Vector soma_input(cfg.branch_dim, 0.0);
    Vector gates = sigmoid(state.gate_g);

    for (int b = 0; b < cfg.num_branches; ++b) {
        // Scale factor per branch = s^(b+1)
        double s = std::pow(cfg.scale_factor, b + 1);
        Vector y_b = fdnn_branch(x, state.branch_W[b], state.branch_b[b], s, cfg);
        for (std::size_t i = 0; i < soma_input.size(); ++i)
            soma_input[i] += gates[b] * y_b[i];
    }

    Vector pre_soma = matvec(state.soma_W, soma_input);
    for (std::size_t i = 0; i < pre_soma.size(); ++i) pre_soma[i] += state.soma_b[i];
    return sigmoid(pre_soma);
}

double fdnn_fractal_loss(const Vector& x, const FDNNState& state, const FDNNConfig& cfg) {
    // L_fract = sum_l ||M_{l+1}(x) - S_s(M_l(S_{1/s}(x)))||^2
    // For efficiency: compare outputs at consecutive recursion depths
    double loss = 0.0;
    Vector x_current = x;
    Vector x_scaled   = scale(x, 1.0 / cfg.scale_factor);
    x_scaled.resize(cfg.input_dim, 0.0);
    Vector out_l   = fdnn_forward(x_current, state, cfg);
    Vector out_lp1 = fdnn_forward(x_scaled,  state, cfg);
    for (std::size_t i = 0; i < out_l.size(); ++i) {
        double diff = out_lp1[i] - out_l[i];
        loss += diff * diff;
    }
    return loss * cfg.lambda_fract;
}

// ─── 2. DDAN ─────────────────────────────────────────────────────────────────
DDANState ddan_init(const DDANConfig& cfg, unsigned seed) {
    std::mt19937 rng(seed);
    DDANState state;
    state.Wk   = rand_matrix(cfg.hidden_dim, cfg.hidden_dim, rng);
    state.Wv   = rand_matrix(cfg.hidden_dim, cfg.hidden_dim, rng);
    state.Wpsi = rand_matrix(cfg.hidden_dim, 1, rng);
    return state;
}

static Vector rnn_encode(const DendriticPath& path, const Matrix& W) {
    // Simple Elman-style: h_t = tanh(W * h_{t-1} + x_t)
    if (path.empty()) return {};
    int hidden = static_cast<int>(W.size());
    Vector h(hidden, 0.0);
    for (const auto& node : path) {
        Vector Wh = matvec(W, h);
        for (int i = 0; i < hidden; ++i)
            h[i] = std::tanh(Wh[i] + (i < static_cast<int>(node.size()) ? node[i] : 0.0));
    }
    return h;
}

Vector ddan_forward(const Vector& query,
                    const std::vector<DendriticPath>& paths,
                    const std::vector<int>& path_depths,
                    const DDANState& state, const DDANConfig& cfg) {
    int P = static_cast<int>(paths.size());
    if (P == 0) throw std::invalid_argument("ddan_forward: no paths");

    // Encode paths
    std::vector<Vector> keys(P), values(P);
    for (int p = 0; p < P; ++p) {
        keys[p]   = rnn_encode(paths[p], state.Wk);
        values[p] = rnn_encode(paths[p], state.Wv);

        // Scale-aware key: k_p += W_psi * psi(sp)
        double sp = path_depths.size() > std::size_t(p) ? path_depths[p] : 0;
        double psi = std::log1p(std::abs(sp)); // simple scale embedding
        for (int i = 0; i < cfg.hidden_dim; ++i)
            if (!state.Wpsi[i].empty())
                keys[p][i] += state.Wpsi[i][0] * psi;
    }

    // Attention scores
    double sqrt_d = std::sqrt(static_cast<double>(cfg.hidden_dim));
    Vector raw_scores(P);
    for (int p = 0; p < P; ++p) {
        double q_len = std::min(query.size(), keys[p].size());
        double s = 0.0;
        for (int i = 0; i < static_cast<int>(q_len); ++i)
            s += query[i] * keys[p][i];
        raw_scores[p] = s / sqrt_d;
    }
    Vector alphas = softmax(raw_scores);

    // Attended output
    int out_dim = cfg.hidden_dim;
    Vector output(out_dim, 0.0);
    for (int p = 0; p < P; ++p) {
        for (int i = 0; i < out_dim && i < static_cast<int>(values[p].size()); ++i)
            output[i] += alphas[p] * values[p][i];
    }
    return output;
}

double ddan_sparse_loss(const std::vector<double>& alphas, const DDANConfig& cfg) {
    double sum = 0.0;
    for (double a : alphas) sum += a;
    return cfg.lambda_sparse * sum;
}

// ─── 3. MSFCN ────────────────────────────────────────────────────────────────
MSFCNState msfcn_init(const MSFCNConfig& cfg, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.1);
    MSFCNState state;
    state.K0.resize(cfg.kernel_size, 0.0);
    for (auto& v : state.K0) v = dist(rng);
    state.agg_weights.resize(cfg.num_scales, 1.0 / cfg.num_scales);
    for (int r = 0; r < cfg.num_ifs_terms; ++r) {
        state.ifs_affines.push_back(rand_matrix(cfg.kernel_size, cfg.kernel_size, rng, 0.05));
        state.ifs_coeffs.push_back(0.5 / cfg.num_ifs_terms);
    }
    return state;
}

Vector msfcn_conv(const Vector& signal, const Vector& kernel) {
    int N = static_cast<int>(signal.size());
    int K = static_cast<int>(kernel.size());
    int out_size = N - K + 1;
    if (out_size <= 0) return {};
    Vector out(out_size, 0.0);
    for (int i = 0; i < out_size; ++i)
        for (int j = 0; j < K; ++j)
            out[i] += signal[i + j] * kernel[j];
    return out;
}

static Vector msfcn_scaled_kernel(const Vector& K0, double s) {
    // K_s(u) = (1/alpha_s) * K0(u/s): nearest-neighbor scaling
    int N = static_cast<int>(K0.size());
    int new_size = std::max(1, static_cast<int>(N * s));
    Vector Ks(new_size);
    double alpha_s = std::sqrt(s); // energy normalisation
    for (int i = 0; i < new_size; ++i) {
        int src = std::min(static_cast<int>(i / s), N - 1);
        Ks[i] = K0[src] / alpha_s;
    }
    return Ks;
}

static Vector project_to_scale(const Vector& h, int target_len) {
    if (static_cast<int>(h.size()) == target_len) return h;
    Vector out(target_len, 0.0);
    for (int i = 0; i < target_len; ++i) {
        int src = static_cast<int>(i * h.size() / target_len);
        out[i] = h[std::min(src, static_cast<int>(h.size()) - 1)];
    }
    return out;
}

Vector msfcn_forward(const Vector& signal, const MSFCNState& state, const MSFCNConfig& cfg) {
    std::vector<Vector> hs;
    for (int si = 0; si < cfg.num_scales; ++si) {
        double s = std::pow(cfg.base_scale, si);
        Vector Ks = msfcn_scaled_kernel(state.K0, s);
        Vector h  = msfcn_conv(signal, Ks);
        hs.push_back(h);
    }
    // Aggregate: H = weighted sum of h_s (project to common length = minimum)
    int min_len = static_cast<int>(signal.size());
    for (auto& h : hs) min_len = std::min(min_len, static_cast<int>(h.size()));
    if (min_len <= 0) return {};

    Vector H(min_len, 0.0);
    for (int si = 0; si < cfg.num_scales; ++si) {
        Vector hproj = project_to_scale(hs[si], min_len);
        for (int i = 0; i < min_len; ++i)
            H[i] += state.agg_weights[si] * hproj[i];
    }
    return H;
}

double msfcn_scale_consistency_loss(const Vector& signal, const MSFCNState& state, const MSFCNConfig& cfg) {
    // L_scale = sum_{s != s'} ||proj(h_s, s') - h_s'||^2
    std::vector<Vector> hs;
    for (int si = 0; si < cfg.num_scales; ++si) {
        double s = std::pow(cfg.base_scale, si);
        Vector Ks = msfcn_scaled_kernel(state.K0, s);
        hs.push_back(msfcn_conv(signal, Ks));
    }
    double loss = 0.0;
    for (int i = 0; i < cfg.num_scales; ++i) {
        for (int j = 0; j < cfg.num_scales; ++j) {
            if (i == j) continue;
            int target = static_cast<int>(hs[j].size());
            if (target == 0) continue;
            Vector hproj = project_to_scale(hs[i], target);
            for (int k = 0; k < target; ++k) {
                double diff = hproj[k] - hs[j][k];
                loss += diff * diff;
            }
        }
    }
    return loss * cfg.lambda_scale;
}

// ─── 4. SNN-FT ───────────────────────────────────────────────────────────────
Matrix snnft_build_adjacency(int n, const SNNFTConfig& cfg, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.01);
    Matrix A = zeros(n, n);
    // Identity + small noise for base fractal structure
    for (int i = 0; i < n; ++i) {
        A[i][i] = 1.0;
        for (int j = 0; j < n; ++j) A[i][j] += noise(rng);
    }
    // IFS scaling: A = sum_r a_r * P_r * S_sr(A) * P_r^T + E
    for (int r = 0; r < cfg.num_ifs_terms; ++r) {
        double ar = 0.5 / cfg.num_ifs_terms;
        std::shuffle(std::vector<int>(n).begin(), std::vector<int>(n).end(), rng); // permutation P_r stub
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A[i][j] += ar * A[std::min(i * 2, n - 1)][std::min(j * 2, n - 1)];
    }
    // Normalise rows
    for (int i = 0; i < n; ++i) {
        double rowsum = 0.0;
        for (int j = 0; j < n; ++j) rowsum += std::abs(A[i][j]);
        if (rowsum > 0) for (int j = 0; j < n; ++j) A[i][j] /= rowsum;
    }
    return A;
}

SNNFTState snnft_init(const SNNFTConfig& cfg, unsigned seed) {
    std::mt19937 rng(seed);
    SNNFTState state;
    state.W        = rand_matrix(cfg.num_compartments, cfg.input_dim, rng);
    state.G        = rand_vector(cfg.num_compartments, rng, 1.0);
    state.V_soma   = Vector(1, cfg.v_reset);
    state.V_comp   = std::vector<Vector>(1, Vector(cfg.num_compartments, cfg.v_reset));
    state.A        = snnft_build_adjacency(cfg.num_compartments, cfg, seed);
    state.trace_pre  = Vector(cfg.input_dim, 0.0);
    state.trace_post = Vector(1, 0.0);
    return state;
}

Vector snnft_step(const Vector& input_spikes, SNNFTState& state, const SNNFTConfig& cfg) {
    double dt = cfg.dt;
    auto& Vc = state.V_comp[0];
    double& Vn = state.V_soma[0];

    // Update compartments: tau_c * dV_c/dt = -V_c + sum_j w_jc * s_j(t)
    for (int c = 0; c < cfg.num_compartments; ++c) {
        double input_current = 0.0;
        for (int j = 0; j < cfg.input_dim && j < static_cast<int>(input_spikes.size()); ++j)
            input_current += state.W[c][j] * input_spikes[j];
        double dVc = (-Vc[c] + input_current) / cfg.tau_c;
        Vc[c] += dVc * dt;
    }

    // Somatic integration: tau_m * dV_n/dt = -V_n + sum_c G_c * tanh(V_c)
    double soma_input = 0.0;
    for (int c = 0; c < cfg.num_compartments; ++c)
        soma_input += state.G[c] * std::tanh(Vc[c]);

    double dVn = (-Vn + soma_input) / cfg.tau_m;
    Vn += dVn * dt;

    // Spike and reset
    Vector spikes(1, 0.0);
    if (Vn >= cfg.v_th) {
        spikes[0] = 1.0;
        Vn = cfg.v_reset;
    }
    return spikes;
}

void snnft_stdp_update(SNNFTState& state, const SNNFTConfig& cfg,
                       const Vector& pre_spikes, const Vector& post_spikes) {
    double tau_plus  = cfg.tau_plus;
    double tau_minus = cfg.tau_minus;
    double dt        = cfg.dt;

    // Update traces
    for (std::size_t j = 0; j < state.trace_pre.size() && j < pre_spikes.size(); ++j)
        state.trace_pre[j] = state.trace_pre[j] * std::exp(-dt / tau_plus) + pre_spikes[j];
    for (std::size_t n = 0; n < state.trace_post.size() && n < post_spikes.size(); ++n)
        state.trace_post[n] = state.trace_post[n] * std::exp(-dt / tau_minus) + post_spikes[n];

    // STDP weight updates: potentiation on post-spike, depression on pre-spike
    for (int c = 0; c < cfg.num_compartments; ++c) {
        double scale_mod = std::pow(static_cast<double>(c + 1), -cfg.eta_scale);
        for (int j = 0; j < cfg.input_dim && j < static_cast<int>(pre_spikes.size()); ++j) {
            double dw = 0.0;
            // Potentiation: post fires, reward pre->post
            if (!post_spikes.empty() && post_spikes[0] > 0.5)
                dw += cfg.A_plus * state.trace_pre[j];
            // Depression: pre fires, punish
            if (pre_spikes[j] > 0.5)
                dw -= cfg.A_minus * (state.trace_post.empty() ? 0.0 : state.trace_post[0]);
            state.W[c][j] += dw * scale_mod;
        }
    }
}

double snnft_energy_loss(const std::vector<int>& spike_counts, const SNNFTConfig& cfg) {
    if (spike_counts.empty()) return 0.0;
    double mean_spikes = 0.0;
    for (int c : spike_counts) mean_spikes += c;
    mean_spikes /= spike_counts.size();
    return cfg.lambda_energy * mean_spikes;
}

} // namespace nn
} // namespace fractalich
