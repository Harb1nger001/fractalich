#pragma once
/**
 * Fractalich - Fractal-Based Data Analysis Library
 * Neural Network Architectures based on Fractal and Dendritic Principles
 * Version: 0.1.0
 */

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <cstddef>

namespace fractalich {
namespace nn {

// ─── Common types ─────────────────────────────────────────────────────────────
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

/// Simple linear (dense) layer: y = W*x + b
Matrix linear(const Matrix& W, const Vector& x, const Vector& b);

/// Activation functions
Vector relu(const Vector& x);
Vector tanh_act(const Vector& x);
Vector sigmoid(const Vector& x);
Vector softmax(const Vector& x);

/// Scaling (dilation) operator: Ss[x] resamples x at scale s
Vector scale(const Vector& x, double s);

// ─── 1. Fractal-Dendritic Neural Network (FDNN) ────────────────────────────────
/**
 * Models dendritic trees as recursively self-similar computational subunits.
 *
 * Single dendritic branch at one scale:
 *   y_b = phi(Wb * Ss[x] + b_b)
 *
 * The recursive self-similar module is parameterised by depth 'depth'.
 * Somatic integration sums gated branch outputs through a sigmoid gate.
 */
struct FDNNConfig {
    int    input_dim      = 64;   ///< Dimensionality of input x
    int    branch_dim     = 32;   ///< Hidden units per branch
    int    output_dim     = 16;   ///< Somatic output dim
    int    num_branches   = 4;    ///< |B| number of dendritic branches
    int    recursion_depth = 2;   ///< Self-similar recursion depth
    double scale_factor   = 2.0; ///< Dilation factor s
    double lambda_fract   = 0.01;///< Weight for fractal consistency regularisation
};

struct FDNNState {
    std::vector<Matrix> branch_W;  ///< [num_branches][branch_dim][input_dim]
    std::vector<Vector> branch_b;
    Vector              gate_g;    ///< Trainable gates g_b
    Vector              soma_b;    ///< Bias at somatic integration
    Matrix              soma_W;    ///< [output_dim][branch_dim]
};

FDNNState  fdnn_init(const FDNNConfig& cfg, unsigned seed = 42);
Vector     fdnn_forward(const Vector& x, const FDNNState& state, const FDNNConfig& cfg);
double     fdnn_fractal_loss(const Vector& x, const FDNNState& state, const FDNNConfig& cfg);

// ─── 2. Dynamic Dendritic Attention Network (DDAN) ─────────────────────────────
/**
 * Dendritic tree represented as a directed graph; paths are encoded with an
 * RNN-style encoder then attended over given a query.
 *
 * Path attention:  alpha_p = softmax(sim(q, k_p) / sqrt(d))_p
 * Output:          o = sum_p alpha_p * v_p
 */
struct DDANConfig {
    int hidden_dim  = 64;  ///< RNN hidden / key-value dimension
    int num_paths   = 8;   ///< |P(T)| – number of dendritic paths
    int path_len    = 4;   ///< Number of nodes per path
    double lambda_sparse = 1e-3; ///< Sparsity regularisation weight
};

/// One dendritic path (sequence of node embeddings)
using DendriticPath = std::vector<Vector>;

struct DDANState {
    Matrix Wk, Wv;    ///< [hidden_dim x hidden_dim] key/value recurrent weights
    Matrix Wpsi;      ///< Scale embedding projection [hidden_dim x 1]
};

DDANState  ddan_init(const DDANConfig& cfg, unsigned seed = 42);
/// Encodes all paths and returns attended output given query q.
Vector     ddan_forward(const Vector& query,
                        const std::vector<DendriticPath>& paths,
                        const std::vector<int>& path_depths,
                        const DDANState& state, const DDANConfig& cfg);
double     ddan_sparse_loss(const std::vector<double>& alphas, const DDANConfig& cfg);

// ─── 3. Multi-Scale Fractal Convolutional Network (MSFCN) ──────────────────────
/**
 * Applies fractally-scaled convolution kernels and aggregates responses.
 *
 * K_s(u) = (1/alpha_s) * K0(u/s)   (energy-normalised)
 * h_s = x * K_s
 * H   = A({h_s})            where A is a learned weighted sum
 */
struct MSFCNConfig {
    int    kernel_size   = 8;
    int    num_scales    = 4;
    double base_scale    = 2.0;  ///< Scale multiplier per level
    double lambda_scale  = 0.01; ///< Scale-consistency loss weight
    int    num_ifs_terms = 4;    ///< R – number of IFS kernel terms
};

struct MSFCNState {
    Vector              K0;         ///< Base convolution kernel
    std::vector<double> agg_weights; ///< Per-scale aggregation weights
    std::vector<Matrix> ifs_affines; ///< R affine scaling transforms for IFS
    std::vector<double> ifs_coeffs;  ///< a_r IFS coefficients
};

MSFCNState msfcn_init(const MSFCNConfig& cfg, unsigned seed = 42);
Vector     msfcn_conv(const Vector& signal, const Vector& kernel);
Vector     msfcn_forward(const Vector& signal, const MSFCNState& state, const MSFCNConfig& cfg);
double     msfcn_scale_consistency_loss(const Vector& signal, const MSFCNState& state, const MSFCNConfig& cfg);

// ─── 4. Spiking Neural Network with Fractal Trees (SNN-FT) ──────────────────────
/**
 * Leaky-Integrate-and-Fire spiking neurons arranged in fractal dendritic trees.
 *
 * Compartment:  tau_c * dV_c/dt = -V_c + sum_{j in Nc} w_jc * s_j(t)
 * Soma:         tau_m * dV_n/dt = -V_n + sum_{c in Cn} G_c * f_c(V_c)
 * Spike when V_n >= V_th, then reset to V_reset.
 *
 * Fractal connectivity: A = sum_r a_r * P_r * S_sr(A) * P_r^T + E
 */
struct SNNFTConfig {
    int    num_compartments  = 8;
    int    input_dim         = 16;
    double tau_c             = 5.0;   ///< Compartment time constant (ms)
    double tau_m             = 10.0;  ///< Somatic time constant (ms)
    double v_th              = 1.0;   ///< Spike threshold
    double v_reset           = 0.0;   ///< Reset potential
    double dt                = 0.1;   ///< Simulation time step (ms)
    // STDP
    double A_plus            = 0.01;
    double A_minus           = 0.012;
    double tau_plus          = 20.0;
    double tau_minus         = 20.0;
    double eta_scale         = 1.0;  ///< Scale-modulation exponent eta
    // IFS fractal connectivity
    int    num_ifs_terms     = 3;
    double lambda_energy     = 1e-4; ///< Energy regularisation weight
};

struct SNNFTState {
    Matrix              W;           ///< [num_compartments x input_dim] weights
    Vector              G;           ///< Dendritic-to-soma conductances
    Vector              V_soma;      ///< Somatic membrane potentials (batch=1)
    std::vector<Vector> V_comp;      ///< Compartment potentials
    Matrix              A;           ///< Fractal adjacency matrix
    // STDP trace buffers
    Vector              trace_pre;
    Vector              trace_post;
};

SNNFTState snnft_init(const SNNFTConfig& cfg, unsigned seed = 42);

/// Simulate one time step; returns spike vector (0/1 per soma).
Vector snnft_step(const Vector& input_spikes,
                  SNNFTState& state,
                  const SNNFTConfig& cfg);

/// Update weights with STDP rule for one time step.
void snnft_stdp_update(SNNFTState& state, const SNNFTConfig& cfg,
                       const Vector& pre_spikes, const Vector& post_spikes);

/// Build fractal adjacency matrix using IFS.
Matrix snnft_build_adjacency(int n, const SNNFTConfig& cfg, unsigned seed = 42);

double snnft_energy_loss(const std::vector<int>& spike_counts_per_window,
                         const SNNFTConfig& cfg);

} // namespace nn
} // namespace fractalich
