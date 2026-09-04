"""
Fractalich Quick-Start Example
================================
Run after installing: pip install fractalich
Or from source: pip install .
"""

import sys
import os
import numpy as np

# Allow running directly from source directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─── 1. Import Fractalich ─────────────────────────────────────────────────────
try:
    from fractalich import metrics, dimensions, neural
    import fractalich
except ImportError as e:
    print(f"ERROR: Fractalich is not installed or extension not built: {e}")
    print("Please install via: pip install fractalich  (or pip install .)")
    sys.exit(1)

rng = np.random.default_rng(42)

print("=" * 60)
print(f"Fractalich v{fractalich.__version__} — Comprehensive Quick-Start Demo")
print("=" * 60)

# ─── 2. Evaluation Metrics ────────────────────────────────────────────────────
print("\n[1. Evaluation Metrics]")
original      = rng.random(1024)
reconstructed = original + rng.normal(0, 0.05, size=1024)

# Fractal Fidelity Score (FFS)
ffs_score = metrics.ffs(original, reconstructed)
print(f"  FFS  (Fractal Fidelity Score)       : {ffs_score:.6f} / 1.0")

# Multiscale Structural Consistency (MSC)
scales_orig = [original[i*256:(i+1)*256] for i in range(4)]
scales_rec  = [reconstructed[i*256:(i+1)*256] for i in range(4)]
msc_score   = metrics.msc(scales_orig, scales_rec)
print(f"  MSC  (Multiscale Structural Consist): {msc_score:.6f} / 1.0")

# Fractal Residual Error (FRE)
dims_meas = np.array([1.5, 1.7, 2.0])
dims_pred = np.array([1.4, 1.8, 1.9])
fre_score = metrics.fre(dims_meas, dims_pred)
print(f"  FRE  (Fractal Residual Error)       : {fre_score:.6f}")

# Fractal Feature Relevance Index (FFRI)
labels_i   = rng.integers(0, 2, size=1024).astype(np.int32)
ffri_score = metrics.ffri(original, labels_i, num_bins=50)
print(f"  FFRI (Feature Relevance Index)      : {ffri_score:.6f}")

# Self-Similarity Retention Index (SSRI)
rfd = np.array([1.51, 1.49, 1.50, 1.52, 1.50])
ssri_score = metrics.ssri(rfd)
print(f"  SSRI (Self-Similarity Retention)    : {ssri_score:.6f}")

# Fractal Recurrence Index (FRI)
match_scores = rng.random(200)
fri_score = metrics.fri(match_scores, theta=0.5)
print(f"  FRI  (Fractal Recurrence Rate)      : {fri_score:.6f}")

# ─── 3. Fractal Dimensions ────────────────────────────────────────────────────
print("\n[2. Fractal Dimensions]")
signal = np.cumsum(rng.standard_normal(1024))

print(f"  Higuchi Dimension (k_max=16)       : {dimensions.higuchi(signal, k_max=16):.4f}")
print(f"  Katz Waveform Dimension            : {dimensions.katz(signal):.4f}")
print(f"  Petrosian (Zero-Crossing) Dim      : {dimensions.petrosian(signal):.4f}")
print(f"  Hurst Exponent (R/S Persistence)   : {dimensions.hurst(signal):.4f}")
print(f"  DFA Scaling Exponent (Alpha)       : {dimensions.dfa(signal):.4f}")
print(f"  Box-Counting Dimension             : {dimensions.box_counting(signal):.4f}")
print(f"  Haar Wavelet Dimension             : {dimensions.wavelet_dim(signal, num_levels=6):.4f}")
print(f"  Spatial Lacunarity (Gappiness)     : {dimensions.lacunarity(signal, box_size=8):.4f}")
print(f"  Renyi Generalized Dim (q=2.0)      : {dimensions.renyi_dim(signal, q=2.0):.4f}")

lyapunov_exp = np.array([0.5, 0.2, -0.1, -1.0])
print(f"  Lyapunov Dim (Kaplan-Yorke)        : {dimensions.lyapunov_dim(lyapunov_exp):.4f}")

mfs = dimensions.multifractal_spectrum(signal, q_min=-4.0, q_max=4.0, num_q=17)
print(f"  Multifractal Spectrum Points       : {len(mfs['alpha'])} alpha values computed")
print(f"  Singularity Spectrum Alpha Range   : [{min(mfs['alpha']):.3f}, {max(mfs['alpha']):.3f}]")

# ─── 4. Fractal Neural Networks ───────────────────────────────────────────────
print("\n[3. Fractal Neural Networks]")

# FDNN (Fractal-Dendritic Neural Network)
cfg_fdnn = neural.fdnn_config(input_dim=64, branch_dim=32, output_dim=8, num_branches=4)
state_fdnn = neural.fdnn_init(cfg_fdnn, seed=42)
x_in = rng.random(64)
out_fdnn = neural.fdnn_forward(x_in, state_fdnn, cfg_fdnn)
fl = neural.fdnn_fractal_loss(x_in, state_fdnn, cfg_fdnn)
print(f"  FDNN 8-dim Output                  : {[round(v, 4) for v in out_fdnn]}")
print(f"  FDNN Fractal Regularization Loss   : {fl:.6f}")

# MSFCN (Multi-Scale Fractal Convolutional Network)
cfg_msfcn = neural.msfcn_config(kernel_size=8, num_scales=4, base_scale=2.0)
state_msfcn = neural.msfcn_init(cfg_msfcn, seed=42)
sig_msfcn = rng.random(128)
out_msfcn = neural.msfcn_forward(sig_msfcn, state_msfcn, cfg_msfcn)
sc_loss = neural.msfcn_scale_consistency_loss(sig_msfcn, state_msfcn, cfg_msfcn)
print(f"  MSFCN Feature Vector Length        : {len(out_msfcn)}")
print(f"  MSFCN Scale Consistency Loss       : {sc_loss:.6f}")

# SNN-FT (Spiking Neural Network with Fractal Trees)
cfg_snn = neural.snnft_config(input_dim=16, num_compartments=8)
state_snn = neural.snnft_init(cfg_snn, seed=42)
T = 20
total_spikes = 0
for t in range(T):
    inp = (rng.random(16) > 0.8).astype(np.float64)
    out_snn = neural.snnft_step(inp, state_snn, cfg_snn)
    total_spikes += int(out_snn[0])
    neural.snnft_stdp_update(state_snn, cfg_snn, pre_spikes=inp, post_spikes=out_snn)

energy_loss = neural.snnft_energy_loss([total_spikes], cfg_snn)
print(f"  SNN-FT Output Spikes in {T} Steps   : {total_spikes}")
print(f"  SNN-FT Energy Regularization Loss  : {energy_loss:.6f}")

print("\n" + "=" * 60)
print("Fractalich quickstart demo completed successfully!")
print("=" * 60)
