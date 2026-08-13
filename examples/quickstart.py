"""
Fractalich Quick-Start Example
================================
Run after building: pip install .
"""

import numpy as np
import sys
import os

# ─── Try to import Fractalich ─────────────────────────────────────────────────
try:
    from fractalich import fractalich_ext as _ext
except ImportError:
    print("ERROR: fractalich_ext is not built. Please run: pip install .")
    sys.exit(1)

rng = np.random.default_rng(42)

# ─── 1. Evaluation Metrics ────────────────────────────────────────────────────
print("=" * 50)
print("Fractalich v0.1.0 — Quick-Start Demo")
print("=" * 50)

original      = rng.random(1024)
reconstructed = original + rng.random(1024) * 0.05

ffs_score = _ext.metrics.ffs(original, reconstructed)
print(f"\n[Metrics]")
print(f"  FFS  (perfect=1.0)  : {ffs_score:.6f}")

scales_orig = [original[i*256:(i+1)*256] for i in range(4)]
scales_rec  = [reconstructed[i*256:(i+1)*256] for i in range(4)]
msc_score   = _ext.metrics.msc(scales_orig, scales_rec)
print(f"  MSC  (perfect=1.0)  : {msc_score:.6f}")

dims_meas = np.array([1.5, 1.7, 2.0])
dims_pred = np.array([1.4, 1.8, 1.9])
fre_score = _ext.metrics.fre(dims_meas, dims_pred)
print(f"  FRE  (lower=better) : {fre_score:.6f}")

labels_i  = rng.integers(0, 2, size=1024).astype(np.int32)
ffri_score = _ext.metrics.ffri(original, labels_i)
print(f"  FFRI (higher=more relevant): {ffri_score:.6f}")

rfd = rng.random(20)
ssri_score = _ext.metrics.ssri(rfd)
print(f"  SSRI (higher=better): {ssri_score:.6f}")

match_scores = rng.random(200)
fri_score = _ext.metrics.fri(match_scores, theta=0.5)
print(f"  FRI  (recurrence)   : {fri_score:.6f}")

# ─── 2. Fractal Dimensions ────────────────────────────────────────────────────
signal = rng.random(512)
print(f"\n[Fractal Dimensions]")
print(f"  Higuchi    : {_ext.dimensions.higuchi(signal, 10):.4f}")
print(f"  Katz       : {_ext.dimensions.katz(signal):.4f}")
print(f"  Petrosian  : {_ext.dimensions.petrosian(signal):.4f}")
print(f"  Hurst (R/S): {_ext.dimensions.hurst(signal):.4f}")
print(f"  DFA alpha  : {_ext.dimensions.dfa(signal):.4f}")
print(f"  Box-Count  : {_ext.dimensions.box_counting(signal):.4f}")
print(f"  Wavelet    : {_ext.dimensions.wavelet_dim(signal):.4f}")
print(f"  Lacunarity : {_ext.dimensions.lacunarity(signal):.4f}")
print(f"  Renyi(q=2) : {_ext.dimensions.renyi_dim(signal, q=2.0):.4f}")
lyapunov_exp = np.array([0.5, 0.2, -0.1, -1.0])
print(f"  Lyapunov   : {_ext.dimensions.lyapunov_dim(lyapunov_exp):.4f}")

mfs = _ext.dimensions.multifractal_spectrum(signal)
print(f"  Multifractal spectrum: {len(mfs['alpha'])} alpha values computed")

# ─── 3. Neural Networks ───────────────────────────────────────────────────────
print(f"\n[Neural Networks]")

# FDNN
cfg_fdnn = _ext.neural.FDNNConfig()
cfg_fdnn.input_dim = 64
cfg_fdnn.output_dim = 8
state_fdnn = _ext.neural.fdnn_init(cfg_fdnn, 0)
x_in = rng.random(64)
out_fdnn = _ext.neural.fdnn_forward(x_in, state_fdnn, cfg_fdnn)
fl = _ext.neural.fdnn_fractal_loss(x_in, state_fdnn, cfg_fdnn)
print(f"  FDNN output (8-d): {[round(v,4) for v in out_fdnn]}")
print(f"  FDNN fractal loss: {fl:.6f}")

# MSFCN
cfg_msfcn = _ext.neural.MSFCNConfig()
cfg_msfcn.kernel_size = 8
cfg_msfcn.num_scales  = 4
state_msfcn = _ext.neural.msfcn_init(cfg_msfcn, 0)
sig_msfcn   = rng.random(128)
out_msfcn   = _ext.neural.msfcn_forward(sig_msfcn, state_msfcn, cfg_msfcn)
sc_loss     = _ext.neural.msfcn_scale_consistency_loss(sig_msfcn, state_msfcn, cfg_msfcn)
print(f"  MSFCN output len : {len(out_msfcn)}")
print(f"  MSFCN scale loss : {sc_loss:.6f}")

# SNN-FT
cfg_snn = _ext.neural.SNNFTConfig()
cfg_snn.input_dim = 16
cfg_snn.num_compartments = 8
state_snn = _ext.neural.snnft_init(cfg_snn, 0)
T = 20
total_spikes = 0
for t in range(T):
    inp = (rng.random(16) > 0.8).astype(np.float64)
    out_snn = _ext.neural.snnft_step(inp, state_snn, cfg_snn)
    total_spikes += int(out_snn[0])
print(f"  SNN-FT spikes in {T} steps: {total_spikes}")

print("\nFractalich demo complete!")
