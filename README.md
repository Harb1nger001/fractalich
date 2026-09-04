<div align="center">

# Fractalich

### High-Performance Fractal Data Analysis & Bio-Inspired Neural Architectures in C++17 and Python

[![PyPI Version](https://img.shields.io/pypi/v/fractalich.svg?color=blue)](https://pypi.org/project/fractalich/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fractalich.svg)](https://pypi.org/project/fractalich/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bindings: Pybind11](https://img.shields.io/badge/bindings-pybind11-red.svg)](https://github.com/pybind/pybind11)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/fractalich/fractalich)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fractalich.svg)](https://pypi.org/project/fractalich/)

<p align="center">
  <a href="#installation"><b>Installation</b></a> •
  <a href="#quick-start"><b>Quick Start</b></a> •
  <a href="#evaluation-metrics-fractalichmetrics"><b>Metrics API</b></a> •
  <a href="#fractal-dimension-estimators-fractalichdimensions"><b>Dimension Suite</b></a> •
  <a href="#fractal-neural-networks-fractalichneural"><b>Neural Networks</b></a> •
  <a href="#end-to-end-workflows--recipes"><b>Workflows</b></a>
</p>

</div>

---

## ⚡ Quick PyPI Install

**Fractalich** is available on PyPI! Install it directly using `pip`:

```bash
pip install fractalich
```

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation Guide](#installation-guide)
  - [Install from PyPI](#install-from-pypi)
  - [Building from Source](#building-from-source)
  - [Prerequisites](#prerequisites)
  - [Verifying Installation](#verifying-installation)
- [Quick Start](#quick-start)
- [Core Modules & Detailed API Reference](#core-modules--detailed-api-reference)
  - [1. Evaluation Metrics (`fractalich.metrics`)](#1-evaluation-metrics-fractalichmetrics)
    - [Fractal Fidelity Score (FFS)](#fractal-fidelity-score-ffs)
    - [Multiscale Structural Consistency (MSC)](#multiscale-structural-consistency-msc)
    - [Fractal Residual Error (FRE)](#fractal-residual-error-fre)
    - [Fractal Feature Relevance Index (FFRI)](#fractal-feature-relevance-index-ffri)
    - [Self-Similarity Retention Index (SSRI)](#self-similarity-retention-index-ssri)
    - [Fractal Recurrence Index (FRI)](#fractal-recurrence-index-fri)
  - [2. Fractal Dimension Estimators (`fractalich.dimensions`)](#2-fractal-dimension-estimators-fractalichdimensions)
    - [Time Series & Biomedical Waveforms](#category-a-time-series--biomedical-waveforms)
    - [Long-Range Memory & Scaling Dynamics](#category-b-long-range-memory--scaling-dynamics)
    - [Information & Multifractal Theory](#category-c-information--multifractal-theory)
    - [Geometric, Covering & Morphometry](#category-d-geometric-covering--morphometry)
    - [Spatial Heterogeneity & Dynamical Systems](#category-e-spatial-heterogeneity--dynamical-systems)
  - [3. Fractal Neural Networks (`fractalich.neural`)](#3-fractal-neural-networks-fractalichneural)
    - [FDNN: Fractal-Dendritic Neural Network](#1-fdnn-fractal-dendritic-neural-network)
    - [DDAN: Dynamic Dendritic Attention Network](#2-ddan-dynamic-dendritic-attention-network)
    - [MSFCN: Multi-Scale Fractal Convolutional Network](#3-msfcn-multi-scale-fractal-convolutional-network)
    - [SNN-FT: Spiking Neural Network with Fractal Trees](#4-snn-ft-spiking-neural-network-with-fractal-trees)
- [End-to-End Workflows & Recipes](#end-to-end-workflows--recipes)
  - [Biomedical Signal Complexity (EEG/ECG)](#workflow-1-biomedical-signal-complexity-eegecg)
  - [Multifractal Financial Volatility & Regime Detection](#workflow-2-multifractal-financial-market-analysis)
  - [Spatial Texture & Porosity Analysis](#workflow-3-spatial-texture-lacunarity--wavelet-analysis)
  - [Custom Neural Training Loop with Fractal Regularization](#workflow-4-training-with-fractal-regularization-losses)
- [Performance & C++ Architecture](#performance--c-architecture)
- [Project Architecture](#project-architecture)
- [Running the Test Suite](#running-the-test-suite)
- [Contributing](#contributing)
- [License & Citation](#license--citation)

---

## Overview

Fractal geometry, non-integer scaling dimensions, and power-law dynamics emerge naturally in complex systems: from **neural dendritic arbors** and **biomedical oscillations (EEG, ECG, EMG)** to **turbulent fluid flow**, **financial tick data**, **porous geomaterials**, and **geophysical phenomena**.

Standard statistical metrics (such as variance, standard Fourier spectral density, or simple autocorrelation) fail when signals exhibit long-range temporal memory, multi-scale self-similarity, or non-stationary fractional Brownian scaling.

**Fractalich** bridges rigorous mathematical fractal geometry with modern high-throughput scientific computing and machine learning:
1. **Zero-Copy C++17 Engine**: High-performance, vectorized C++ core operating directly on contiguous NumPy array buffers.
2. **Exhaustive Estimator Suite**: 17 dimension estimators spanning geometric covering, time series roughness, dynamical chaos, and multifractal singularity spectrums.
3. **Standardized Evaluation Metrics**: Mathematical scoring functions for reconstruction fidelity, cross-scale consistency, and mutual-information feature ranking.
4. **Dendritic & Fractal Neural Networks**: Architectures leveraging scale-invariance, recursive dendritic branching, multi-scale IFS convolutions, and scale-modulated STDP spiking dynamics.

---

## Key Features

| Capability | Description |
| :--- | :--- |
| ⚡ **Vectorized C++17 Core** | Native multi-threaded and vectorized C++17 implementation for ultra-low latency. |
| 🐍 **Seamless Python Bindings** | Built with [Pybind11](https://github.com/pybind/pybind11); pass NumPy ndarrays directly without memory duplication. |
| 📐 **17 Fractal Estimators** | Box-Counting, Higuchi, Katz, Petrosian, DFA, Multifractal Spectrum $f(\alpha)$, Wavelet, Lacunarity, Hurst R/S, Rényi, Correlation, Hausdorff, Packing, Lyapunov, Information, Minkowski, and Temporal Graph Dimension. |
| 📊 **6 Quantitative Metrics** | FFS (Fractal Fidelity), MSC (Multiscale Structural Consistency), FRE (Residual Error), FFRI (Relevance Index), SSRI (Self-Similarity Retention), and FRI (Recurrence Index). |
| 🧠 **4 Fractal Neural Paradigms** | **FDNN** (Recursive dendritic branching), **DDAN** (Dendritic graph attention), **MSFCN** (IFS dilated multiscale convolutions), and **SNN-FT** (Fractal-compartment LIF neurons with STDP). |

---

## Installation Guide

### Install from PyPI

Fractalich is published on PyPI. You can install it directly via `pip`:

```bash
pip install fractalich
```

### Building from Source

If you wish to compile from source or contribute to development:

```bash
# 1. Clone the repository
git clone https://github.com/fractalich/fractalich.git
cd fractalich

# 2. Install build dependencies
pip install --upgrade setuptools wheel cmake numpy pytest

# 3. Build and install the package
pip install .

# Or for editable/development mode:
pip install -e .
```

### Prerequisites

| Component | Minimum Requirement | Notes |
| :--- | :--- | :--- |
| **Python** | `>= 3.8` | Tested on Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 |
| **C++ Compiler** | C++17 compliant | GCC 9+, Clang 10+, or MSVC 2019+ (Visual Studio 16.0+) |
| **CMake** | `>= 3.18` | Automatically invokes Pybind11 fetch or local resolution |
| **NumPy** | `>= 1.21` | Supports any NumPy `float64` array |

### Verifying Installation

Verify that the C++ extension loads properly:

```bash
python -c "import fractalich; print(f'Fractalich v{fractalich.__version__} successfully loaded!')"
```

---

## Quick Start

Here is a 60-second end-to-end example demonstrating dimension estimation, evaluation metrics, and a fractal-dendritic forward pass:

```python
import numpy as np
from fractalich import dimensions, metrics, neural

# 1. Generate synthetic fractional Brownian motion (rough random walk)
rng = np.random.default_rng(42)
signal = np.cumsum(rng.standard_normal(2048))

# 2. Extract Fractal & Scaling Dimensions
h_rs      = dimensions.hurst(signal)               # Long-range memory (R/S)
dfa_alpha = dimensions.dfa(signal)                 # Detrended Fluctuation Analysis
h_dim     = dimensions.higuchi(signal, k_max=16)   # Higuchi waveform roughness
katz_dim  = dimensions.katz(signal)                # Katz waveform dimension
box_dim   = dimensions.box_counting(signal)        # Geometric box-counting

print("=== Fractal Dimensions ===")
print(f"Hurst Exponent (R/S)       : {h_rs:.4f}  (>0.5 indicates persistence)")
print(f"DFA Scaling Alpha          : {dfa_alpha:.4f}")
print(f"Higuchi Fractal Dimension  : {h_dim:.4f}")
print(f"Katz Dimension             : {katz_dim:.4f}")
print(f"Box-Counting Dimension     : {box_dim:.4f}")

# 3. Assess Reconstruction Fidelity with Metrics
noisy_recon = signal + rng.normal(0, 0.05 * np.std(signal), size=len(signal))
fidelity    = metrics.ffs(signal, noisy_recon)
print(f"\nFractal Fidelity Score (FFS): {fidelity:.6f} / 1.000000")

# 4. Process Through a Fractal-Dendritic Neural Network (FDNN)
cfg   = neural.fdnn_config(input_dim=64, branch_dim=32, output_dim=8, num_branches=4)
state = neural.fdnn_init(cfg, seed=42)

input_sample = signal[:64]
soma_out = neural.fdnn_forward(input_sample, state, cfg)
reg_loss = neural.fdnn_fractal_loss(input_sample, state, cfg)

print(f"\nFDNN 8-dim Somatic Output  : {[round(v, 4) for v in soma_out]}")
print(f"FDNN Fractal Consistency Loss: {reg_loss:.6f}")
```

---

## Core Modules & Detailed API Reference

```
fractalich/
├── metrics      ──> Evaluation & scoring functions (FFS, MSC, FRE, FFRI, SSRI, FRI)
├── dimensions   ──> 17 fractal, multifractal, and dynamical dimension algorithms
└── neural       ──> Bio-inspired & multiscale architectures (FDNN, DDAN, MSFCN, SNN-FT)
```

---

### 1. Evaluation Metrics (`fractalich.metrics`)

The `fractalich.metrics` module provides specialized statistical and information-theoretic metrics designed specifically for fractal, multiscale, and recurrent signals.

```python
from fractalich import metrics
```

#### Fractal Fidelity Score (`FFS`)
Quantifies how accurately a reconstructed signal $X_f$ preserves the energy and morphology of the reference signal $X_o$.

$$\text{FFS} = 1 - \frac{\|X_f - X_o\|_2}{\|X_o\|_2}$$

- **Range**: $(-\infty, 1.0]$. A score of `1.0` indicates an exact match.
- **Usage**:
  ```python
  import numpy as np
  from fractalich import metrics

  orig = np.sin(np.linspace(0, 20, 1000))
  noisy_rec = orig + np.random.normal(0, 0.02, size=1000)

  score = metrics.ffs(orig, noisy_rec)
  print(f"FFS: {score:.5f}")  # e.g., 0.98521
  ```

---

#### Multiscale Structural Consistency (`MSC`)
Computes the mean structural similarity index (SSIM) across multi-resolution decompositions or scale hierarchies.

$$\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \text{SSIM}(S_i, S'_i)$$

- **Range**: $[0.0, 1.0]$. Higher values indicate superior preservation of structural features across all scales.
- **Usage**:
  ```python
  from fractalich import metrics
  import numpy as np

  # Decompose signal into 4 sub-scale resolutions
  scale_orig = [np.random.randn(256) for _ in range(4)]
  scale_rec  = [s + np.random.normal(0, 0.05, size=256) for s in scale_orig]

  msc_val = metrics.msc(scale_orig, scale_rec)
  print(f"Multiscale Structural Consistency: {msc_val:.4f}")
  ```

---

#### Fractal Residual Error (`FRE`)
Measures the absolute discrepancy between empirically measured fractal dimensions $\mathbf{D}$ and model-predicted dimensions $\mathbf{D}'$.

$$\text{FRE} = \sum_{i} |D_i - D'_i|$$

- **Range**: $[0, \infty)$. Lower values denote higher accuracy.
- **Usage**:
  ```python
  from fractalich import metrics
  import numpy as np

  measured  = np.array([1.45, 1.62, 1.88])
  predicted = np.array([1.42, 1.65, 1.80])

  error = metrics.fre(measured, predicted)
  print(f"Fractal Residual Error: {error:.4f}")  # 0.1400
  ```

---

#### Fractal Feature Relevance Index (`FFRI`)
Measures the mutual information $I(f; y)$ between a fractal feature $f$ and discrete target labels $y$, normalized by the Shannon entropy $H(f)$ of the feature.

$$\text{FFRI} = \frac{I(f; y)}{H(f)}$$

- **Range**: $[0.0, 1.0]$. Higher values indicate features that are more informative for downstream classification tasks.
- **Usage**:
  ```python
  from fractalich import metrics
  import numpy as np

  # 500 samples with class labels (0 or 1)
  feature = np.random.randn(500)
  labels  = (feature > 0.2).astype(np.int32)

  relevance = metrics.ffri(feature, labels, num_bins=50)
  print(f"Feature Relevance (FFRI): {relevance:.4f}")
  ```

---

#### Self-Similarity Retention Index (`SSRI`)
Evaluates the stability and variance of Relative Fractal Dimension (RFD) across multiple scales.

$$\text{SSRI} = 1 - \frac{\text{Var}(\text{RFD}(s))}{\text{Mean}(\text{RFD}(s))}$$

- **Range**: Higher denotes stable self-similar scaling across resolution levels.
- **Usage**:
  ```python
  from fractalich import metrics
  import numpy as np

  rfd_per_scale = np.array([1.51, 1.49, 1.50, 1.52, 1.50])
  ssri_score = metrics.ssri(rfd_per_scale)
  print(f"Self-Similarity Retention (SSRI): {ssri_score:.5f}")
  ```

---

#### Fractal Recurrence Index (`FRI`)
Computes the empirical probability that pattern match scores exceed a threshold $\theta$ over temporal sequences.

$$\text{FRI} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}_{\{\text{Match}(t) > \theta\}}$$

- **Range**: $[0.0, 1.0]$.
- **Usage**:
  ```python
  from fractalich import metrics
  import numpy as np

  match_scores = np.random.uniform(0, 1, size=500)
  fri_val = metrics.fri(match_scores, theta=0.7)
  print(f"Fractal Recurrence Rate: {fri_val:.2%}")
  ```

---

### 2. Fractal Dimension Estimators (`fractalich.dimensions`)

Fractalich implements 17 distinct dimension estimation algorithms across five scientific categories:

```python
from fractalich import dimensions
```

```
                                  ┌─ Category A: Time Series & Biomedical (Higuchi, Katz, Petrosian, Temporal)
                                  ├─ Category B: Long-Range Memory & Scaling (Hurst R/S, DFA)
Fractal Dimension Estimators ────┼─ Category C: Information & Multifractal (Information, Rényi D_q, Multifractal Spectrum)
                                  ├─ Category D: Geometric & Covering (Box-Counting, Hausdorff, Packing, Minkowski)
                                  └─ Category E: Spatial & Dynamical Systems (Wavelet, Lacunarity, Correlation, Lyapunov)
```

---

#### Category A: Time Series & Biomedical Waveforms

Ideal for EEG, ECG, EMG, seismic vibrations, and speech signals.

| Function | Method | Signature | Recommended Domain |
| :--- | :--- | :--- | :--- |
| `higuchi(signal, k_max=10)` | Higuchi Fractal Dimension | `higuchi(signal: ndarray, k_max: int = 10) -> float` | Biomedical time series, EEG depth of anesthesia, cognitive load |
| `katz(signal)` | Katz Waveform Dimension | `katz(signal: ndarray) -> float` | Ultra-fast real-time EMG & robotic prosthetics control |
| `petrosian(signal)` | Petrosian (Zero-Crossing) | `petrosian(signal: ndarray) -> float` | Epileptic seizure detection, brain state transition segmentation |
| `temporal_fractal_dim(signal, dt=1.0)` | Temporal Graph Dimension | `temporal_fractal_dim(signal: ndarray, dt: float = 1.0) -> float` | Sensor stream trajectory complexity & traffic flow |

```python
from fractalich import dimensions
import numpy as np

# Simulate an EEG burst (500 Hz signal)
t = np.linspace(0, 2.0, 1000)
eeg = np.sin(2 * np.pi * 10 * t) + 0.4 * np.random.randn(1000)

d_higuchi   = dimensions.higuchi(eeg, k_max=16)
d_katz      = dimensions.katz(eeg)
d_petrosian = dimensions.petrosian(eeg)
d_temporal  = dimensions.temporal_fractal_dim(eeg, dt=0.002)

print(f"Higuchi Dimension  : {d_higuchi:.4f}")
print(f"Katz Dimension     : {d_katz:.4f}")
print(f"Petrosian Dimension: {d_petrosian:.4f}")
print(f"Temporal Dimension : {d_temporal:.4f}")
```

---

#### Category B: Long-Range Memory & Scaling Dynamics

Quantifies persistence, anti-persistence (mean-reversion), and non-stationary scaling behavior.

| Function | Method | Signature | Output Interpretation |
| :--- | :--- | :--- | :--- |
| `hurst(signal)` | Rescaled Range ($R/S$) | `hurst(signal: ndarray) -> float` | $H > 0.5$: Persistent memory; $H < 0.5$: Anti-persistent; $H=0.5$: White noise |
| `dfa(signal, scales=None)` | Detrended Fluctuation Analysis | `dfa(signal: ndarray, scales: list[int] = None) -> float` | $\alpha \approx 0.5$: White noise; $\alpha \approx 1.0$: $1/f$ pink noise; $\alpha \approx 1.5$: Brownian motion |

```python
from fractalich import dimensions
import numpy as np

# Persistent random walk (trend-following)
walk = np.cumsum(np.random.randn(4096))

h_val = dimensions.hurst(walk)
dfa_a = dimensions.dfa(walk, scales=[8, 16, 32, 64, 128, 256])

print(f"Hurst Exponent (R/S) : {h_val:.4f} ({'Persistent' if h_val > 0.5 else 'Mean-reverting'})")
print(f"DFA Scaling Exponent : {dfa_a:.4f}")
```

---

#### Category C: Information & Multifractal Theory

Characterizes complex signals governed by multiple scaling exponents rather than a single global fractal dimension.

| Function | Method | Signature | Key Parameters |
| :--- | :--- | :--- | :--- |
| `information_dim(signal, num_bins=64)` | Information Dimension | `information_dim(signal: ndarray, num_bins: int = 64) -> float` | `num_bins`: histogram resolution |
| `renyi_dim(signal, q=2.0, num_bins=64)` | Generalized Rényi ($D_q$) | `renyi_dim(signal: ndarray, q: float = 2.0, num_bins: int = 64) -> float` | $q=0$: Capacity, $q=1$: Info, $q=2$: Correlation |
| `multifractal_spectrum(signal, ...)` | Singularity Spectrum $f(\alpha)$ | `multifractal_spectrum(signal, q_min=-5.0, q_max=5.0, num_q=21, num_scales=10) -> dict` | Returns `{"alpha": [...], "f": [...]}` |

```python
from fractalich import dimensions
import numpy as np

# Turbulent velocity fluctuation series
turbulent = np.random.standard_cauchy(2048)

# 1. Compute Generalized Renyi spectrum D_q for q in [0, 1, 2]
d0 = dimensions.renyi_dim(turbulent, q=0.0) # Capacity dimension
d1 = dimensions.renyi_dim(turbulent, q=1.0) # Information dimension
d2 = dimensions.renyi_dim(turbulent, q=2.0) # Correlation dimension
print(f"Renyi Spectrum: D_0={d0:.3f}, D_1={d1:.3f}, D_2={d2:.3f}")

# 2. Extract Multifractal Singularity Spectrum f(alpha)
spectrum = dimensions.multifractal_spectrum(turbulent, q_min=-4.0, q_max=4.0, num_q=17)
alphas = spectrum["alpha"]
f_vals = spectrum["f"]

print(f"Extracted {len(alphas)} points on the singularity spectrum curve.")
print(f"Singularity Width (Delta-alpha): {max(alphas) - min(alphas):.4f}")
```

---

#### Category D: Geometric, Covering & Morphometry

Estimates spatial footprint, geometric roughness, and packing bounds.

| Function | Method | Signature | Applications |
| :--- | :--- | :--- | :--- |
| `box_counting(signal, ...)` | Box-Counting | `box_counting(signal: ndarray, min_box=2, max_box=64, num_scales=10) -> float` | 1D curves, boundary profiles |
| `hausdorff(points, D=2, ...)` | Hausdorff Dimension | `hausdorff(points: ndarray, D: int = 2, min_box=2, max_box=64, num_scales=10) -> float` | $N \times D$ Point clouds, strange attractors |
| `packing_dim(signal, ...)` | Packing Dimension | `packing_dim(signal: ndarray, min_box=2, max_box=64, num_scales=10) -> float` | Porous media, sparse structures |
| `minkowski_bouligand(signal, ...)` | Minkowski-Bouligand | `minkowski_bouligand(signal: ndarray, min_box=2, max_box=64, num_scales=10) -> float` | Coastlines, crack morphology |

```python
from fractalich import dimensions
import numpy as np

# 2D Point cloud on a fractal boundary (e.g., Sierpinski / Cantor in 2D)
points_2d = np.random.uniform(-1, 1, size=(500, 2))

d_box       = dimensions.box_counting(points_2d[:, 0])
d_hausdorff = dimensions.hausdorff(points_2d, D=2, min_box=4, max_box=64)
d_packing   = dimensions.packing_dim(points_2d[:, 0])
d_minkowski = dimensions.minkowski_bouligand(points_2d[:, 0])

print(f"Box-Counting       : {d_box:.4f}")
print(f"Hausdorff (2D)     : {d_hausdorff:.4f}")
print(f"Packing Dimension  : {d_packing:.4f}")
print(f"Minkowski-Bouligand: {d_minkowski:.4f}")
```

---

#### Category E: Spatial Heterogeneity & Dynamical Systems

For texture analysis, state-space attractors, and multiresolution wavelet energy decay.

| Function | Method | Signature | Use Case |
| :--- | :--- | :--- | :--- |
| `wavelet_dim(signal, num_levels=6)` | Haar Wavelet Scaling | `wavelet_dim(signal: ndarray, num_levels: int = 6) -> float` | Wavelet multiresolution energy scaling |
| `lacunarity(signal, box_size=8)` | Spatial Lacunarity | `lacunarity(signal: ndarray, box_size: int = 8) -> float` | Spatial gappiness, porous heterogeneity, texture |
| `correlation_dim(data, embedding_dim=3, tau=1)` | Grassberger-Procaccia | `correlation_dim(data: ndarray, embedding_dim: int = 3, tau: int = 1) -> float` | Chaotic dynamics, phase space reconstruction |
| `lyapunov_dim(exponents)` | Kaplan-Yorke Dimension | `lyapunov_dim(exponents: ndarray) -> float` | Strange attractor dimension from Lyapunov spectrum |

```python
from fractalich import dimensions
import numpy as np

# Chaotic time series (e.g. Lorenz-like sequence)
chaos_data = np.sin(np.linspace(0, 50, 1024)) + np.cos(np.linspace(0, 100, 1024))

d_corr = dimensions.correlation_dim(chaos_data, embedding_dim=3, tau=2)
d_wave = dimensions.wavelet_dim(chaos_data, num_levels=6)
lacuna = dimensions.lacunarity(chaos_data, box_size=16)

# Kaplan-Yorke from sorted Lyapunov spectrum [lambda_1 >= lambda_2 >= ...]
lyap_exps = np.array([0.41, 0.00, -0.75, -1.20])
d_lyap = dimensions.lyapunov_dim(lyap_exps)

print(f"Correlation Dimension (GP) : {d_corr:.4f}")
print(f"Haar Wavelet Dimension     : {d_wave:.4f}")
print(f"Spatial Lacunarity (gaps)  : {lacuna:.4f}")
print(f"Kaplan-Yorke Lyapunov Dim  : {d_lyap:.4f}")
```

---

### 3. Fractal Neural Networks (`fractalich.neural`)

Fractalich introduces four specialized neural network models based on fractal and dendritic principles:

```python
from fractalich import neural
```

---

#### 1. FDNN: Fractal-Dendritic Neural Network

Models biological dendritic trees as recursively self-similar computational subunits:
$$y_b = \phi\left(W_b \mathcal{S}_s[x] + b_b\right)$$
Outputs of all dendritic branches are gated and integrated somatically.

```python
from fractalich import neural
import numpy as np

# 1. Configure the FDNN architecture
config = neural.fdnn_config(
    input_dim=64,           # Dimensionality of input signal
    branch_dim=32,          # Hidden dimensionality per dendritic branch
    output_dim=16,          # Output dimension after somatic integration
    num_branches=4,         # Number of parallel dendritic branches
    recursion_depth=2,      # Self-similar fractal recursion depth
    scale_factor=2.0,       # Scale dilation factor s
    lambda_fract=0.01       # Fractal consistency regularization weight
)

# 2. Initialize weights with seed
state = neural.fdnn_init(config, seed=42)

# 3. Forward pass
x_input = np.random.randn(64)
soma_output = neural.fdnn_forward(x_input, state, config)

# 4. Compute fractal consistency regularization loss
loss_fractal = neural.fdnn_fractal_loss(x_input, state, config)

print(f"FDNN Output Shape : {len(soma_output)}")
print(f"Fractal Loss      : {loss_fractal:.6f}")
```

---

#### 2. DDAN: Dynamic Dendritic Attention Network

Represents the dendritic arbor as a directed graph of branching paths. An RNN-style path encoder embeds each path, followed by depth-aware query-key attention:
$$\alpha_p = \text{softmax}\left(\frac{\text{sim}(q, k_p)}{\sqrt{d}}\right)_p, \quad \mathbf{o} = \sum_{p} \alpha_p v_p$$

```python
from fractalich import neural
import numpy as np

# 1. Configure DDAN
config = neural.ddan_config(
    hidden_dim=32,          # Key / Value embedding dimension
    num_paths=4,            # Total dendritic paths in tree
    path_len=3,             # Depth / nodes per path
    lambda_sparse=0.001     # Sparsity penalty
)
state = neural.ddan_init(config, seed=42)

# 2. Define Query and Dendritic Paths (P x L x hidden_dim)
query = np.random.randn(32)
paths = [
    [np.random.randn(32) for _ in range(3)]  # Path 0: 3 nodes
    for _ in range(4)                        # 4 paths
]
path_depths = [1, 2, 2, 3]                  # Depth level for each path

# 3. Attended forward pass
attended_output = neural.ddan_forward(query, paths, path_depths, state, config)
print(f"DDAN Output vector (length {len(attended_output)}): {[round(v, 4) for v in attended_output[:4]]}...")
```

---

#### 3. MSFCN: Multi-Scale Fractal Convolutional Network

Performs convolutions across fractally-dilated kernels governed by an Iterated Function System (IFS):
$$K_s(u) = \frac{1}{\alpha_s} K_0\left(\frac{u}{s}\right)$$

```python
from fractalich import neural
import numpy as np

# 1. Configure MSFCN
config = neural.msfcn_config(
    kernel_size=8,          # Base kernel size
    num_scales=4,           # Number of fractal scaling levels
    base_scale=2.0,         # Multiplier between levels
    lambda_scale=0.01,      # Scale-consistency regularization penalty
    num_ifs_terms=4         # Number of IFS affine maps
)
state = neural.msfcn_init(config, seed=42)

# 2. Convolve across scales and aggregate
signal = np.random.randn(256)
features = neural.msfcn_forward(signal, state, config)
scale_loss = neural.msfcn_scale_consistency_loss(signal, state, config)

print(f"MSFCN Feature Vector Length : {len(features)}")
print(f"Scale Consistency Loss      : {scale_loss:.6f}")
```

---

#### 4. SNN-FT: Spiking Neural Network with Fractal Trees

Multi-compartment Leaky Integrate-and-Fire (LIF) neurons structured via IFS fractal adjacency matrices, featuring scale-modulated Spike-Timing-Dependent Plasticity (STDP):

$$\tau_c \frac{dV_c}{dt} = -V_c + \sum_{j} w_{jc} s_j(t), \quad \tau_m \frac{dV_{\text{soma}}}{dt} = -V_{\text{soma}} + \sum_{c} G_c f_c(V_c)$$

```python
from fractalich import neural
import numpy as np

# 1. Configure SNN-FT
config = neural.snnft_config(
    num_compartments=8,     # Compartments per neuron
    input_dim=16,           # Input spike dimension
    tau_c=5.0,              # Compartment time constant (ms)
    tau_m=10.0,             # Somatic time constant (ms)
    v_th=1.0,               # Membrane threshold
    v_reset=0.0,            # Post-spike reset
    dt=0.1,                 # Integration step
    A_plus=0.01,            # STDP potentiation amplitude
    A_minus=0.012,          # STDP depression amplitude
    eta_scale=1.0,          # Scale-modulation exponent
    lambda_energy=1e-4      # Metabolic energy penalty
)
state = neural.snnft_init(config, seed=42)

# 2. Simulate 50 time steps of Poisson spike train
total_spikes = 0
spike_history = []

for step in range(50):
    # Binary input spike vector
    pre_spikes = (np.random.rand(16) > 0.75).astype(np.float64)
    
    # Forward simulation step
    post_spikes = neural.snnft_step(pre_spikes, state, config)
    total_spikes += int(post_spikes[0])
    spike_history.append(int(post_spikes[0]))
    
    # In-place scale-modulated STDP weight adaptation
    neural.snnft_stdp_update(state, config, pre_spikes, post_spikes)

# 3. Metabolic energy regularization loss
energy_loss = neural.snnft_energy_loss([total_spikes], config)

print(f"SNN-FT Spikes generated in 50 steps: {total_spikes}")
print(f"Metabolic Energy Regularization Loss : {energy_loss:.6f}")
```

---

## End-to-End Workflows & Recipes

### Workflow 1: Biomedical Signal Complexity (EEG/ECG)

Build an automated feature extraction and evaluation pipeline for EEG/ECG anomaly detection:

```python
import numpy as np
from fractalich import dimensions, metrics

def extract_biomedical_fractal_features(signal_epoch: np.ndarray) -> np.ndarray:
    """Extract a rich 6-dimensional fractal fingerprint from a biological signal."""
    return np.array([
        dimensions.higuchi(signal_epoch, k_max=16),
        dimensions.katz(signal_epoch),
        dimensions.petrosian(signal_epoch),
        dimensions.dfa(signal_epoch),
        dimensions.hurst(signal_epoch),
        dimensions.wavelet_dim(signal_epoch, num_levels=5)
    ], dtype=np.float64)

# Simulate 200 EEG epochs (100 healthy, 100 seizure-state)
rng = np.random.default_rng(123)
epochs = []
labels = []

for i in range(200):
    if i < 100:
        # Healthy: complex, high fractal dimension
        epoch = np.cumsum(rng.standard_normal(512)) + rng.normal(0, 0.5, 512)
        labels.append(0)
    else:
        # Seizure: hypersynchronous, reduced complexity
        t = np.linspace(0, 10, 512)
        epoch = 3.0 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.1, 512)
        labels.append(1)
    epochs.append(epoch)

# Extract features
features = np.array([extract_biomedical_fractal_features(ep) for ep in epochs])
labels = np.array(labels, dtype=np.int32)

feature_names = ["Higuchi", "Katz", "Petrosian", "DFA", "Hurst", "Wavelet"]
print("=== Feature Relevance Index (FFRI) for Seizure Classification ===")
for col_idx, name in enumerate(feature_names):
    rel_score = metrics.ffri(features[:, col_idx], labels, num_bins=30)
    print(f"  {name:12s} : FFRI = {rel_score:.4f}")
```

---

### Workflow 2: Multifractal Financial Market Analysis

Analyze asset return persistence, heavy tails, and singularity spectrums:

```python
import numpy as np
from fractalich import dimensions

# Simulated financial log returns with volatility clustering
rng = np.random.default_rng(42)
innovations = rng.standard_t(df=3.5, size=4096)
returns = innovations * np.exp(np.sin(np.linspace(0, 20, 4096)) * 0.5)

# 1. Persistence & Memory
h_rs  = dimensions.hurst(returns)
alpha = dimensions.dfa(returns)

print("=== Market Memory & Scaling ===")
print(f"Hurst Index (R/S) : {h_rs:.4f} ({'Trend-following' if h_rs > 0.5 else 'Mean-reverting'})")
print(f"DFA Scaling Alpha : {alpha:.4f}")

# 2. Multifractal Singularity Spectrum
spec = dimensions.multifractal_spectrum(returns, q_min=-6.0, q_max=6.0, num_q=25)
alphas = spec["alpha"]
f_vals = spec["f"]

# Multifractality degree = width of the alpha singularity spectrum
spectrum_width = max(alphas) - min(alphas)
print(f"Singularity Spectrum Width : {spectrum_width:.4f}")
if spectrum_width > 0.4:
    print("Market regime is strongly MULTIFRACTAL (intermittent volatility).")
else:
    print("Market regime is MONOFRACTAL.")
```

---

### Workflow 3: Spatial Texture, Lacunarity & Wavelet Analysis

Characterize porosity, roughness, and texture gaps in materials or imaging:

```python
import numpy as np
from fractalich import dimensions

# Generate a heterogeneous porous cross-section profile
profile = (np.random.rand(1024) > 0.65).astype(np.float64)

# 1. Lacunarity across multiple box window sizes
for box_sz in [4, 8, 16, 32, 64]:
    lac = dimensions.lacunarity(profile, box_size=box_sz)
    print(f"Lacunarity (box size {box_sz:2d}) : {lac:.4f}")

# 2. Wavelet & Box Dimensions
w_dim = dimensions.wavelet_dim(profile, num_levels=7)
b_dim = dimensions.box_counting(profile)
print(f"Haar Wavelet Dimension    : {w_dim:.4f}")
print(f"Box-Counting Dimension    : {b_dim:.4f}")
```

---

### Workflow 4: Training with Fractal Regularization Losses

Incorporate Fractalich loss terms into custom machine learning loops:

```python
import numpy as np
from fractalich import neural, metrics

# Instantiate an MSFCN model
cfg = neural.msfcn_config(kernel_size=16, num_scales=4, base_scale=2.0, lambda_scale=0.05)
state = neural.msfcn_init(cfg, seed=1)

# Training batch simulation
batch_signals = np.random.randn(10, 128)
total_loss = 0.0

for signal in batch_signals:
    # Forward representation
    representation = neural.msfcn_forward(signal, state, cfg)
    
    # Scale-consistency regularization loss
    loss_scale = neural.msfcn_scale_consistency_loss(signal, state, cfg)
    
    # Mock task loss (e.g. MSE or cross-entropy)
    loss_task = np.mean(np.array(representation)**2)
    
    sample_loss = loss_task + cfg.lambda_scale * loss_scale
    total_loss += sample_loss

print(f"Mean Batch Loss with Scale Consistency: {total_loss / len(batch_signals):.6f}")
```

---

## Performance & C++ Architecture

Fractalich is built from the ground up for scientific performance:

1. **Zero-Copy NumPy Interop**: `pybind11::array_t<double>` buffers are inspected and traversed as raw contiguous pointers (`const double*`), avoiding allocations or data copying.
2. **Deterministic & Thread-Safe**: Core numerical routines do not mutate shared global state and can be called safely in multithreaded Python environments (`threading` / OpenMP).
3. **Optimized Linear Algebra**: Recursive tree walks and convolutions leverage cache-friendly localized buffers.

---

## Project Architecture

```
Fractalich/
├── CMakeLists.txt                 # Modern CMake build configuration
├── pyproject.toml                 # PEP 517/518 build specification
├── setup.py                       # Setuptools CMakeExtension wrapper
├── VERSION.txt                    # Project semantic version
├── LICENSE.txt                    # MIT License terms
├── README.md                      # Comprehensive documentation
├── include/                       # C++ Header declarations
│   ├── fractal_metrics.hpp        # 6 Evaluation metrics definitions
│   ├── fractal_dimensions.hpp     # 17 Fractal dimension estimators
│   └── fractal_nn.hpp             # FDNN, DDAN, MSFCN, SNN-FT headers
├── src/                           # High-performance C++17 implementations
│   ├── fractal_metrics.cpp        # Metric algorithms (FFS, MSC, FRE, FFRI, SSRI, FRI)
│   ├── fractal_dimensions.cpp     # Vectorized fractal dimension calculations
│   └── fractal_nn.cpp             # Neural forward, recurrence, and regularizers
├── bindings/                      # Pybind11 Python extension module
│   └── python_bindings.cpp        # C++ to Python type mappings & NumPy buffers
├── fractalich/                    # Python package wrapper
│   ├── __init__.py                # Package entrypoint & metadata re-exports
│   ├── metrics.py                 # Pythonic metrics API
│   ├── dimensions.py              # Pythonic dimension estimation API
│   └── neural.py                  # Pythonic neural networks API
├── examples/                      # Example workflows and quickstarts
│   └── quickstart.py              # Comprehensive quickstart demonstration
└── tests/                         # Automated test suite
    └── test_fractalich.py         # Complete suite with 30+ unit tests
```

---

## Running the Test Suite

Run the full automated test suite using `pytest`:

```bash
# Run all tests
python -m pytest tests/ -v

# Run only dimension tests
python -m pytest tests/test_fractalich.py -k "TestFractalDimensions" -v

# Run only neural network tests
python -m pytest tests/test_fractalich.py -k "FDNN or MSFCN or SNNFT" -v
```

---

## Contributing

Contributions, bug reports, and feature requests are welcome!

1. **Fork the Repository** on GitHub.
2. **Create a Feature Branch**: `git checkout -b feature/new-estimator`.
3. **Commit your changes**: `git commit -m "Add new fractal estimator"`.
4. **Push to the branch**: `git push origin feature/new-estimator`.
5. **Open a Pull Request**.

Please ensure that all tests pass (`pytest tests/`) and that any new C++ routines include corresponding Python bindings and unit tests.

---

## License & Citation

This project is licensed under the **MIT License** - see the [LICENSE.txt](LICENSE.txt) file for details.

### Citation

If you use **Fractalich** in your research or applications, please cite:

```bibtex
@software{fractalich2026,
  author = {Fractalich Contributors},
  title = {Fractalich: High-Performance Fractal Data Analysis and Neural Architectures},
  url = {https://github.com/fractalich/fractalich},
  version = {0.1.0},
  year = {2026}
}
```
