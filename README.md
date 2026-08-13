# Fractalich

**Fractalich** is a high-performance Fractal-Based Data Analysis Library with a C++ backend and Python bindings via Pybind11.

## Features

- **6 Evaluation Metrics**: FFS, MSC, FRE, FFRI, SSRI, FRI
- **17 Fractal Dimension Estimators**: Box-Counting, Hausdorff, Higuchi, Hurst, DFA, and more
- **4 Neural Network Architectures**: FDNN, DDAN, MSFCN, SNN-FT
- Fast C++ backend with a clean Pythonic API

## Installation

### Prerequisites
- CMake >= 3.18
- A C++17 compatible compiler (MSVC, GCC, Clang)
- Python >= 3.8
- pip

### Build and Install
```bash
cd Fractalich
pip install .
```

## Usage

```python
import numpy as np
from fractalich import metrics, dimensions, nn

# Evaluate fractal reconstruction fidelity
original = np.random.rand(1000)
reconstructed = original + np.random.rand(1000) * 0.01
print(metrics.ffs(original, reconstructed))

# Estimate fractal dimension
signal = np.random.rand(512)
print(dimensions.higuchi(signal, k_max=10))
```

## Version
See `VERSION.txt` for the current version.
