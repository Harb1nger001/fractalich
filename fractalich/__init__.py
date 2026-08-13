"""
Fractalich - Fractal-Based Data Analysis Library
=================================================
High-performance C++ backend with Python bindings via Pybind11.

Submodules:
    fractalich.metrics    – Evaluation metrics (FFS, MSC, FRE, FFRI, SSRI, FRI)
    fractalich.dimensions – 17 fractal dimension estimators
    fractalich.neural     – Neural network architectures (FDNN, DDAN, MSFCN, SNN-FT)

Usage:
    from fractalich import metrics, dimensions, neural
    import numpy as np

    x = np.random.rand(1024)
    print(metrics.ffs(x, x))      # 1.0
    print(dimensions.higuchi(x))  # ~1.5
"""

# Import the compiled C++ extension
try:
    from . import fractalich_ext as _ext
except ImportError as e:
    raise ImportError(
        "Fractalich C++ extension not found. "
        "Please build the library first: pip install ."
    ) from e

# Re-export high-level Python wrappers
from . import metrics, dimensions, neural

__version__: str = "0.1.0"
__library__: str = "Fractalich"

__all__ = ["metrics", "dimensions", "neural", "__version__", "__library__"]
