"""
fractalich.neural
=================
Python-level wrapper for Fractalich neural network architectures.
"""

from . import fractalich_ext as _ext
import numpy as np


# ─── FDNN ────────────────────────────────────────────────────────────────────
def fdnn_config(**kwargs):
    """Create and return an FDNNConfig with optional field overrides.

    Fields: input_dim, branch_dim, output_dim, num_branches,
            recursion_depth, scale_factor, lambda_fract.
    """
    cfg = _ext.neural.FDNNConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def fdnn_init(config=None, seed=42):
    """Initialise FDNN weights. Returns FDNNState."""
    if config is None:
        config = fdnn_config()
    return _ext.neural.fdnn_init(config, seed)


def fdnn_forward(x, state, config):
    """FDNN forward pass.

    Args:
        x      (ndarray): Input signal.
        state  (FDNNState): Weights from fdnn_init.
        config (FDNNConfig): Architecture config.

    Returns:
        list[float]: Somatic output vector.
    """
    return _ext.neural.fdnn_forward(np.asarray(x, np.float64), state, config)


def fdnn_fractal_loss(x, state, config):
    """Compute FDNN fractal consistency regularisation loss."""
    return _ext.neural.fdnn_fractal_loss(np.asarray(x, np.float64), state, config)


# ─── DDAN ────────────────────────────────────────────────────────────────────
def ddan_config(**kwargs):
    """Create and return a DDANConfig. Fields: hidden_dim, num_paths, path_len, lambda_sparse."""
    cfg = _ext.neural.DDANConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def ddan_init(config=None, seed=42):
    """Initialise DDAN weights. Returns DDANState."""
    if config is None:
        config = ddan_config()
    return _ext.neural.ddan_init(config, seed)


def ddan_forward(query, paths, path_depths, state, config):
    """DDAN attended forward pass.

    Args:
        query       (ndarray): Query vector.
        paths       (list of list of ndarray): Dendritic paths (P x L x hidden).
        path_depths (list of int): Depth of each path.
        state       (DDANState): Weights.
        config      (DDANConfig): Architecture config.

    Returns:
        list[float]: Attended output vector.
    """
    q = np.asarray(query, np.float64)
    py_paths = [[np.asarray(node, np.float64) for node in path] for path in paths]
    return _ext.neural.ddan_forward(q, py_paths, path_depths, state, config)


# ─── MSFCN ───────────────────────────────────────────────────────────────────
def msfcn_config(**kwargs):
    """Create and return an MSFCNConfig. Fields: kernel_size, num_scales, base_scale, lambda_scale, num_ifs_terms."""
    cfg = _ext.neural.MSFCNConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def msfcn_init(config=None, seed=42):
    """Initialise MSFCN kernel and aggregation weights. Returns MSFCNState."""
    if config is None:
        config = msfcn_config()
    return _ext.neural.msfcn_init(config, seed)


def msfcn_forward(signal, state, config):
    """MSFCN multi-scale convolution forward pass. Returns aggregated feature vector."""
    return _ext.neural.msfcn_forward(np.asarray(signal, np.float64), state, config)


def msfcn_scale_consistency_loss(signal, state, config):
    """MSFCN scale-consistency regularisation loss."""
    return _ext.neural.msfcn_scale_consistency_loss(np.asarray(signal, np.float64), state, config)


# ─── SNN-FT ──────────────────────────────────────────────────────────────────
def snnft_config(**kwargs):
    """Create and return an SNNFTConfig.

    Key fields: num_compartments, input_dim, tau_c, tau_m, v_th,
                v_reset, dt, A_plus, A_minus, tau_plus, tau_minus,
                eta_scale, num_ifs_terms, lambda_energy.
    """
    cfg = _ext.neural.SNNFTConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def snnft_init(config=None, seed=42):
    """Initialise SNN-FT state and fractal adjacency. Returns SNNFTState."""
    if config is None:
        config = snnft_config()
    return _ext.neural.snnft_init(config, seed)


def snnft_step(input_spikes, state, config):
    """Simulate one SNN-FT time step.

    Args:
        input_spikes (ndarray): Binary spike vector (float64).
        state        (SNNFTState): Mutable network state.
        config       (SNNFTConfig): Config.

    Returns:
        list[float]: Output spike vector (one value per soma).
    """
    return _ext.neural.snnft_step(np.asarray(input_spikes, np.float64), state, config)


def snnft_stdp_update(state, config, pre_spikes, post_spikes):
    """Apply scale-modulated STDP weight update in-place."""
    _ext.neural.snnft_stdp_update(
        state, config,
        np.asarray(pre_spikes, np.float64),
        np.asarray(post_spikes, np.float64)
    )


def snnft_energy_loss(spike_counts_per_window, config):
    """SNN-FT energy regularisation loss."""
    return _ext.neural.snnft_energy_loss(list(spike_counts_per_window), config)
