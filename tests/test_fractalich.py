"""
Tests for Fractalich library.

Run:
    python -m pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest


# Allow running tests directly without installing.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Native C++ extension detection
# ---------------------------------------------------------------------------

try:
    from fractalich import fractalich_ext as _ext

    HAS_EXT = True
    SKIP_MSG = ""

except ImportError:
    try:
        import fractalich_ext as _ext

        HAS_EXT = True
        SKIP_MSG = ""

    except ImportError:
        HAS_EXT = False
        SKIP_MSG = (
            "fractalich_ext not built. "
            "Run `pip install .` first."
        )


skip_if_no_ext = pytest.mark.skipif(
    not HAS_EXT,
    reason=SKIP_MSG,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def white_noise():
    rng = np.random.default_rng(0)
    return rng.random(512).astype(np.float64)


@pytest.fixture
def sine_wave():
    t = np.linspace(0, 4 * np.pi, 512)
    return np.sin(t).astype(np.float64)


@pytest.fixture
def constant():
    return np.ones(512, dtype=np.float64)


# ===========================================================================
# Evaluation Metrics
# ===========================================================================

class TestFFS:

    @skip_if_no_ext
    def test_perfect_reconstruction(self, white_noise):
        score = _ext.metrics.ffs(
            white_noise,
            white_noise,
        )

        assert abs(score - 1.0) < 1e-10

    @skip_if_no_ext
    def test_reconstruction_below_one_when_noisy(self, white_noise):
        rng = np.random.default_rng(1)

        noisy = (
            white_noise
            + rng.random(512) * 0.1
        )

        score = _ext.metrics.ffs(
            white_noise,
            noisy,
        )

        assert score < 1.0

    @skip_if_no_ext
    def test_mismatched_lengths_raises(self, white_noise):
        with pytest.raises(Exception):
            _ext.metrics.ffs(
                white_noise,
                white_noise[:100],
            )


class TestMSC:

    @skip_if_no_ext
    def test_identical_scales(self, white_noise):
        scales = [
            white_noise[i * 128:(i + 1) * 128]
            for i in range(4)
        ]

        score = _ext.metrics.msc(
            scales,
            scales,
        )

        assert abs(score - 1.0) < 1e-6


class TestFRE:

    @skip_if_no_ext
    def test_zero_error_identical(self):
        d = np.array(
            [1.5, 1.7, 2.0],
            dtype=np.float64,
        )

        assert _ext.metrics.fre(d, d) == pytest.approx(0.0)

    @skip_if_no_ext
    def test_nonzero_error(self):
        d = np.array(
            [1.5, 1.7, 2.0],
            dtype=np.float64,
        )

        dp = np.array(
            [1.0, 2.0, 1.5],
            dtype=np.float64,
        )

        assert _ext.metrics.fre(d, dp) == pytest.approx(1.3)


class TestFFRI:

    @skip_if_no_ext
    def test_random_returns_float(self, white_noise):
        rng = np.random.default_rng(2)

        labels = rng.integers(
            0,
            2,
            size=512,
            dtype=np.int32,
        )

        score = _ext.metrics.ffri(
            white_noise,
            labels,
        )

        assert isinstance(score, float)
        assert score >= 0.0

    @skip_if_no_ext
    def test_constant_feature_zero_entropy(self, constant):
        labels = np.zeros(
            512,
            dtype=np.int32,
        )

        score = _ext.metrics.ffri(
            constant,
            labels,
        )

        assert score == pytest.approx(0.0)


class TestSSRI:

    @skip_if_no_ext
    def test_constant_rfd_returns_one(self):
        rfd = np.ones(
            20,
            dtype=np.float64,
        )

        score = _ext.metrics.ssri(rfd)

        assert score == pytest.approx(1.0)

    @skip_if_no_ext
    def test_high_var_rfd_below_one(self):
        rfd = np.array(
            [1.0, 10.0, 0.5, 8.0, 2.0],
            dtype=np.float64,
        )

        score = _ext.metrics.ssri(rfd)

        assert score < 1.0


class TestFRI:

    @skip_if_no_ext
    def test_all_above_threshold(self):
        scores = np.ones(
            100,
            dtype=np.float64,
        )

        assert _ext.metrics.fri(
            scores,
            0.5,
        ) == pytest.approx(1.0)

    @skip_if_no_ext
    def test_none_above_threshold(self):
        scores = np.zeros(
            100,
            dtype=np.float64,
        )

        assert _ext.metrics.fri(
            scores,
            0.5,
        ) == pytest.approx(0.0)


# ===========================================================================
# Fractal Dimensions
# ===========================================================================

class TestFractalDimensions:

    @skip_if_no_ext
    def test_higuchi_returns_positive(self, white_noise):
        d = _ext.dimensions.higuchi(
            white_noise,
            10,
        )

        assert d > 0.0

    @skip_if_no_ext
    def test_katz_returns_positive(self, sine_wave):
        d = _ext.dimensions.katz(sine_wave)

        assert d > 0.0

    @skip_if_no_ext
    def test_petrosian_returns_positive(self, white_noise):
        d = _ext.dimensions.petrosian(white_noise)

        assert d > 0.0

    @skip_if_no_ext
    def test_hurst_between_zero_and_two(self, white_noise):
        h = _ext.dimensions.hurst(white_noise)

        assert 0.0 < h < 2.0

    @skip_if_no_ext
    def test_dfa_returns_float(self, white_noise):
        alpha = _ext.dimensions.dfa(white_noise)

        assert isinstance(alpha, float)

    @skip_if_no_ext
    def test_box_counting_returns_positive(self, white_noise):
        d = _ext.dimensions.box_counting(white_noise)

        assert d > 0.0

    @skip_if_no_ext
    def test_wavelet_dim_returns_positive(self, white_noise):
        d = _ext.dimensions.wavelet_dim(
            white_noise,
            6,
        )

        assert d > 0.0

    @skip_if_no_ext
    def test_lacunarity_returns_float(self, white_noise):
        lc = _ext.dimensions.lacunarity(
            white_noise,
            8,
        )

        assert isinstance(lc, float)

    @skip_if_no_ext
    def test_lyapunov_kaplan_yorke(self):
        exps = np.array(
            [0.5, 0.2, -0.1, -1.0],
            dtype=np.float64,
        )

        d = _ext.dimensions.lyapunov_dim(exps)

        assert d > 0.0

    @skip_if_no_ext
    def test_multifractal_spectrum_has_keys(self, white_noise):
        result = _ext.dimensions.multifractal_spectrum(
            white_noise
        )

        assert "alpha" in result
        assert "f" in result

    @skip_if_no_ext
    def test_correlation_dim_positive(self, white_noise):
        d = _ext.dimensions.correlation_dim(
            white_noise,
            2,
            1,
        )

        assert isinstance(d, float)


# ===========================================================================
# Neural Networks
# ===========================================================================

class TestFDNN:

    @skip_if_no_ext
    def test_forward_output_shape(self):
        cfg = _ext.neural.FDNNConfig()

        cfg.input_dim = 32
        cfg.output_dim = 8

        state = _ext.neural.fdnn_init(
            cfg,
            0,
        )

        x = np.random.default_rng(3).random(32)
        x = x.astype(np.float64)

        out = _ext.neural.fdnn_forward(
            x,
            state,
            cfg,
        )

        assert len(out) == 8

    @skip_if_no_ext
    def test_fractal_loss_nonnegative(self):
        cfg = _ext.neural.FDNNConfig()

        state = _ext.neural.fdnn_init(
            cfg,
            1,
        )

        x = np.random.default_rng(4).random(
            cfg.input_dim
        )

        x = x.astype(np.float64)

        loss = _ext.neural.fdnn_fractal_loss(
            x,
            state,
            cfg,
        )

        assert loss >= 0.0


class TestMSFCN:

    @skip_if_no_ext
    def test_forward_returns_vector(self):
        cfg = _ext.neural.MSFCNConfig()

        cfg.kernel_size = 4
        cfg.num_scales = 3

        state = _ext.neural.msfcn_init(
            cfg,
            0,
        )

        sig = np.random.default_rng(5).random(128)
        sig = sig.astype(np.float64)

        out = _ext.neural.msfcn_forward(
            sig,
            state,
            cfg,
        )

        assert len(out) > 0

    @skip_if_no_ext
    def test_scale_consistency_loss_nonnegative(self):
        cfg = _ext.neural.MSFCNConfig()

        state = _ext.neural.msfcn_init(
            cfg,
            0,
        )

        sig = np.random.default_rng(6).random(128)
        sig = sig.astype(np.float64)

        loss = _ext.neural.msfcn_scale_consistency_loss(
            sig,
            state,
            cfg,
        )

        assert loss >= 0.0


class TestSNNFT:

    @skip_if_no_ext
    def test_step_returns_spikes(self):
        cfg = _ext.neural.SNNFTConfig()

        cfg.input_dim = 8
        cfg.num_compartments = 4

        state = _ext.neural.snnft_init(
            cfg,
            0,
        )

        spikes = np.zeros(
            8,
            dtype=np.float64,
        )

        spikes[0] = 1.0

        out = _ext.neural.snnft_step(
            spikes,
            state,
            cfg,
        )

        assert len(out) == 1

    @skip_if_no_ext
    def test_stdp_runs_without_error(self):
        cfg = _ext.neural.SNNFTConfig()

        cfg.input_dim = 8

        state = _ext.neural.snnft_init(
            cfg,
            0,
        )

        pre = np.zeros(
            8,
            dtype=np.float64,
        )

        post = np.zeros(
            1,
            dtype=np.float64,
        )

        _ext.neural.snnft_stdp_update(
            state,
            cfg,
            pre,
            post,
        )

    @skip_if_no_ext
    def test_energy_loss(self):
        cfg = _ext.neural.SNNFTConfig()

        loss = _ext.neural.snnft_energy_loss(
            [10, 20, 30],
            cfg,
        )

        assert loss >= 0.0

    # IMPORTANT:
    # This test also requires the native extension because
    # fractalich.neural ultimately depends on fractalich_ext.
    @skip_if_no_ext
    def test_python_neural_snnft_wrapper(self):
        import fractalich.neural as fn

        cfg = fn.snnft_config(
            input_dim=8,
            num_compartments=4,
        )

        state = fn.snnft_init(
            cfg,
            seed=42,
        )

        out = fn.snnft_step(
            np.ones(8, dtype=np.float64),
            state,
            cfg,
        )

        assert isinstance(out, list)
        assert len(out) == 1

        fn.snnft_stdp_update(
            state,
            cfg,
            np.ones(8, dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )

        loss = fn.snnft_energy_loss(
            [5, 10],
            cfg,
        )

        assert loss >= 0.0