/**
 * Fractalich - python_bindings.cpp
 * Pybind11 bindings for all C++ modules.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fractal_metrics.hpp"
#include "fractal_dimensions.hpp"
#include "fractal_nn.hpp"

namespace py = pybind11;
using namespace fractalich;

// ─── Helper: numpy array → std::vector<double> ──────────────────────────────
static std::vector<double> to_vec(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    auto* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

static std::vector<int> to_vec_int(py::array_t<int> arr) {
    py::buffer_info buf = arr.request();
    auto* ptr = static_cast<int*>(buf.ptr);
    return std::vector<int>(ptr, ptr + buf.size);
}

static std::vector<std::vector<double>> to_vec2d(py::list list_of_arrays) {
    std::vector<std::vector<double>> result;
    for (auto item : list_of_arrays) {
        result.push_back(to_vec(item.cast<py::array_t<double>>()));
    }
    return result;
}

PYBIND11_MODULE(fractalich_ext, m) {
    m.doc() = "Fractalich: Fractal-Based Data Analysis Library (C++ backend)";

    // ─── Metrics submodule ───────────────────────────────────────────────────
    py::module_ met = m.def_submodule("metrics", "Fractal Evaluation Metrics");

    met.def("ffs", [](py::array_t<double> original, py::array_t<double> reconstructed) {
        return metrics::ffs(to_vec(original), to_vec(reconstructed));
    }, py::arg("original"), py::arg("reconstructed"),
    R"(Fractal Fidelity Score (FFS).

    FFS = 1 - ||Xf - Xo||_2 / ||Xo||_2

    Args:
        original (ndarray): Original signal.
        reconstructed (ndarray): Fractal-reconstructed signal.

    Returns:
        float: FFS score in (-inf, 1]. 1.0 = perfect reconstruction.)");

    met.def("msc", [](py::list orig_scales, py::list rec_scales) {
        return metrics::msc(to_vec2d(orig_scales), to_vec2d(rec_scales));
    }, py::arg("substructures_orig"), py::arg("substructures_rec"),
    R"(Multiscale Structural Consistency (MSC).

    MSC = (1/n) * sum_i SSIM(Si, S'i)

    Args:
        substructures_orig (list of ndarray): Original substructures at each scale.
        substructures_rec  (list of ndarray): Reconstructed substructures.

    Returns:
        float: Mean SSIM across all scales.)");

    met.def("fre", [](py::array_t<double> measured, py::array_t<double> predicted) {
        return metrics::fre(to_vec(measured), to_vec(predicted));
    }, py::arg("measured"), py::arg("predicted"),
    R"(Fractal Residual Error (FRE).

    FRE = sum_i |Di - D'i|

    Args:
        measured  (ndarray): Empirically measured fractal dimensions.
        predicted (ndarray): Model-predicted fractal dimensions.

    Returns:
        float: Total absolute residual error.)");

    met.def("ffri", [](py::array_t<double> feature, py::array_t<int> labels, int num_bins) {
        return metrics::ffri(to_vec(feature), to_vec_int(labels), num_bins);
    }, py::arg("feature"), py::arg("labels"), py::arg("num_bins") = 50,
    R"(Fractal Feature Relevance Index (FFRI).

    FFRI = MI(f, y) / H(f)

    Args:
        feature  (ndarray): Fractal feature vector (float64).
        labels   (ndarray): Class labels (int32).
        num_bins (int):     Histogram bins for entropy estimation.

    Returns:
        float: FFRI score; higher = more relevant.)");

    met.def("ssri", [](py::array_t<double> rfd_values) {
        return metrics::ssri(to_vec(rfd_values));
    }, py::arg("rfd_values"),
    R"(Self-Similarity Retention Index (SSRI).

    SSRI = 1 - Var(RFD(s)) / mean(RFD(s))

    Args:
        rfd_values (ndarray): Relative Fractal Dimension at each scale.

    Returns:
        float: SSRI score; higher = better self-similarity preservation.)");

    met.def("fri", [](py::array_t<double> match_scores, double theta) {
        return metrics::fri(to_vec(match_scores), theta);
    }, py::arg("match_scores"), py::arg("theta") = 0.5,
    R"(Fractal Recurrence Index (FRI).

    FRI = (1/T) * sum_{t} 1{FractalMatch(t, t+tau) > theta}

    Args:
        match_scores (ndarray): FractalMatch scores at each time step.
        theta        (float):   Recurrence threshold.

    Returns:
        float: FRI in [0, 1].)");

    // ─── Dimensions submodule ────────────────────────────────────────────────
    py::module_ dim = m.def_submodule("dimensions", "Fractal Dimension Estimators");

    dim.def("box_counting", [](py::array_t<double> sig, int mn, int mx, int ns) {
        return dimensions::box_counting(to_vec(sig), mn, mx, ns);
    }, py::arg("signal"), py::arg("min_box")=2, py::arg("max_box")=64, py::arg("num_scales")=10,
    "Box-Counting Dimension.");

    dim.def("hausdorff", [](py::array_t<double> pts, int D, int mn, int mx, int ns) {
        return dimensions::hausdorff(to_vec(pts), D, mn, mx, ns);
    }, py::arg("points"), py::arg("D")=2,
       py::arg("min_box")=2, py::arg("max_box")=64, py::arg("num_scales")=10,
    "Hausdorff Dimension (numerical estimate).");

    dim.def("correlation_dim", [](py::array_t<double> sig, int m, int tau) {
        return dimensions::correlation_dim(to_vec(sig), m, tau);
    }, py::arg("data"), py::arg("embedding_dim")=3, py::arg("tau")=1,
    "Correlation Dimension (Grassberger-Procaccia).");

    dim.def("information_dim", [](py::array_t<double> sig, int nb) {
        return dimensions::information_dim(to_vec(sig), nb);
    }, py::arg("signal"), py::arg("num_bins")=64,
    "Information Dimension.");

    dim.def("packing_dim", [](py::array_t<double> sig, int mn, int mx, int ns) {
        return dimensions::packing_dim(to_vec(sig), mn, mx, ns);
    }, py::arg("signal"), py::arg("min_box")=2, py::arg("max_box")=64, py::arg("num_scales")=10,
    "Packing Dimension.");

    dim.def("minkowski_bouligand", [](py::array_t<double> sig, int mn, int mx, int ns) {
        return dimensions::minkowski_bouligand(to_vec(sig), mn, mx, ns);
    }, py::arg("signal"), py::arg("min_box")=2, py::arg("max_box")=64, py::arg("num_scales")=10,
    "Minkowski-Bouligand Dimension.");

    dim.def("lyapunov_dim", [](py::array_t<double> exponents) {
        return dimensions::lyapunov_dim(to_vec(exponents));
    }, py::arg("lyapunov_exponents"),
    "Lyapunov Dimension (Kaplan-Yorke formula).");

    dim.def("renyi_dim", [](py::array_t<double> sig, double q, int nb) {
        return dimensions::renyi_dim(to_vec(sig), q, nb);
    }, py::arg("signal"), py::arg("q")=2.0, py::arg("num_bins")=64,
    "Renyi Dimension D_q.");

    dim.def("higuchi", [](py::array_t<double> sig, int k_max) {
        return dimensions::higuchi(to_vec(sig), k_max);
    }, py::arg("signal"), py::arg("k_max")=10,
    "Higuchi Fractal Dimension.");

    dim.def("katz", [](py::array_t<double> sig) {
        return dimensions::katz(to_vec(sig));
    }, py::arg("signal"),
    "Katz Fractal Dimension.");

    dim.def("petrosian", [](py::array_t<double> sig) {
        return dimensions::petrosian(to_vec(sig));
    }, py::arg("signal"),
    "Petrosian Fractal Dimension.");

    dim.def("dfa", [](py::array_t<double> sig, std::vector<int> scales) {
        return dimensions::dfa(to_vec(sig), scales);
    }, py::arg("signal"), py::arg("scales")=std::vector<int>{},
    "Detrended Fluctuation Analysis (DFA) exponent.");

    dim.def("multifractal_spectrum", [](py::array_t<double> sig,
                                        double q_min, double q_max, int num_q, int num_scales) {
        auto result = dimensions::multifractal_spectrum(to_vec(sig), q_min, q_max, num_q, num_scales);
        return py::dict(py::arg("alpha") = result.alpha, py::arg("f") = result.f);
    }, py::arg("signal"),
       py::arg("q_min")=-5.0, py::arg("q_max")=5.0,
       py::arg("num_q")=21, py::arg("num_scales")=10,
    "Multifractal Spectrum f(alpha). Returns dict with 'alpha' and 'f' keys.");

    dim.def("wavelet_dim", [](py::array_t<double> sig, int levels) {
        return dimensions::wavelet_dim(to_vec(sig), levels);
    }, py::arg("signal"), py::arg("num_levels")=6,
    "Wavelet-Based Fractal Dimension (Haar).");

    dim.def("lacunarity", [](py::array_t<double> sig, int box_size) {
        return dimensions::lacunarity(to_vec(sig), box_size);
    }, py::arg("signal"), py::arg("box_size")=8,
    "Lacunarity.");

    dim.def("hurst", [](py::array_t<double> sig) {
        return dimensions::hurst(to_vec(sig));
    }, py::arg("signal"),
    "Hurst Exponent (R/S method).");

    dim.def("temporal_fractal_dim", [](py::array_t<double> sig, double dt) {
        return dimensions::temporal_fractal_dim(to_vec(sig), dt);
    }, py::arg("signal"), py::arg("dt")=1.0,
    "Temporal Fractal Dimension.");

    // ─── NN submodule ─────────────────────────────────────────────────────────
    py::module_ nmod = m.def_submodule("neural", "Fractal Neural Network Architectures");

    // FDNN Config struct
    py::class_<nn::FDNNConfig>(nmod, "FDNNConfig")
        .def(py::init<>())
        .def_readwrite("input_dim",       &nn::FDNNConfig::input_dim)
        .def_readwrite("branch_dim",      &nn::FDNNConfig::branch_dim)
        .def_readwrite("output_dim",      &nn::FDNNConfig::output_dim)
        .def_readwrite("num_branches",    &nn::FDNNConfig::num_branches)
        .def_readwrite("recursion_depth", &nn::FDNNConfig::recursion_depth)
        .def_readwrite("scale_factor",    &nn::FDNNConfig::scale_factor)
        .def_readwrite("lambda_fract",    &nn::FDNNConfig::lambda_fract);

    py::class_<nn::FDNNState>(nmod, "FDNNState");

    nmod.def("fdnn_init", &nn::fdnn_init,
             py::arg("config"), py::arg("seed")=42,
             "Initialise FDNN weights.");

    nmod.def("fdnn_forward", [](py::array_t<double> x, const nn::FDNNState& state, const nn::FDNNConfig& cfg) {
        return nn::fdnn_forward(to_vec(x), state, cfg);
    }, py::arg("x"), py::arg("state"), py::arg("config"),
    "FDNN forward pass. Returns output vector.");

    nmod.def("fdnn_fractal_loss", [](py::array_t<double> x, const nn::FDNNState& state, const nn::FDNNConfig& cfg) {
        return nn::fdnn_fractal_loss(to_vec(x), state, cfg);
    }, py::arg("x"), py::arg("state"), py::arg("config"),
    "FDNN fractal consistency regularisation loss.");

    // DDAN
    py::class_<nn::DDANConfig>(nmod, "DDANConfig")
        .def(py::init<>())
        .def_readwrite("hidden_dim",    &nn::DDANConfig::hidden_dim)
        .def_readwrite("num_paths",     &nn::DDANConfig::num_paths)
        .def_readwrite("path_len",      &nn::DDANConfig::path_len)
        .def_readwrite("lambda_sparse", &nn::DDANConfig::lambda_sparse);

    py::class_<nn::DDANState>(nmod, "DDANState");

    nmod.def("ddan_init", &nn::ddan_init,
             py::arg("config"), py::arg("seed")=42,
             "Initialise DDAN weights.");

    nmod.def("ddan_forward", [](py::array_t<double> query,
                                py::list paths_list,
                                std::vector<int> depths,
                                const nn::DDANState& state,
                                const nn::DDANConfig& cfg) {
        // paths_list: list of list of ndarray (paths x nodes x hidden)
        std::vector<nn::DendriticPath> paths;
        for (auto path_obj : paths_list) {
            nn::DendriticPath path;
            for (auto node_obj : path_obj.cast<py::list>()) {
                path.push_back(to_vec(node_obj.cast<py::array_t<double>>()));
            }
            paths.push_back(path);
        }
        return nn::ddan_forward(to_vec(query), paths, depths, state, cfg);
    }, py::arg("query"), py::arg("paths"), py::arg("path_depths"),
       py::arg("state"), py::arg("config"),
    "DDAN attended forward pass over dendritic paths.");

    // MSFCN
    py::class_<nn::MSFCNConfig>(nmod, "MSFCNConfig")
        .def(py::init<>())
        .def_readwrite("kernel_size",   &nn::MSFCNConfig::kernel_size)
        .def_readwrite("num_scales",    &nn::MSFCNConfig::num_scales)
        .def_readwrite("base_scale",    &nn::MSFCNConfig::base_scale)
        .def_readwrite("lambda_scale",  &nn::MSFCNConfig::lambda_scale)
        .def_readwrite("num_ifs_terms", &nn::MSFCNConfig::num_ifs_terms);

    py::class_<nn::MSFCNState>(nmod, "MSFCNState");

    nmod.def("msfcn_init", &nn::msfcn_init,
             py::arg("config"), py::arg("seed")=42,
             "Initialise MSFCN kernel and aggregation weights.");

    nmod.def("msfcn_forward", [](py::array_t<double> sig, const nn::MSFCNState& state, const nn::MSFCNConfig& cfg) {
        return nn::msfcn_forward(to_vec(sig), state, cfg);
    }, py::arg("signal"), py::arg("state"), py::arg("config"),
    "MSFCN multi-scale convolution forward pass.");

    nmod.def("msfcn_scale_consistency_loss", [](py::array_t<double> sig, const nn::MSFCNState& state, const nn::MSFCNConfig& cfg) {
        return nn::msfcn_scale_consistency_loss(to_vec(sig), state, cfg);
    }, py::arg("signal"), py::arg("state"), py::arg("config"),
    "MSFCN scale-consistency regularisation loss.");

    // SNN-FT
    py::class_<nn::SNNFTConfig>(nmod, "SNNFTConfig")
        .def(py::init<>())
        .def_readwrite("num_compartments", &nn::SNNFTConfig::num_compartments)
        .def_readwrite("input_dim",        &nn::SNNFTConfig::input_dim)
        .def_readwrite("tau_c",            &nn::SNNFTConfig::tau_c)
        .def_readwrite("tau_m",            &nn::SNNFTConfig::tau_m)
        .def_readwrite("v_th",             &nn::SNNFTConfig::v_th)
        .def_readwrite("v_reset",          &nn::SNNFTConfig::v_reset)
        .def_readwrite("dt",               &nn::SNNFTConfig::dt)
        .def_readwrite("A_plus",           &nn::SNNFTConfig::A_plus)
        .def_readwrite("A_minus",          &nn::SNNFTConfig::A_minus)
        .def_readwrite("tau_plus",         &nn::SNNFTConfig::tau_plus)
        .def_readwrite("tau_minus",        &nn::SNNFTConfig::tau_minus)
        .def_readwrite("eta_scale",        &nn::SNNFTConfig::eta_scale)
        .def_readwrite("num_ifs_terms",    &nn::SNNFTConfig::num_ifs_terms)
        .def_readwrite("lambda_energy",    &nn::SNNFTConfig::lambda_energy);

    py::class_<nn::SNNFTState>(nmod, "SNNFTState");

    nmod.def("snnft_init", &nn::snnft_init,
             py::arg("config"), py::arg("seed")=42,
             "Initialise SNN-FT state and fractal adjacency matrix.");

    nmod.def("snnft_step", [](py::array_t<double> spikes, nn::SNNFTState& state, const nn::SNNFTConfig& cfg) {
        return nn::snnft_step(to_vec(spikes), state, cfg);
    }, py::arg("input_spikes"), py::arg("state"), py::arg("config"),
    "Simulate one SNN-FT time step. Returns output spike vector.");

    nmod.def("snnft_stdp_update", [](nn::SNNFTState& state, const nn::SNNFTConfig& cfg,
                                     py::array_t<double> pre, py::array_t<double> post) {
        nn::snnft_stdp_update(state, cfg, to_vec(pre), to_vec(post));
    }, py::arg("state"), py::arg("config"), py::arg("pre_spikes"), py::arg("post_spikes"),
    "Apply scale-modulated STDP weight update.");

    nmod.def("snnft_energy_loss", [](std::vector<int> counts, const nn::SNNFTConfig& cfg) {
        return nn::snnft_energy_loss(counts, cfg);
    }, py::arg("spike_counts_per_window"), py::arg("config"),
    "SNN-FT energy regularisation loss.");

    // ─── Utility ──────────────────────────────────────────────────────────────
    m.attr("__version__") = "0.1.0";
    m.attr("__library__") = "Fractalich";
}
