import glob
import Final_plots as plots
import numpy as np
from scipy.linalg import eig



# filenames = sorted(glob.glob("./Positive/GCM_skin_N_51_R025_eps_*.mat"))
# filenames = sorted(glob.glob("./Negative/GCM_skin_N_51_R025_eps_m*.mat"))
filenames = sorted(glob.glob("./gamma1/GCM_skin_N101_R02_eta_*.mat"))

eigs = {}
approx_eigs = {}
modes = {}
approx_modes = {}
params = {}
mats = {}
approx_mats = {}

for filename in filenames:
    param = plots.extract_params(filename)  # [R, r, eps]
    N, r, eps = param

    # print(param)
    mat = plots.read_hdf5_modes(filename, "GCM_skin", "real")
    approx_mat = plots.approximation_bandes(mat, 50)
    freq, mode = eig(mat, left=False, right=True)
    approx_freq, approx_mode = eig(approx_mat, left=False, right=True)
    idx = np.lexsort((freq.imag, freq.real))
    approx_idx = np.lexsort((approx_freq.imag, approx_freq.real))
    freq = freq[idx]
    approx_freq = approx_freq[approx_idx]
    mode = mode[:, idx]
    approx_mode = approx_mode[:, approx_idx]

    size_key = f"{N}"  # ex: -0.100, 0.000, 0.200
    eps_key = f"{eps}"
    mats[eps_key] = mat
    approx_mats[eps_key] = approx_mat
    eigs[eps_key] = freq
    approx_eigs[eps_key] = approx_freq
    modes[eps_key] = mode
    approx_modes[eps_key] = approx_mode
    params[eps_key] = param


sorted_keys = sorted(params, key=lambda k: params[k][2])
mats_sorted = {k: mats[k] for k in sorted_keys}
approx_mats_sorted = {k: approx_mats[k] for k in sorted_keys}
mats = mats_sorted
approx_mats = approx_mats_sorted
eigs_sorted = {k: eigs[k] for k in sorted_keys}
approx_eigs_sorted = {k: approx_eigs[k] for k in sorted_keys}
eigs = eigs_sorted
approx_eigs = approx_eigs_sorted
modes_sorted = {k: modes[k] for k in sorted_keys}
approx_modes_sorted = {k: approx_modes[k] for k in sorted_keys}
modes = modes_sorted
approx_modes = approx_modes_sorted
params_sorted = {k: params[k] for k in sorted_keys}
params = params_sorted

deloc_modes = [mode[:,-1] for mode in modes.values()]
approx_deloc_modes = [approx_mode[:,-1] for approx_mode in approx_modes.values()]
# print(deloc_modes)
# 
arr = np.array(list(modes.keys()), dtype=float)


filename = "GCM_skin_N101_R02_eta_0.mat"
mat0 = plots.read_hdf5_modes(filename, "GCM_skin", "real")
approx0 = plots.best_Toeplitz_approx(plots.approximation_bandes(mat0, 50))
plots.scatter_decay(mat0)

plots.decay_from_means(mat0, cut=70)


# plots.plot_real_eigenfrequencies(eigs)
# plots.plot_modes(modes["0.04"])
# print(sorted_keys)

# plots.plot_delocalized_modes(deloc_modes, arr)
# plots.plot_complex_band_structure(approx0, n_lambda=400, lambda_range=(1e-4, 2e-3))