import glob
import linear_plots as plots
import numpy as np
from scipy.linalg import eig

filenames_N100 = sorted(glob.glob("GCM_linear_N*_r02_gamma1.mat"))
# filenames_R10 = sorted(glob.glob("GCM_linear_R*_eps0.mat"))
# filenames_R10 = 
# filename = "GCM_linear_N50_r03_gamma1_epsm03.mat"

eigs = {}
modes = {}
params = {}
mats = {}

for filename in filenames_N100:
    param = plots.extract_params(filename)  # [R, r, eps]
    N, r, gamma = param

    mat = plots.read_hdf5_modes(filename, "GCM_skin", "real")
    freq, mode = eig(mat, left=False, right=True)
    idx = np.lexsort((freq.imag, freq.real))
    freq = freq[idx]
    mode = mode[:, idx]

    size_key = f"{N}"  # ex: -0.100, 0.000, 0.200
    gamma_key = f"{gamma}"
    mats[size_key] = mat
    eigs[size_key] = freq
    modes[size_key] = mode
    params[size_key] = param

sorted_keys = sorted(params, key=lambda k: params[k][0])
mats_sorted = {k: mats[k] for k in sorted_keys}
mats = mats_sorted
eigs_sorted = {k: eigs[k] for k in sorted_keys}
eigs = eigs_sorted
modes_sorted = {k: modes[k] for k in sorted_keys}
modes = modes_sorted
params_sorted = {k: params[k] for k in sorted_keys}
params = params_sorted

print(params)
# mat0 = plots.read_hdf5_modes(filename, "GCM_skin", "none")

# freq, modes = eig(mat0, left=False, right=True)
# idx = np.lexsort((freq.imag, freq.real))
# freq = freq[idx]
# modes = modes[:,idx]
# print(freq)
# plots.plot_complex_eigenfrequencies(freq,None)
# plots.plot_real_eigenfrequencies(freq,None)
# plots.plot_modes(modes, None)

# print(eigs)
# plots.plot_all_eigenfrequencies(eigs,params)
# plots.plot_avgerage_distribution(modes,params)
# plots.plot_matrix_entries(mats, params)
# plot_matrix_entries_heatmap(mats["0.0"])
gamma_array = np.array([float(k) for k in modes.keys()])
print(gamma_array)
plots.plot_IPR(modes,gamma_array)

# plot_real_eigenfrequencies(eigs[0],params[0])




# plot_modes(modes["0.0"],params["0.0"])
# plot_avgerage_distribution(modes["0.0"],params["0.0"])
