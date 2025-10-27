import glob
import circular_plots as plots
import numpy as np
from scipy.linalg import eig

filenames_R10 = sorted(glob.glob("GCM_circular_N50_R20_r03_eps*.mat"))
# filenames_R10 = sorted(glob.glob("GCM_circular_N50_R*_eps0.mat"))

print(filenames_R10)


eigs = {}
modes = {}
params = {}
mats = {}

for filename in filenames_R10:
    param = plots.extract_params(filename)  # [R, r, eps]
    size, R, r, eps = param
    print(param)
    print("\n")
    mat = plots.read_hdf5_modes(filename, "GCM_circular", "real")
    freq, mode = eig(mat, left=False, right=True)
    idx = np.lexsort((freq.imag, freq.real))
    freq = freq[idx]
    mode = mode[:, idx]

    size_key = f"{size}"  # ex: -0.100, 0.000, 0.200
    eps_key = f"{eps:.3f}"  # ex: -0.100, 0.000, 0.200
    R_key = f"{R}"
    mats[eps_key] = mat
    eigs[eps_key] = freq
    modes[eps_key] = mode
    params[eps_key] = param



sorted_keys = sorted(params, key=lambda k: params[k][3])
mats_sorted = {k: mats[k] for k in sorted_keys}
mats = mats_sorted
eigs_sorted = {k: eigs[k] for k in sorted_keys}
eigs = eigs_sorted
modes_sorted = {k: modes[k] for k in sorted_keys}
modes = modes_sorted
params_sorted = {k: params[k] for k in sorted_keys}
params = params_sorted


# mat = plots.read_hdf5_modes(filename, "GCM_circular", "real")
# freq, mode = eig(mat, left=False, right=True)
# idx = np.lexsort((freq.imag, freq.real))
# freq = freq[idx]
# mode = mode[:, idx]
# print(modes)
# plots.plot_modes(modes["0.100"],params["0.100"])

R_array = np.array([float(k) for k in modes.keys()])
# plots.plot_real_eigenfrequencies(eigs["0.100"], params["0.100"])
plots.plot_IPR(modes, R_array)
# print(eigs)
# plots.plot_all_eigenfrequencies(eigs,params)
# plots.plot_avgerage_distribution(modes,R_array)
# plots.plot_matrix_entries(mats, params)
# plots.plot_matrix_entries_heatmap(mats["0.000"])



# plot_real_eigenfrequencies(eigs[0],params[0])




# plot_modes(modes["0.0"],params["0.0"])
# plot_avgerage_distribution(modes["0.0"],params["0.0"])
