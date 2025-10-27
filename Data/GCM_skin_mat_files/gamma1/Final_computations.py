import glob
import Final_plots as plots
import numpy as np
from scipy.linalg import eig



filenames_N1 = sorted(glob.glob("GCM_skin_N_101_R01_eps_m*.mat"))
filenames_N2 = sorted(glob.glob("GCM_skin_N_101_R025_eps_m*.mat"))
filenames_N3 = sorted(glob.glob("GCM_skin_N_101_R04_eps_m*.mat"))
# filenames_R10 = sorted(glob.glob("GCM_linear_R*_eps0.mat"))
# filenames_R10 = 
# filename = "GCM_linear_N50_r03_gamma1_epsm03.mat"

# filename1 = "GCM_skin_N_101_R01_eps_m00.mat"
# filename2 = "GCM_skin_N_101_R01_eps_m00014.mat"
# filename3 = "GCM_skin_N_101_R01_eps_m00017.mat"

# filenames = [filename1, filename2, filename3]

eigs1 = {}
modes1 = {}
params1 = {}
mats1 = {}

for filename in filenames_N1:
    param = plots.extract_params(filename)  # [R, r, eps]
    N, r, eps = param

    # print(param)
    mat = plots.read_hdf5_modes(filename, "GCM_skin", "real")
    freq, mode = eig(mat, left=False, right=True)
    idx = np.lexsort((freq.imag, freq.real))
    freq = freq[idx]
    mode = mode[:, idx]

    size_key = f"{N}"  # ex: -0.100, 0.000, 0.200
    eps_key = f"{eps}"
    mats1[eps_key] = mat
    eigs1[eps_key] = freq
    modes1[eps_key] = mode
    params1[eps_key] = param

sorted_keys = sorted(params1, key=lambda k: params1[k][2])
mats_sorted1 = {k: mats1[k] for k in sorted_keys}
mats1 = mats_sorted1
eigs_sorted1 = {k: eigs1[k] for k in sorted_keys}
eigs1 = eigs_sorted1
modes_sorted1 = {k: modes1[k] for k in sorted_keys}
modes1 = modes_sorted1
params_sorted1 = {k: params1[k] for k in sorted_keys}
params1 = params_sorted1


eigs2 = {}
modes2 = {}
params2 = {}
mats2 = {}

for filename in filenames_N2:
    param = plots.extract_params(filename)  # [R, r, eps]
    N, r, eps = param

    # print(param)
    mat = plots.read_hdf5_modes(filename, "GCM_skin", "real")
    freq, mode = eig(mat, left=False, right=True)
    idx = np.lexsort((freq.imag, freq.real))
    freq = freq[idx]
    mode = mode[:, idx]

    size_key = f"{N}"  # ex: -0.100, 0.000, 0.200
    eps_key = f"{eps}"
    mats2[eps_key] = mat
    eigs2[eps_key] = freq
    modes2[eps_key] = mode
    params2[eps_key] = param

sorted_keys = sorted(params2, key=lambda k: params2[k][2])
mats_sorted2 = {k: mats2[k] for k in sorted_keys}
mats2 = mats_sorted2
eigs_sorted2 = {k: eigs2[k] for k in sorted_keys}
eigs2 = eigs_sorted2
modes_sorted2 = {k: modes2[k] for k in sorted_keys}
modes2 = modes_sorted2
params_sorted2 = {k: params2[k] for k in sorted_keys}
params2 = params_sorted2


eigs3 = {}
modes3 = {}
params3 = {}
mats3 = {}

for filename in filenames_N3:
    param = plots.extract_params(filename)  # [R, r, eps]
    N, r, eps = param

    # print(param)
    mat = plots.read_hdf5_modes(filename, "GCM_skin", "real")
    freq, mode = eig(mat, left=False, right=True)
    idx = np.lexsort((freq.imag, freq.real))
    freq = freq[idx]
    mode = mode[:, idx]

    size_key = f"{N}"  # ex: -0.100, 0.000, 0.200
    eps_key = f"{eps}"
    mats3[eps_key] = mat
    eigs3[eps_key] = freq
    modes3[eps_key] = mode
    params3[eps_key] = param

sorted_keys = sorted(params3, key=lambda k: params3[k][2])
mats_sorted3 = {k: mats3[k] for k in sorted_keys}
mats3 = mats_sorted3
eigs_sorted3 = {k: eigs3[k] for k in sorted_keys}
eigs3 = eigs_sorted3
modes_sorted3 = {k: modes3[k] for k in sorted_keys}
modes3 = modes_sorted3
params_sorted3 = {k: params3[k] for k in sorted_keys}
params3 = params_sorted3


# print(sorted_keys)
# print(params)

filename = "GCM_skin_N_300_R02_eps0.mat"
mat0 = plots.read_hdf5_modes(filename, "GCM_skin", "real")



freq, modes = eig(mat0, left=False, right=True)
idx = np.lexsort((freq.imag, freq.real))
freq = freq[idx]
modes = modes[:,idx]
# print(freq)
# plots.plot_complex_eigenfrequencies(freq,None)
# plots.plot_real_eigenfrequencies(freq,None)
# plots.plot_modes(modes, None)

eps_IPR = [-1.5e-5, -0.003425, -0.03475]

matrices = [modes1["-1.5e-05"], modes2["-0.003425"], modes3["-0.03475"]]
params = [0.1, 0.25, 0.4]

# plots.plot_Nmodes(matrices, params)



# print(eigs)
# plots.plot_all_eigenfrequencies(eigs,params)
# plots.plot_avgerage_distribution(modes,params)
# plots.plot_matrix_entries(mats, params)
# plot_matrix_entries_heatmap(mats["0.0"])
gamma_array = np.array([float(k) for k in modes2.keys()])
# print(gamma_array)
print(gamma_array)
plots.plot_IPR(modes2,gamma_array)

# plot_real_eigenfrequencies(eigs[0],params[0])
eps_array1 = np.array([float(k) for k in modes1.keys()])
eps_array2 = np.array([float(k) for k in modes2.keys()])
eps_array3 = np.array([float(k) for k in modes3.keys()])

# plots.plot_IPR(modes, eps_array)
mat_file1 = "GCM_skin_N_101_R01_eps0.mat"
mat_file2 = "GCM_skin_N_101_R025_eps0.mat"
mat_file3 = "GCM_skin_N_101_R04_eps0.mat"

deriv_matfile1 = "GCM_skin_N_101_R01_eps_m0000001.mat"
deriv_mat1 = plots.read_hdf5_modes(deriv_matfile1, "GCM_skin", "real")
eps1 = -0.000001

deriv_matfile2 = "GCM_skin_N_101_R025_eps_m00033.mat"
deriv_mat2 = plots.read_hdf5_modes(deriv_matfile2, "GCM_skin", "real")
eps2 = -0.0033

deriv_matfile3 = "GCM_skin_N_101_R04_eps_m003.mat"
deriv_mat3 = plots.read_hdf5_modes(deriv_matfile3, "GCM_skin", "real")
eps3 = -0.03


mat1 = plots.read_hdf5_modes(mat_file1, "GCM_skin", "real")
mat2 = plots.read_hdf5_modes(mat_file2, "GCM_skin", "real")
mat3 = plots.read_hdf5_modes(mat_file3, "GCM_skin", "real")

deriv1 = plots.compute_deriv(mat1, deriv_mat1, eps1)
deriv2 = plots.compute_deriv(mat2, deriv_mat2, eps2)
deriv3 = plots.compute_deriv(mat3, deriv_mat3, eps3)


eps_full_1 = plots.compute_full_critical(mat1, deriv1)
eps_full_2 = plots.compute_full_critical(mat2, deriv2)
eps_full_3 = plots.compute_full_critical(mat3, deriv3)


eps_tri_1 = plots.compute_tridiagonal_critical(mat1,deriv1)
eps_tri_2 = plots.compute_tridiagonal_critical(mat2,deriv2)
eps_tri_3 = plots.compute_tridiagonal_critical(mat3,deriv3)

# print(eigs1["-1.5e-05"])
# print(eigs2["-0.003425"])
# print(eigs3["-0.03475"])

eps_full = [eps_full_1, eps_full_2, eps_full_3]
eps_tri = [eps_tri_1, eps_tri_2, eps_tri_3]
eps_IPR = [-1.5e-5, -0.003425, -0.03475]
eps_eye = [-0.0000985, -0.00348, -0.035]
eps_list = [eps_IPR, eps_eye, eps_full, eps_tri]
# print(eps_full)
# print(eps_tri)
# plots.plot_defect_threshold([0.1, 0.25, 0.4], eps_list)


approx_mat1 = plots.compute_approx_mat(mat1)
approx_mat2 = plots.compute_approx_mat(mat2)
approx_mat3 = plots.compute_approx_mat(mat3)

# pred1 = plots.compute_approx_eigs(approx_mat1, deriv1[55][55], eps_array1)
# pred2 = plots.compute_approx_eigs(approx_mat2, deriv2[55][55], eps_array2)
# pred3 = plots.compute_approx_eigs(approx_mat3, deriv3[55][55], eps_array3)

# print(eigs1)
# plots.plot_expansion_eigenfrequencies([eigs1, eigs2, eigs3], [pred1, pred2, pred3], [eps_array1, eps_array2, eps_array3])


