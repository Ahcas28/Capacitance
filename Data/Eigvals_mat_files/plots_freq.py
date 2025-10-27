import h5py
import numpy as np
import matplotlib.pyplot as plt




def read_hdf5_modes(fichier_mat, nom_groupe, return_type):
    """
    Charge une matrice complexe à partir d'un fichier .mat au format HDF5 (MATLAB v7.3 / Octave -hdf5).

    Paramètres :
        fichier_mat (str) : chemin vers le fichier .mat
        nom_variable (str): nom de la variable complexe à extraire

    Retour :
        np.ndarray : matrice complexe
    """
    with h5py.File(fichier_mat, 'r') as f:
        dataset = f[nom_groupe + "/value"]
        data = dataset[:]

        # Reconstruction de la matrice complexe
        real = data['real']
        imag = data['imag']
        matrice_complexe = real + 1j * imag

        if return_type == "real":
            return real.T
        elif return_type == "imag":
            return imag.T
        elif return_type == "abs":
            return np.abs(matrice_complexe.T)
        else:
            return matrice_complexe.T        


def self_interaction_derivative(R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)

def self_interaction(R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)


def compute_mean_interaction_terms(mat):
    N = len(mat)
    mean_m1 = 0
    for i in range(1, N):
        mean_m1 += mat[i][i-1]

    mean_m2 = 0
    for i in range(2, N):
        mean_m2 += mat[i][i-2]

    mean_p1 = 0
    for i in range(0, N-1):
        mean_p1 += mat[i][i+1]

    mean_p2 = 0
    for i in range(0, N-2):
        mean_p2 += mat[i][i+2]

    return([mean_m2/N, mean_m1/N, mean_p1/N, mean_p2/N])

def compute_zeroth_order_frequency_approx(R,gamma,mat):
    C_gamma = self_interaction(R, gamma)
    [Cm2, Cm1, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    return(C_gamma + Cm1 + Cp1)



def plot_localized_frequency(freqs, GC_matrices, loc_index, R_list, gamma_list):

    plt.figure(figsize=(10, 6))

    loc_freq = freqs[:][loc_index]
    zeroth_orderfreq = np.array([compute_zeroth_order_frequency_approx(R, gamma, mat) for R,gamma,mat in zip(R_list, gamma_list, GC_matrices)])
    plt.plot(R_list, loc_freq, label = 'data', color='k')
    plt.plot(R_list, zeroth_orderfreq, label = '0th order', color='b')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(r'$R$')
    plt.ylabel(r'$\omega_0(\varepsilon)$')
    plt.title('Localized eigenfrequency')
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()






GCM_folder ='./GCM_skin_mat_files/gamma1/'
freq_folder ='./Eigvals_mat_files/gamma1/'

# R = 0.1
GCM_N101_R01_eps_m000008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m000008.mat", 'GCM_skin','real')
mat_N101_R01_eps_m000008 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m000008.mat', 'eval_skin','real')

# R = 0.15
GCM_N101_R015_eps_m000035 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m000035.mat", 'GCM_skin','real')
mat_N101_R015_eps_m000035 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m000035.mat', 'eval_skin','real')

# R = 0.2
GCM_N101_R02_eps_m00013 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m00013.mat", 'GCM_skin','real')
mat_N101_R02_eps_m00013 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m00013.mat', 'eval_skin','real')

# R = 0.25
GCM_N101_R025_eps_m00035 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m00035.mat", 'GCM_skin','real')
mat_N101_R025_eps_m00035 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m00035.mat', 'eval_skin','real')

# R = 0.3
GCM_N101_R03_eps_m0008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0008.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0008 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m0008.mat', 'eval_skin','real')

# R = 0.35
GCM_N101_R035_eps_m0016 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m0016.mat", 'GCM_skin','real')
mat_N101_R035_eps_m0016 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m0016.mat', 'eval_skin','real')

# R = 0.4
GCM_N101_R04_eps_m0035 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m0035.mat", 'GCM_skin','real')
mat_N101_R04_eps_m0035 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0035.mat', 'eval_skin','real')

# R = 0.45
GCM_N101_R045_eps_m0078 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0078.mat", 'GCM_skin','real')
mat_N101_R045_eps_m0078 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0078.mat', 'eval_skin','real')


GCMs = [GCM_N101_R01_eps_m000008, GCM_N101_R015_eps_m000035, GCM_N101_R02_eps_m00013, GCM_N101_R025_eps_m00035, GCM_N101_R03_eps_m0008, GCM_N101_R035_eps_m0016, GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0078]
Freqs = [mat_N101_R01_eps_m000008, mat_N101_R015_eps_m000035, mat_N101_R02_eps_m00013, mat_N101_R025_eps_m00035, mat_N101_R03_eps_m0008, mat_N101_R035_eps_m0016, mat_N101_R04_eps_m0035, mat_N101_R045_eps_m0078]
Rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
gammas = [1, 1, 1, 1, 1, 1, 1, 1]
plot_localized_frequency(Freqs, GCMs, 101, Rs, gammas)