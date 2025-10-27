import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eig




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


def self_interaction_derivative(delta, v, R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = delta * v * v * gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)

def self_interaction(R, gamma, delta, v):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = delta * v * v * gamma * gamma * sinhgR
    denom = gR * coshgR - sinhgR
    return(num/denom)


def plot_complex_frequencies(data, title=None, save_name=None):
    """
    Affiche des nombres complexes dans le plan complexe (Argand).
    
    Paramètres :
        data (list or np.ndarray): liste ou tableau de nombres complexes
        title (str): titre du graphique
        save_name (str or None): si fourni, nom du fichier pour sauvegarde
    """
    data = np.asarray(data)
    x = data.real
    y = data.imag

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', marker='o')
    
    # Axes
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.xscale('linear')
    plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel("Partie réelle")
    plt.ylabel("Partie imaginaire")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.axis('equal')  # Pour que le cercle unité soit rond
    plt.tight_layout()
    plt.show()

def plot_multi_complex_frequencies(datas, labels, title=None, save_name=None):
    """
    Affiche des nombres complexes dans le plan complexe (Argand).
    
    Paramètres :
        data (list or np.ndarray): liste ou tableau de nombres complexes
        title (str): titre du graphique
        save_name (str or None): si fourni, nom du fichier pour sauvegarde
    """
    nb_list = len(datas)
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.3,0.7,nb_list))
    markers = ["s", "x", "*"]
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(datas):
        # data = np.asarray(data)
        x = data.real
        y = data.imag
        plt.scatter(x, y, color=colors[i], label = labels[i], marker=markers[i], s = 8)
    
    # Axes
    # plt.axhline(0, color='gray', linewidth=0.8)
    # plt.axvline(0, color='gray', linewidth=0.8)
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel("Partie réelle")
    plt.ylabel("Partie imaginaire")
    plt.legend(loc = 'best')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    # plt.axis('equal')  # Pour que le cercle unité soit rond
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)


def compute_mean_interaction_terms(mat):
    N = len(mat)
    mean_0 = 0
    for i in range(0,N):
        mean_0 += mat[i][i]
    
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

    return([mean_m2/(N-2), mean_m1/(N-1), mean_0/N, mean_p1/(N-1), mean_p2/(N-2)])

def compute_approx_threshold(R, gamma, delta, v, mat):
    [Cm2,Cm1,C0,Cp1,Cp2] = compute_mean_interaction_terms(mat)
    deriv = self_interaction_derivative(delta, v, R, gamma)
    return( (Cp1 - Cm1)/deriv )



def compute_zeroth_order_frequency_approx_quintdiag(R, gamma, mat, delta, v):
    C_gamma = self_interaction(R, gamma, delta, v)
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    return(C_gamma - (Cm1 + Cp1) + Cm2 + Cp2)


def compute_zeroth_order_frequency_quintdiag_from_mat(R, gamma, mat, delta, v):
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    return(C0 - (Cm1 + Cp1) + Cm2 + Cp2)

def compute_zeroth_order_frequency_approx_tridiag(R,gamma,mat, delta, v):
    C_gamma = self_interaction(R, gamma, delta, v)
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    return(C_gamma - (Cm1 + Cp1))

def compute_zeroth_order_frequency_tridiag_from_mat(R,gamma,mat, delta, v):
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    return(C0 - (Cm1 + Cp1))

def compute_first_order_frequency_approx_tridiag(R,gamma,mat, eps, delta, v):
    dC = self_interaction_derivative(delta, v, R, gamma)
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    zO = compute_zeroth_order_frequency_approx_tridiag(R, gamma, mat, delta, v)
    fO = dC * (Cp1 - Cm1)/(Cp1 + Cm1)
    return(zO - eps * fO)

def compute_first_order_frequency_tridiag_from_mat(R,gamma,mat, eps, delta, v):
    dC = self_interaction_derivative(delta, v, R, gamma)
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    zO = compute_zeroth_order_frequency_tridiag_from_mat(R, gamma, mat, delta, v)
    fO = dC * (Cp1 - Cm1)/(Cp1 + Cm1)
    return(zO - eps * fO)


def compute_second_order_frequency_approx_tridiag(R,gamma,mat, eps, delta, v):
    dC = self_interaction_derivative(delta, v, R, gamma)
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    zO = compute_zeroth_order_frequency_approx_tridiag(R, gamma, mat, delta, v)
    fO = dC * (Cp1 - Cm1)/(Cp1 + Cm1)
    sO = dC * dC / (2 * (Cm1 + Cp1))
    return(zO - eps * fO - eps * eps * sO)

def compute_second_order_frequency_tridiag_from_mat(R,gamma,mat, eps, delta, v):
    dC = self_interaction_derivative(delta, v, R, gamma)
    [Cm2, Cm1, C0, Cp1, Cp2] = compute_mean_interaction_terms(mat)
    zO = compute_zeroth_order_frequency_tridiag_from_mat(R, gamma, mat, delta, v)
    fO = dC * (Cp1 - Cm1)/(Cp1 + Cm1)
    sO = dC * dC / (2 * (Cm1 + Cp1))
    return(zO - eps * fO - eps * eps * sO)

def plot_localized_frequency(freqs, GC_matrices, loc_index, R_list, gamma_list):

    nb_list = len(freqs)
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, (freq, mat) in enumerate(zip(freqs, GC_matrices)):
        gamma_i = gamma_list[i]
        loc_freqi = np.array([np.diag(freq[j])[loc_index-1] for j in range(0,len(freq))])
        zeroth_orderfreq_tri = np.array([compute_zeroth_order_frequency_tridiag_from_mat(R, gamma_i, mat_, delta=1e-5, v=1) for (R, mat_) in zip(R_list, mat)])
        zeroth_orderfreq_quint = np.array([compute_zeroth_order_frequency_quintdiag_from_mat(R, gamma_i, mat_, delta=1e-5, v=1) for (R, mat_) in zip(R_list, mat)])
        # ax[i].plot(R_list, loc_freqi, label = r'data, $\gamma =$'+ f'{gamma_i}', color='k')
        ax[i].plot(R_list, np.abs(loc_freqi-zeroth_orderfreq_tri), label = r'0th order tridiag approx', color='b')
        ax[i].plot(R_list, np.abs(loc_freqi-zeroth_orderfreq_quint), label = r'0th order quintadiag approx', color='g')
        ax[i].set_xscale("linear")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("R")
        ax[i].set_ylabel(r"error $\omega_0(R,\varepsilon)$")
        # ax[i].set_title(r'$\gamma =$'+ f'{gamma_i}')
        ax[i].legend(loc="best")
        ax[i].grid(True)

    fig.suptitle(r"error: $|\omega_0(R,\varepsilon) - (\mathcal{C}^{\gamma}(R) - (\mathcal{C}^{\gamma}_{+1} + \mathcal{C}^{\gamma}_{-1}) + (\mathcal{C}^{\gamma}_{+2} + \mathcal{C}^{\gamma}_{-2}))|$, data SI term.")
    plt.tight_layout()
    plt.show()


def plot_max_freq_eps(eps_lists, skin_mats, deriv_mats, gammas):
    nb_list = len(freqs)
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    plt.figure(figsize=(10,6))
    for i, (eps_list,skin_mat,deriv_mat,gamma) in enumerate(zip(eps_lists,skin_mats,deriv_mats,gammas)):
        max_eig = []
        for eps in eps_list:
            eigvals, right_eigvecs = eig(skin_mat+eps*deriv_mat, left=False, right=True)
            max_eig.append(eigvals.max())

        plt.plot(eps_list, max_eig, label = rf'$\gamma = {gamma}$', color = colors[i], linestyle = "-.")
    plt.xscale('linear')
    plt.xscale('linear')
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$\max \sigma(C^\gamma_s + \varepsilon V)$")
    plt.legend(loc = 'best')
    plt.title('Maximum eigenvalue')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()



def plot_spectrum_vs_epsilon(H0, V, eps_list):
    """
    Affiche le spectre de H(eps) = H0 + eps * V dans le plan complexe
    pour une liste de valeurs de eps.

    Paramètres :
    - H0 : np.ndarray, matrice non hermitienne de base
    - V : np.ndarray, perturbation (même taille que H0)
    - eps_list : array-like, liste de valeurs de epsilon
    """
    eps_list = np.array(eps_list)
    num_eps = len(eps_list)

    # Colormap pour epsilon
    cmap = plt.get_cmap("inferno")
    norm=plt.Normalize(vmin=eps_list.min(), vmax=eps_list.max())

    all_real = []
    all_imag = []
    all_colors = []

    for i, eps in enumerate(eps_list):
        H = H0 + eps * V
        eigvals = np.linalg.eigvals(H)
        all_real.append(eigvals.real)
        all_imag.append(eigvals.imag)
        all_colors.append([cmap(norm(eps))] * len(eigvals))  # associe couleur à chaque point

    # Aplatir tout
    all_real = np.concatenate(all_real)
    all_imag = np.concatenate(all_imag)
    all_colors = np.concatenate(all_colors)
    # Création du plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(all_real, all_imag, c=all_colors, cmap = cmap, s=7)

    # Ajout de la barre de couleur
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # requis pour certaines versions de matplotlib
    fig.colorbar(sm, ax=ax, label=r"$\epsilon$")

    # Mise en forme
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_xscale('linear')
    ax.set_yscale('symlog', linthresh=1e-15)    

    ax.set_title(r"Spectre de $H(\epsilon) = H_0 + \epsilon V$")
    ax.grid(True)
    # ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

def plot_corrected_localized_frequency(freqs, GC_matrices, loc_index, R_list, gamma_list, epss):

    nb_list = len(freqs)
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, (freq, mat, eps) in enumerate(zip(freqs, GC_matrices, epss)):
        gamma_i = gamma_list[i]
        corrected_eps = [epsl - compute_approx_threshold(R,gamma_i,1e-5,1,matR) for (R,epsl,matR) in zip(R_list,eps,mat)]
        loc_freqi = np.array([np.diag(freq[j])[loc_index-1] for j in range(0,len(freq))])
        zeroth_orderfreq_tri = np.array([compute_zeroth_order_frequency_tridiag_from_mat(R, gamma_i, mat_, delta=1e-5, v=1) for (R, mat_) in zip(R_list, mat)])
        first_orderfreq_tri = np.array([compute_first_order_frequency_tridiag_from_mat(R, gamma_i, mat_, eps_, delta=1e-5, v=1) for (R, mat_, eps_) in zip(R_list, mat, corrected_eps)])
        second_orderfreq_tri = np.array([compute_second_order_frequency_tridiag_from_mat(R, gamma_i, mat_, eps_, delta=1e-5, v=1) for (R, mat_, eps_) in zip(R_list, mat, corrected_eps)])
        # zeroth_orderfreq_quint = np.array([compute_zeroth_order_frequency_approx_quintdiag(R, gamma_i, mat_, delta=1e-5, v=1) for (R, mat_) in zip(R_list, mat)])
        # ax[i].plot(R_list, loc_freqi, label = r'data, $\gamma =$'+ f'{gamma_i}', color='k')
        ax[i].plot(R_list, np.abs(loc_freqi - zeroth_orderfreq_tri), label = r'0th order 2scale approx', color='b')
        ax[i].plot(R_list, np.abs(loc_freqi - first_orderfreq_tri), label = r'1st order 2scale approx', color='g')
        ax[i].plot(R_list, np.abs(loc_freqi - second_orderfreq_tri), label = r'2nd order 2scale approx', color='y')
        # ax[i].plot(R_list, zeroth_orderfreq_quint, label = r'0th order $\gamma =$'+ f'{gamma_i}'+',diag = 5', color='g')
        ax[i].set_xscale("linear")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("R")
        ax[i].set_ylabel(r"$\omega(R,\varepsilon)$")
        ax[i].legend(loc="best")
        ax[i].grid(True)

    fig.suptitle(r"Error: $\omega(R,\varepsilon) - (\omega_0(R) + (\varepsilon - \varepsilon_c)\omega_1(R) + (\varepsilon - \varepsilon_c)^2\omega_2(R))$, data SI term")
    plt.tight_layout()
    plt.show()






###GAMMA1

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

eps_gamma1 = [0.00008, 0.00035, 0.0013, 0.0035, 0.008, 0.016, 0.035, 0.078]
GCMs_gamma1 = [GCM_N101_R01_eps_m000008, GCM_N101_R015_eps_m000035, GCM_N101_R02_eps_m00013, GCM_N101_R025_eps_m00035, GCM_N101_R03_eps_m0008, GCM_N101_R035_eps_m0016, GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0078]
Freq_gamma1 = [mat_N101_R01_eps_m000008, mat_N101_R015_eps_m000035, mat_N101_R02_eps_m00013, mat_N101_R025_eps_m00035, mat_N101_R03_eps_m0008, mat_N101_R035_eps_m0016, mat_N101_R04_eps_m0035, mat_N101_R045_eps_m0078]


### GAMMA2


GCM_folder ='./GCM_skin_mat_files/gamma2/'
freq_folder ='./Eigvals_mat_files/gamma2/'

# R = 0.1
GCM_N101_R01_eps_m000013 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m000013.mat", 'GCM_skin','real')
mat_N101_R01_eps_m000013 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m000013.mat', 'eval_skin','real')

# R = 0.15
GCM_N101_R015_eps_m00008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m00008.mat", 'GCM_skin','real')
mat_N101_R015_eps_m00008 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m00008.mat', 'eval_skin','real')

# R = 0.2
GCM_N101_R02_eps_m00027 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m00027.mat", 'GCM_skin','real')
mat_N101_R02_eps_m00027 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m00027.mat', 'eval_skin','real')

# R = 0.25
GCM_N101_R025_eps_m0007 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m0007.mat", 'GCM_skin','real')
mat_N101_R025_eps_m0007 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m0007.mat', 'eval_skin','real')

# R = 0.3
GCM_N101_R03_eps_m0015 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0015.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0015 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m0015.mat', 'eval_skin','real')

# R = 0.35
GCM_N101_R035_eps_m0031 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m0031.mat", 'GCM_skin','real')
mat_N101_R035_eps_m0031 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m0031.mat', 'eval_skin','real')

# R = 0.4
GCM_N101_R04_eps_m0058 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m0058.mat", 'GCM_skin','real')
mat_N101_R04_eps_m0058 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0058.mat', 'eval_skin','real')

# R = 0.45
GCM_N101_R045_eps_m0107 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0107.mat", 'GCM_skin','real')
mat_N101_R045_eps_m0107 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0107.mat', 'eval_skin','real')

eps_gamma2 = [0.00013, 0.0008, 0.0027, 0.007, 0.015, 0.031, 0.058, 0.107]
GCMs_gamma2 = [GCM_N101_R01_eps_m000013, GCM_N101_R015_eps_m00008, GCM_N101_R02_eps_m00027, GCM_N101_R025_eps_m0007, GCM_N101_R03_eps_m0015, GCM_N101_R035_eps_m0031, GCM_N101_R04_eps_m0058, GCM_N101_R045_eps_m0107]
Freq_gamma2 = [mat_N101_R01_eps_m000013, mat_N101_R015_eps_m00008, mat_N101_R02_eps_m00027, mat_N101_R025_eps_m0007, mat_N101_R03_eps_m0015, mat_N101_R035_eps_m0031, mat_N101_R04_eps_m0058, mat_N101_R045_eps_m0107]



GCM_folder ='./GCM_skin_mat_files/gamma05/'
freq_folder ='./Eigvals_mat_files/gamma05/'

# R = 0.1
#eps0
GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
mat_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')

GCM_N101_R01_eps_m00000485 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m00000485.mat", 'GCM_skin','real')
mat_N101_R01_eps_m00000485 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m00000485.mat', 'eval_skin','none')

# R = 0.15
#eps0
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')

GCM_N101_R015_eps_m0000245 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m0000245.mat", 'GCM_skin','real')
mat_N101_R015_eps_m0000245 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m0000245.mat', 'eval_skin','none')

# R = 0.2
#eps0
GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')

GCM_N101_R02_eps_m000074 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m000074.mat", 'GCM_skin','real')
mat_N101_R02_eps_m000074 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m000074.mat', 'eval_skin','none')

GCM_N101_R02_eps_m00005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00005.mat', 'GCM_skin','real')
deriv_mat_R02_gamma05 = (GCM_N101_R02_eps0 - GCM_N101_R02_eps_m00005)/(0.0005)

# R = 0.25
#eps0
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')

GCM_N101_R025_eps_m00018 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m00018.mat", 'GCM_skin','real')
mat_N101_R025_eps_m00018 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m00018.mat', 'eval_skin','none')

# R = 0.3
#eps0
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')

GCM_N101_R03_eps_m0003945 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0003945.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0003945 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m0003945.mat', 'eval_skin','none')

# R = 0.35
#eps0
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')

GCM_N101_R035_eps_m000853 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m000853.mat", 'GCM_skin','real')
mat_N101_R035_eps_m000853 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m000853.mat', 'eval_skin','none')

# R = 0.4
#eps0
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')

GCM_N101_R04_eps_m00198 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m00198.mat", 'GCM_skin','real')
mat_N101_R04_eps_m00198 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m00198.mat', 'eval_skin','none')

# R = 0.45
#eps0
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')

GCM_N101_R045_eps_m0059 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0059.mat", 'GCM_skin','real')
mat_N101_R045_eps_m0059 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0059.mat', 'eval_skin','none')

eps_gamma05 = [-0.0000485, -0.000245, -0.00074, -0.0018, -0.003945, -0.00853, -0.0198, -0.059]
GCMs_gamma05 = [GCM_N101_R01_eps_m00000485, GCM_N101_R015_eps_m0000245, GCM_N101_R02_eps_m000074, GCM_N101_R025_eps_m00018, GCM_N101_R03_eps_m0003945, GCM_N101_R035_eps_m000853, GCM_N101_R04_eps_m00198, GCM_N101_R045_eps_m0059]
Freq_gamma05 = [mat_N101_R01_eps_m00000485, mat_N101_R015_eps_m0000245, mat_N101_R02_eps_m000074, mat_N101_R025_eps_m00018, mat_N101_R03_eps_m0003945, mat_N101_R035_eps_m000853, mat_N101_R04_eps_m00198, mat_N101_R045_eps_m0059]

Skin_mats_gamma05 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]


# eps = [eps_gamma05, eps_gamma1, eps_gamma2]
eps = [eps_gamma05,eps_gamma05]
# GCMs = [GCMs_gamma05, GCMs_gamma1, GCMs_gamma2]
GCMs = [GCMs_gamma05,GCMs_gamma05]
# Freqs = [Freq_gamma05, Freq_gamma1, Freq_gamma2]
Freqs = [Freq_gamma05,Freq_gamma05]
Rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
# gammas = [0.5, 1, 2]
gammas = [0.5,0.5]

# plot_localized_frequency(Freqs, GCMs, 101, Rs, gammas)

# plot_corrected_localized_frequency(Freqs, GCMs, 101, Rs, gammas, eps)

# print(GCM_N101_R04_eps_m0035[0])
# list_derivative = [self_interaction_derivative(1e-5, 1, R, 1) for R in Rs]
# list = [self_interaction(R, 1, 1e-5, 1,) for R in Rs]
# # plt.plot(Rs, list_derivative)
# plt.plot(Rs, list)
# plt.show()



complex_freq02 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m002.mat', 'eval_skin','none')
complex_freq021 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0021.mat', 'eval_skin','none')
complex_freq019 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0019.mat', 'eval_skin','none')


freqs = [mat_N101_R025_eps0, mat_N101_R025_eps_m00018]
# labels = ["under", "localized", "over"]
# plot_multi_complex_frequencies(freqs,["skin","transition"],None,None)

# plot_complex_frequencies(complex_freq, None, None)

# plot_max_freq_eps([np.linspace(-0.001, 0, 100)], [Skin_mats_gamma05[2]], [deriv_mat_R02_gamma05], [0.5])
plot_spectrum_vs_epsilon(Skin_mats_gamma05[2], deriv_mat_R02_gamma05, np.linspace(-0.001, 0, 100))
print(Skin_mats_gamma05[2])