import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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


def compute_IPR(mode):
    n = len(mode)
    num = 0
    denom = 0
    # mode = np.abs(mode)/np.linalg.norm(mode, np.inf)
    mode = np.abs(mode)
    for i in range(0,n):
        num += np.abs(mode[i])**4
        denom += np.abs(mode[i])**2
    IPR = num / (denom**2)
    return(IPR)


def self_interaction_derivative(delta, v, R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = delta * v * v * gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)



def plot_localized_modes(modes, loc_index, labels, title = None, save_name = None):
    N_points, N_modes = modes[0].shape
    nb_list = len(modes)
    print(nb_list)
    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, mode in enumerate(modes):
        # ax[i].plot(x, np.abs(mode[:,loc_index-1])/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
        ax[i].plot(x, np.abs(mode[:,loc_index-1]), color=colors[i], label = labels[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
        # ax[i].set_ylim(-1.2, 1.2)
        ax[i].set_xscale("linear")
        ax[i].set_yscale("linear")
        ax[i].set_xlabel("Index")
        ax[i].set_ylabel("Mode value")
        ax[i].legend(loc="best")
        ax[i].grid(True)

    fig.suptitle(title)
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
        data = np.diagonal(data)
        x = data.real
        y = data.imag
        print(x[17])
        print(x[18])
        plt.scatter(x, y, color=colors[i], label = labels[i], marker=markers[i], s = 8)
        for j, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, str(j), fontsize=9, ha='right', va='bottom')     
    # Axes
    # plt.axhline(0, color='gray', linewidth=0.8)
    # plt.axvline(0, color='gray', linewidth=0.8)
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel(r"$\Re(\omega)$")
    plt.ylabel(r"$\Im(\omega)$")
    plt.legend(loc = 'best')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    # plt.axis('equal')  # Pour que le cercle unité soit rond
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)

def plot_spectrum_vs_epsilon_winding(H0, V, eps_list):
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
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno", cmap(np.linspace(0.2, 0.8, 256)))
    norm=plt.Normalize(vmin=eps_list.min(), vmax=eps_list.max())

    all_real = []
    all_imag = []
    all_colors = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, eps in enumerate(eps_list):
        H = H0 + eps * V
        eigvals = np.linalg.eigvals(H)
        idx = np.argsort(eigvals.real)
        eigvals = eigvals[idx]
        if eps == 0:
            for idx, val in enumerate(eigvals):
                ax.text(val.real, val.imag, str(idx), fontsize=9, ha='right', va='bottom')
        all_real.append(eigvals.real)
        all_imag.append(eigvals.imag)
        all_colors.append([trunccmap(norm(eps))] * len(eigvals))  # associe couleur à chaque point

    # Aplatir tout
    all_real = np.concatenate(all_real)
    all_imag = np.concatenate(all_imag)
    all_colors = np.concatenate(all_colors)

    # -------- Construction du symbole de H0 --------
    N = H0.shape[0]
    diagonals = {}
    for i in range(-N + 1, N):
        diag = np.diag(H0, k=i)
        if len(diag) > 0:
            diagonals[i] = np.mean(diag)  # moyenne sur diagonale

    # Symbol h(k)
    k_vals = np.linspace(0, 2 * np.pi, 1000)
    f_k = np.zeros_like(k_vals, dtype=complex)
    for n, val in diagonals.items():
        f_k += val * np.exp(1j * n * k_vals)

    
    # Création du plot
    scatter = ax.scatter(all_real, all_imag, c=all_colors, cmap = trunccmap, s=7)
    ax.plot(f_k.real, f_k.imag, color='k', lw=1, label=r'Toeplitz symbol $f(s)$')

    # Ajout de la barre de couleur
    sm = cm.ScalarMappable(cmap=trunccmap, norm=norm)
    sm.set_array([])  # requis pour certaines versions de matplotlib
    fig.colorbar(sm, ax=ax, label=r"$\varepsilon$")

    # Mise en forme
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_xscale('linear')
    ax.set_yscale('symlog', linthresh=1e-15)    

    ax.set_title(r"Spectre de $C^\gamma_d(\varepsilon) = C^\gamma_s + \varepsilon V$")
    ax.grid(True)
    ax.legend()
    # ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def plot_SR_frequencies_error(datas, predictions, labels, title=None, save_name=None):
    """
    Affiche des nombres complexes dans le plan complexe (Argand).
    
    Paramètres :
        data (list or np.ndarray): liste ou tableau de nombres complexes
        title (str): titre du graphique
        save_name (str or None): si fourni, nom du fichier pour sauvegarde
    """
    nb_list = len(datas)
    nb_freq = len(datas[0])
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*"]
    plt.figure(figsize=(10, 6))
    for i, (data, pred) in enumerate(zip(datas,predictions)):
        data = np.diag(data)
        pred = np.diag(pred)
        plt.plot(np.arange(1,nb_freq+1), np.abs(data-pred)/np.abs(data), color=colors[i], label = labels[i], marker=markers[i], markersize = 3)
        # plt.plot(np.arange(0,nb_freq), np.abs(data-pred), color=colors[i], label = labels[i], marker=markers[i], markersize = 3)
    
    # Axes
    # plt.axhline(0, color='gray', linewidth=0.8)
    # plt.axvline(0, color='gray', linewidth=0.8)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"frequency index $k$")
    plt.ylabel(r"$|\omega^k_\varepsilon - \omega^1_{SR}(\varepsilon)| / |\omega^k_\varepsilon|$")
    plt.legend(loc = 'best')
    plt.title(title)
    plt.minorticks_on()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)

def plot_SR_modes_error(datas, predictions, index, labels, title=None, save_name=None):
    """
    Affiche des nombres complexes dans le plan complexe (Argand).
    
    Paramètres :
        data (list or np.ndarray): liste ou tableau de nombres complexes
        title (str): titre du graphique
        save_name (str or None): si fourni, nom du fichier pour sauvegarde
    """
    nb_list = len(datas)
    nb_freq = len(datas[0])
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.3,0.7,nb_list))
    markers = ["s", "x", "*"]
    plt.figure(figsize=(10, 6))
    for i, (data, pred) in enumerate(zip(datas,predictions)):
        data = data[index]
        pred = pred[index]
        # plt.plot(np.arange(0,nb_freq), np.abs(data-pred)/np.abs(data), color=colors[i], label = labels[i], marker=markers[i], markersize = 3)
        plt.plot(np.arange(0,nb_freq), np.abs(data-pred), color=colors[i], label = labels[i], marker=markers[i], markersize = 3)
    
    # Axes
    # plt.axhline(0, color='gray', linewidth=0.8)
    # plt.axvline(0, color='gray', linewidth=0.8)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"index")
    plt.ylabel(r"$|\psi^k_\varepsilon - \psi(SR)| / |\psi^k_\varepsilon|$")
    plt.legend(loc = 'best')
    plt.title(title)
    plt.minorticks_on()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)

def plot_IPR(modes, gammas):
    plt.figure(figsize=(10,6))
    nb_list = len(gammas)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    for i, (modes_gamma,gamma) in enumerate(zip(modes,gammas)):
        IPRS = []
        index = np.arange(1,len(modes_gamma)+1)
        for k in range(0,len(modes_gamma)):
            IPR = compute_IPR(modes_gamma[:,k])
            IPRS.append(IPR)
        plt.plot(index, IPRS, label = rf'$\gamma = {gamma}$', color = colors[i], linestyle = "-.")
    plt.xscale('linear')
    plt.xscale('linear')
    plt.xlabel(r"index k")
    plt.ylabel(r"$IPR(\psi_k)$")
    plt.legend(loc = 'best')
    plt.title('Inverse participation ratio')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def compute_SR_eigenfrequencies(skin_mat, deriv_mat, eps):

    # Calcul des valeurs propres et des vecteurs propres droits et gauches
    eigvals, right_eigvecs = eig(skin_mat, left=False, right=True)
    eigvals_left, left_eigvecs = eig(skin_mat, left=True, right=False)
    
    # Normalisation biorthogonale des vecteurs propres
    # Attention : scipy ne normalise pas les vecteurs propres !
    N = len(eigvals)
    corrections = np.zeros(N, dtype=np.complex128)
    idx = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[idx]
    right_sorted = right_eigvecs[:, idx]
    left_sorted = left_eigvecs[:, idx]

    for k in range(N):
        Rk = right_sorted[:, k]
        Lk = left_sorted[:, k]

        # Produit scalaire biorthogonal sans vdot
        norm_factor = np.dot(Lk.conj().T, Rk)
        correction = np.dot(Lk.conj().T, deriv_mat @ Rk) / norm_factor
        corrections[k] = eps * correction

    SR_eigs = eigvals_sorted + corrections
    # print(corrections)
    idx = np.argsort(SR_eigs.real)
    return np.diag(SR_eigs)

def compute_SR_modes(skin_mat, deriv_mat, eps):
    # Diagonalisation de H0 (valeurs propres, vecteurs propres droits et gauches)
    eigvals, right_eigvecs = eig(skin_mat, left=False, right=True)
    eigvals_left, left_eigvecs = eig(skin_mat, left=True, right=False)

    # Dimensions
    N = skin_mat.shape[0]
    idx = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[idx]
    right_sorted = right_eigvecs[:, idx]
    left_sorted = left_eigvecs[:, idx]
    eig_modes = np.zeros((N, N), dtype=np.complex128)
    for k in range(N):
        # Vecteur propre droit non perturbé
        R0_k = right_sorted[:, k].copy()
        
        # Correction de premier ordre
        correction = np.zeros(N, dtype=np.complex128)
        for l in range(N):
            if l != k:
                Ll = left_sorted[:, l]
                Rl = right_sorted[:, l]
                num = np.vdot(Ll, deriv_mat @ R0_k)
                # denom = (eigvals_sorted[k] - eigvals_sorted[l])
                denom = (eigvals_sorted[k] - eigvals_sorted[l]) * np.vdot(Ll,Rl)
                correction += (num / denom) * right_sorted[:, l]

        # Vecteur propre perturbé : ordre 0 + correction d'ordre 1
        eig_modes[:, k] = R0_k + eps * correction
    return(eig_modes)

def plot_SR_multi_frequency_error(skin_mats_gamma, deriv_mats_gamma, freq_datas_gamma, eps_datas_gamma, Rs, gammas):
    nb_gamma = len(gammas)
    colors = plt.get_cmap("inferno")(np.linspace(0.25,0.75,nb_gamma))
    plt.figure(figsize=(10,6))
    
    for i, (skin_mats,deriv_mats, eps_datas, freq_datas, gamma) in enumerate(zip(skin_mats_gamma, deriv_mats_gamma, eps_datas_gamma, freq_datas_gamma, gammas)):
        error_R_gamma = []
        for (skin_mat, deriv_mat, eps_data, freq_data) in zip(skin_mats, deriv_mats, eps_datas, freq_datas):
            predicted_freqs = compute_SR_eigenfrequencies(skin_mat,deriv_mat, eps_data)
            sup_error = np.max(np.abs(np.diag(predicted_freqs.real - freq_data.real))/np.abs(np.diag(freq_data)))
            error_R_gamma.append(sup_error)
        plt.plot(Rs, error_R_gamma, color = colors[i], label = r'$\gamma=$' + f'{gamma}')

    plt.legend(loc="best")
    plt.title("Self interaction derivative: model vs data")
    plt.xlabel(r"$R$")
    plt.ylabel(r"$\sup_k|\omega_k(\varepsilon)-\omega_k(SR)|/|\omega_k(\varepsilon)|$")
    plt.xscale("linear")
    plt.yscale("log")
    plt.minorticks_on()
    # # Définir le minor locator manuellement pour le symlog
    # minor_locator = SymmetricalLogLocator(base=10, linthresh=1e-7, subs=np.arange(2, 10)*0.1)
    # plt.gca().yaxis.set_minor_locator(minor_locator)    
    plt.grid(which="major")
    plt.grid(True, which="minor", linestyle = ":", alpha = 0.8)
    plt.tight_layout()
    plt.show()


###=========================================GAMMA05===============================================
GCM_folder ='./GCM_skin_mat_files/gamma05/'
mode_folder ='./Modes_mat_files/gamma05/'
freq_folder ='./Eigvals_mat_files/gamma05/'
###========================eps0===================================

mat_N101_R01_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R01_eps0.mat', 'modes_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R015_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R025_eps0.mat', 'modes_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R03_eps0.mat', 'modes_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R035_eps0.mat', 'modes_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R04_eps0.mat', 'modes_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R045_eps0.mat', 'modes_skin','real')
modes_skin_gamma05 = [mat_N101_R01_eps0, mat_N101_R015_eps0, mat_N101_R02_eps0, mat_N101_R025_eps0, mat_N101_R03_eps0, mat_N101_R035_eps0, mat_N101_R04_eps0, mat_N101_R045_eps0]

GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
GCM_N101_R01_eps_m000001 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m000001.mat", 'GCM_skin','real')
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
GCM_N101_R015_eps_m00001 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m00001.mat", 'GCM_skin','real')
GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
GCM_N101_R02_eps_m00005 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m00005.mat", 'GCM_skin','real')
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
GCM_N101_R025_eps_m00015 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m00015.mat", 'GCM_skin','real')
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
GCM_N101_R03_eps_m00039 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m00039.mat", 'GCM_skin','real')
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
GCM_N101_R035_eps_m0008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m0008.mat", 'GCM_skin','real')
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
GCM_N101_R04_eps_m0019 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m0019.mat", 'GCM_skin','real')
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
GCM_N101_R045_eps_m005 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m005.mat", 'GCM_skin','real')

GCM_skin_gamma05 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]
GCM_defect_gamma05 = [GCM_N101_R01_eps_m000001, GCM_N101_R015_eps_m00001, GCM_N101_R02_eps_m00005, GCM_N101_R025_eps_m00015, GCM_N101_R03_eps_m00039, GCM_N101_R035_eps_m0008, GCM_N101_R04_eps_m0019, GCM_N101_R045_eps_m005]
eps_defect_gamma05 = [-0.00001, -0.0001, -0.0005, -0.0015, -0.0039, -0.008, -0.019, -0.05]
deriv_skin_gamma05 = [(skin-defect)/np.abs(eps) for (defect,skin,eps) in zip(GCM_defect_gamma05,GCM_skin_gamma05,eps_defect_gamma05)]


freq_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')
freq_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')
freq_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')
freq_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')
freq_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')
freq_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')
freq_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')
freq_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')
freq_skin_gamma05 = [freq_N101_R01_eps0, freq_N101_R015_eps0, freq_N101_R02_eps0, freq_N101_R025_eps0, freq_N101_R03_eps0, freq_N101_R035_eps0, freq_N101_R04_eps0, freq_N101_R045_eps0]



###=========================================GAMMA1===============================================
GCM_folder ='./GCM_skin_mat_files/gamma1/'
mode_folder ='./Modes_mat_files/gamma1/'
freq_folder ='./Eigvals_mat_files/gamma1/'
###========================eps0===================================

mat_N101_R01_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R01_eps0.mat', 'modes_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R015_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R025_eps0.mat', 'modes_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R03_eps0.mat', 'modes_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R035_eps0.mat', 'modes_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R04_eps0.mat', 'modes_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R045_eps0.mat', 'modes_skin','real')
modes_skin_gamma1 = [mat_N101_R01_eps0, mat_N101_R015_eps0, mat_N101_R02_eps0, mat_N101_R025_eps0, mat_N101_R03_eps0, mat_N101_R035_eps0, mat_N101_R04_eps0, mat_N101_R045_eps0]

GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
GCM_N101_R01_eps_m000008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m000008.mat", 'GCM_skin','real')
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
GCM_N101_R015_eps_m000035 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m000035.mat", 'GCM_skin','real')
GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
GCM_N101_R02_eps_m00013 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m00013.mat", 'GCM_skin','real')
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
GCM_N101_R025_eps_m00034 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m00034.mat", 'GCM_skin','real')
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
GCM_N101_R03_eps_m0008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0008.mat", 'GCM_skin','real')
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
GCM_N101_R035_eps_m0016 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m0016.mat", 'GCM_skin','real')
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
GCM_N101_R04_eps_m0035 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m0035.mat", 'GCM_skin','real')
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
GCM_N101_R045_eps_m0078 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0078.mat", 'GCM_skin','real')
GCM_skin_gamma1 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]
GCM_defect_gamma1 = [GCM_N101_R01_eps_m000008, GCM_N101_R015_eps_m000035, GCM_N101_R02_eps_m00013, GCM_N101_R025_eps_m00034, GCM_N101_R03_eps_m0008, GCM_N101_R035_eps_m0016, GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0078]
eps_defect_gamma1 = [-0.00008, -0.00035, -0.0013, -0.0034, -0.008, -0.016, -0.035, -0.078]
deriv_skin_gamma1 = [(skin-defect)/np.abs(eps) for (defect,skin,eps) in zip(GCM_defect_gamma1,GCM_skin_gamma1,eps_defect_gamma1)]


freq_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')
freq_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')
freq_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')
freq_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')
freq_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')
freq_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')
freq_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')
freq_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')
freq_skin_gamma1 = [freq_N101_R01_eps0, freq_N101_R015_eps0, freq_N101_R02_eps0, freq_N101_R025_eps0, freq_N101_R03_eps0, freq_N101_R035_eps0, freq_N101_R04_eps0, freq_N101_R045_eps0]


###=========================================GAMMA2===============================================
GCM_folder ='./GCM_skin_mat_files/gamma2/'
mode_folder ='./Modes_mat_files/gamma2/'
freq_folder ='./Eigvals_mat_files/gamma2/'
###========================eps0===================================

mat_N101_R01_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R01_eps0.mat', 'modes_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R015_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R025_eps0.mat', 'modes_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R03_eps0.mat', 'modes_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R035_eps0.mat', 'modes_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R04_eps0.mat', 'modes_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R045_eps0.mat', 'modes_skin','real')
modes_skin_gamma2 = [mat_N101_R01_eps0, mat_N101_R015_eps0, mat_N101_R02_eps0, mat_N101_R025_eps0, mat_N101_R03_eps0, mat_N101_R035_eps0, mat_N101_R04_eps0, mat_N101_R045_eps0]

GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
GCM_N101_R01_eps_m00001 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m00001.mat", 'GCM_skin','real')
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
GCM_N101_R015_eps_m00005 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m00005.mat", 'GCM_skin','real')
GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
GCM_N101_R02_eps_m0003 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m0003.mat", 'GCM_skin','real')
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
GCM_N101_R025_eps_m0005 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m0005.mat", 'GCM_skin','real')
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
GCM_N101_R03_eps_m001 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m001.mat", 'GCM_skin','real')
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
GCM_N101_R035_eps_m002 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m002.mat", 'GCM_skin','real')
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
GCM_N101_R04_eps_m005 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m005.mat", 'GCM_skin','real')
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
GCM_N101_R045_eps_m01 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m01.mat", 'GCM_skin','real')
GCM_skin_gamma2 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]
GCM_defect_gamma2 = [GCM_N101_R01_eps_m00001, GCM_N101_R015_eps_m00005, GCM_N101_R02_eps_m0003, GCM_N101_R025_eps_m0005, GCM_N101_R03_eps_m001, GCM_N101_R035_eps_m002, GCM_N101_R04_eps_m005, GCM_N101_R045_eps_m01]
eps_defect_gamma2 = [-0.0001, -0.0005, -0.003, -0.005, -0.01, -0.02, -0.05, -0.1]
deriv_skin_gamma2 = [(skin-defect)/np.abs(eps) for (defect,skin,eps) in zip(GCM_defect_gamma2,GCM_skin_gamma2,eps_defect_gamma2)]


freq_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')
freq_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')
freq_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')
freq_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')
freq_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')
freq_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')
freq_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')
freq_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')
freq_skin_gamma2 = [freq_N101_R01_eps0, freq_N101_R015_eps0, freq_N101_R02_eps0, freq_N101_R025_eps0, freq_N101_R03_eps0, freq_N101_R035_eps0, freq_N101_R04_eps0, freq_N101_R045_eps0]



#=============================================COMPUTATIONS============================================================
### DATAS TO CHECK GAMMA05
freq_folder_gamma05 ='./Eigvals_mat_files/gamma05/'
freq_N101_R01_eps_m000007 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R01_eps_m000007.mat', 'eval_skin','none')
freq_N101_R015_eps_m000024 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R015_eps_m000024.mat', 'eval_skin','none')
freq_N101_R02_eps_m00008 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R02_eps_m00008.mat', 'eval_skin','none')
freq_N101_R025_eps_m00019 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R025_eps_m00019.mat', 'eval_skin','none')
freq_N101_R03_eps_m000395 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R03_eps_m000395.mat', 'eval_skin','none')
freq_N101_R035_eps_m0009 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R035_eps_m0009.mat', 'eval_skin','none')
freq_N101_R04_eps_m0021 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R04_eps_m0021.mat', 'eval_skin','none')
freq_N101_R045_eps_m0055 = read_hdf5_modes(freq_folder_gamma05 + 'eval_skin_N_101_R045_eps_m0055.mat', 'eval_skin','none')

freq_mats_gamma05 = [freq_N101_R01_eps_m000007, freq_N101_R015_eps_m000024, freq_N101_R02_eps_m00008, freq_N101_R025_eps_m00019, freq_N101_R03_eps_m000395, freq_N101_R035_eps_m0009, freq_N101_R04_eps_m0021, freq_N101_R045_eps_m0055]
eps_data_gamma05 = [-0.00007, -0.00024, -0.0008, -0.0019, -0.00395, -0.009, -0.021, -0.055]

### DATAS TO CHECK GAMMA1
freq_folder_gamma1 ='./Eigvals_mat_files/gamma1/'
freq_N101_R01_eps_m000009 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R01_eps_m000009.mat', 'eval_skin','none')
freq_N101_R015_eps_m000045 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R015_eps_m000045.mat', 'eval_skin','none')
freq_N101_R02_eps_m00013 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R02_eps_m00013.mat', 'eval_skin','none')
freq_N101_R025_eps_m0004 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R025_eps_m0004.mat', 'eval_skin','none')
freq_N101_R03_eps_m00078 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R03_eps_m00078.mat', 'eval_skin','none')
freq_N101_R035_eps_m0015 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R035_eps_m0015.mat', 'eval_skin','none')
freq_N101_R04_eps_m004 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R04_eps_m004.mat', 'eval_skin','none')
freq_N101_R045_eps_m0075 = read_hdf5_modes(freq_folder_gamma1 + 'eval_skin_N_101_R045_eps_m0075.mat', 'eval_skin','none')

freq_mats_gamma1 = [freq_N101_R01_eps_m000009, freq_N101_R015_eps_m000045, freq_N101_R02_eps_m00013, freq_N101_R025_eps_m0004, freq_N101_R03_eps_m00078, freq_N101_R035_eps_m0015, freq_N101_R04_eps_m004, freq_N101_R045_eps_m0075]
eps_data_gamma1 = [-0.00009, -0.00045, -0.0013, -0.004, -0.0078, -0.015, -0.04, -0.075]


### DATAS TO CHECK GAMMA2
freq_folder_gamma2 ='./Eigvals_mat_files/gamma2/'
freq_N101_R01_eps_m00005 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R01_eps_m00005.mat', 'eval_skin','none')
freq_N101_R015_eps_m00009 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R015_eps_m00009.mat', 'eval_skin','none')
freq_N101_R02_eps_m0003 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R02_eps_m0003.mat', 'eval_skin','none')
freq_N101_R025_eps_m0006 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R025_eps_m0006.mat', 'eval_skin','none')
freq_N101_R03_eps_m0015 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R03_eps_m0015.mat', 'eval_skin','none')
freq_N101_R035_eps_m0025 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R035_eps_m0025.mat', 'eval_skin','none')
freq_N101_R04_eps_m006 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R04_eps_m006.mat', 'eval_skin','none')
freq_N101_R045_eps_m011 = read_hdf5_modes(freq_folder_gamma2 + 'eval_skin_N_101_R045_eps_m011.mat', 'eval_skin','none')

freq_mats_gamma2 = [freq_N101_R01_eps_m00005, freq_N101_R015_eps_m00009, freq_N101_R02_eps_m0003, freq_N101_R025_eps_m0006, freq_N101_R03_eps_m0015, freq_N101_R035_eps_m0025, freq_N101_R04_eps_m006, freq_N101_R045_eps_m011]
eps_data_gamma2 = [-0.0005, -0.0009, -0.003, -0.006, -0.05, -0.015, -0.06, -0.11]


skin_mats_gammas = [GCM_skin_gamma05, GCM_skin_gamma1, GCM_skin_gamma2]
deriv_mats_gammas = [deriv_skin_gamma05, deriv_skin_gamma1, deriv_skin_gamma2]
freqs_mats_datas = [freq_mats_gamma05, freq_mats_gamma1, freq_mats_gamma2]
eps_datas = [eps_data_gamma05, eps_data_gamma1, eps_data_gamma2]
Rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
gammas = [0.5, 1, 2]


# plot_SR_multi_frequency_error(skin_mats_gammas, deriv_mats_gammas, freqs_mats_datas, eps_datas, Rs, gammas)



GCM_N101_R02_eps_m00008 = read_hdf5_modes('./GCM_skin_mat_files/gamma05/'+'GCM_skin_N_101_R02_eps_m00008.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00013 = read_hdf5_modes('./GCM_skin_mat_files/gamma1/'+'GCM_skin_N_101_R02_eps_m00013.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00027 = read_hdf5_modes('./GCM_skin_mat_files/gamma2/'+'GCM_skin_N_101_R02_eps_m00027.mat', 'GCM_skin','real')
freq_N101_R02_eps_m00008, modes_N101_R02_eps_m00008 = eig(GCM_N101_R02_eps_m00008, right=True)
idx = np.argsort(freq_N101_R02_eps_m00008.real)
freq_N101_R02_eps_m00008 = freq_N101_R02_eps_m00008[idx]

freq_N101_R02_eps_m00013, modes_N101_R02_eps_m00013 = eig(GCM_N101_R02_eps_m00013, right=True)
idx = np.argsort(freq_N101_R02_eps_m00013.real)
freq_N101_R02_eps_m00013 = freq_N101_R02_eps_m00013[idx]
modes_N101_R02_eps_m00013 = modes_N101_R02_eps_m00013[:,idx]

freq_N101_R02_eps_m00027, modes_N101_R02_eps_m00027 = eig(GCM_N101_R02_eps_m00027, right=True)
idx = np.argsort(freq_N101_R02_eps_m00027.real)
freq_N101_R02_eps_m00027 = freq_N101_R02_eps_m00027[idx]

freq_N101_R02_eps_m00008 = np.diag(freq_N101_R02_eps_m00008)
freq_N101_R02_eps_m00013 = np.diag(freq_N101_R02_eps_m00013)
freq_N101_R02_eps_m00027 = np.diag(freq_N101_R02_eps_m00027)
# freq_N101_R02_eps_m00008 = read_hdf5_modes('./Eigvals_mat_files/gamma05/'+'eval_skin_N_101_R02_eps_m00008.mat', 'eval_skin','none')
# freq_N101_R02_eps_m00013 = read_hdf5_modes('./Eigvals_mat_files/gamma1/'+'eval_skin_N_101_R02_eps_m00013.mat', 'eval_skin','none')
# freq_N101_R02_eps_m00027 = read_hdf5_modes('./Eigvals_mat_files/gamma2/'+'eval_skin_N_101_R02_eps_m00027.mat', 'eval_skin','none')


predicted_freq_gamma05 = compute_SR_eigenfrequencies(GCM_skin_gamma05[2], deriv_skin_gamma05[2],-0.0008)
predicted_freq_gamma1 = compute_SR_eigenfrequencies(GCM_skin_gamma1[2], deriv_skin_gamma1[2],-0.0013)
predicted_freq_gamma2 = compute_SR_eigenfrequencies(GCM_skin_gamma2[2], deriv_skin_gamma2[2],-0.0027)
# predicted_freq_gamma05 = compute_SR_eigenfrequencies(GCM_skin_gamma05[6], deriv_skin_gamma05[6],-0.02)
# predicted_freq_gamma1 = compute_SR_eigenfrequencies(GCM_skin_gamma1[6], deriv_skin_gamma1[6],-0.04)
# predicted_freq_gamma2 = compute_SR_eigenfrequencies(GCM_skin_gamma2[6], deriv_skin_gamma2[6],-0.07)

# plot_SR_frequencies_error([freq_N101_R02_eps_m00008,freq_N101_R02_eps_m00013,freq_N101_R02_eps_m00027], [predicted_freq_gamma05, predicted_freq_gamma1, predicted_freq_gamma2], [r" $\gamma=0.5$", r" $\gamma=1.0$", r" $\gamma=2.0$"], "Relative error for defect eigenvalue prediction", None)

# plot_multi_complex_frequencies([predicted_freq_gamma05, freq_N101_R02_eps_m00008],["predicted","data"], "Eigenvalues in complex plane", None)
# plot_multi_complex_frequencies([predicted_freq_gamma1, freq_N101_R02_eps_m00013],["predicted","data"], "Eigenvalues in complex plane", None)
# plot_multi_complex_frequencies([predicted_freq_gamma2, freq_N101_R02_eps_m00027],["predicted","data"], "Eigenvalues in complex plane", None)





# modes_N101_R02_eps_m00008 = read_hdf5_modes('./Modes_mat_files/gamma05/'+'modes_skin_N_101_R02_eps_m00008.mat', 'modes_skin','none')
# modes_N101_R02_eps_m00013 = read_hdf5_modes('./Modes_mat_files/gamma1/'+'modes_skin_N_101_R02_eps_m00013.mat', 'modes_skin','none')
# modes_N101_R02_eps_m00027 = read_hdf5_modes('./Modes_mat_files/gamma2/'+'modes_skin_N_101_R02_eps_m00027.mat', 'modes_skin','none')

predicted_modes_gamma05 = compute_SR_modes(GCM_skin_gamma05[2],deriv_skin_gamma05[2],-0.0008)
predicted_modes_gamma1 = compute_SR_modes(GCM_skin_gamma1[2],deriv_skin_gamma1[2],-0.0013)
predicted_modes_gamma2 = compute_SR_modes(GCM_skin_gamma2[2],deriv_skin_gamma2[2],-0.0027)

# plot_SR_modes_error([modes_N101_R02_eps_m00008,modes_N101_R02_eps_m00013,modes_N101_R02_eps_m00027], [predicted_modes_gamma05, predicted_modes_gamma1, predicted_modes_gamma2], 100, [r" $\gamma=0.5$", r" $\gamma=1.0$", r" $\gamma=2.0$"], "Relative error for mode prediction", None)


plot_IPR([modes_N101_R02_eps_m00013, predicted_modes_gamma1], [1, 2])

plot_localized_modes([modes_N101_R02_eps_m00013, predicted_modes_gamma1], 78, ["data", "predicted"], None, None)
plot_localized_modes([modes_N101_R02_eps_m00013, predicted_modes_gamma1], 79, ["data", "predicted"], None, None)
plot_localized_modes([modes_N101_R02_eps_m00013, predicted_modes_gamma1], 80, ["data", "predicted"], None, None)
plot_localized_modes([modes_N101_R02_eps_m00013, predicted_modes_gamma1], 81, ["data", "predicted"], None, None)
plot_localized_modes([modes_N101_R02_eps_m00013, predicted_modes_gamma1], 82, ["data", "predicted"], None, None)

# print(compute_IPR(modes_N101_R02_eps_m00013[:,49]))
# print(compute_IPR(predicted_modes_gamma1[:,49]))



# plot_spectrum_vs_epsilon_winding(GCM_skin_gamma05[2], deriv_skin_gamma05[2], np.linspace(-0.005, 0, 100))
# plot_spectrum_vs_epsilon_winding(GCM_skin_gamma1[2], deriv_skin_gamma1[2], np.linspace(-0.005, 0, 100))
# plot_spectrum_vs_epsilon_winding(GCM_skin_gamma2[2], deriv_skin_gamma2[2], np.linspace(-0.005, 0, 100))

