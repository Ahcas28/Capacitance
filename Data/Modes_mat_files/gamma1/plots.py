import h5py
import numpy as np
import matplotlib.pyplot as plt

import "./gamma1"

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


def read_hdf5_frequencies(fichier_mat, nom_variable, return_type):
    """
    Charge une matrice complexe à partir d'un fichier .mat au format HDF5 (MATLAB v7.3 / Octave -hdf5).

    Paramètres :
        fichier_mat (str) : chemin vers le fichier .mat
        nom_variable (str): nom de la variable complexe à extraire

    Retour :
        np.ndarray : matrice complexe
    """
    with h5py.File(fichier_mat, 'r') as f:
        dataset = f[nom_variable + "/value"]
        data = dataset[:]

        # Accès aux champs 'real' et 'imag'
        real = data['real']
        imag = data['imag']
        complex_array = real + 1j * imag

        if return_type == "real":
            return np.diag(real).T
        elif return_type == "imag":
            return np.diag(imag).T
        elif return_type == "abs":
            return np.diag(np.abs(complex_array.T))
        else:
            return np.diag(complex_array).T        



def explorer_hdf5(fichier):
    with h5py.File(fichier, 'r') as f:
        def explorer(obj, chemin="/"):
            for cle in obj:
                chemin_complet = chemin + cle
                sous_obj = obj[cle]
                if isinstance(sous_obj, h5py.Group):
                    print(f"Groupe : {chemin_complet}/")
                    explorer(sous_obj, chemin_complet + "/")
                elif isinstance(sous_obj, h5py.Dataset):
                    print(f"Dataset: {chemin_complet} | shape: {sous_obj.shape} | dtype: {sous_obj.dtype}")
        
        explorer(f)


def plot_modes(matrice, xlabel="Index spatial", ylabel="Amplitude", title="Modes", save_name=None, max_modes=None):
    N_points, N_modes = matrice.shape

    plt.figure(figsize=(10, 6))
    
    # Limiter le nombre de modes à tracer si max_modes est donné
    if max_modes is not None:
        modes_to_plot = min(max_modes, N_modes)
    else:
        modes_to_plot = N_modes

    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)

    amplitudes_max = np.max(np.abs(matrice[:, :modes_to_plot]), axis=0)
    
    # Normaliser entre 0 et 1 pour la colormap
    norm_amplitudes = (amplitudes_max - amplitudes_max.min()) / (np.ptp(amplitudes_max) + 1e-12)
    
    # Choisir une colormap
    cmap = plt.get_cmap("inferno")

    for i in range(modes_to_plot):
        color = cmap(norm_amplitudes[i])
        plt.plot(x, matrice[:, i], color=None, linewidth = 1., marker = "^", markersize = 2)

    plt.ylim(-0.8, 1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)



def plot_localized_modes(modes, loc_index, labels, title = None, save_name = None):
    N_points, N_modes = modes[0].shape
    nb_list = len(modes)
    print(nb_list)
    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, mode in enumerate(modes):
        ax[i].plot(x, np.abs(mode[:,loc_index-1])/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
        # ax[i].set_ylim(-1.2, 1.2)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("Index")
        ax[i].set_ylabel("Mode value")
        ax[i].legend(loc="best")
        ax[i].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)


def plot_frequencies(data, xlabel="Index spatial", ylabel="Value", title="Frequencies", save_name=None):
    N_points = data.shape[0]
    print(N_points)
    plt.figure(figsize=(10, 6))
    

    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)

    # amplitudes_max = np.max(np.abs(data[:, :modes_to_plot]), axis=0)
    
    # Normaliser entre 0 et 1 pour la colormap
    # norm_amplitudes = (amplitudes_max - amplitudes_max.min()) / (np.ptp(amplitudes_max) + 1e-12)
    
    # Choisir une colormap
    cmap = plt.get_cmap("inferno")
    plt.plot(x, data, color=plt.get_cmap("inferno")(0.3), linestyle ='None', marker = "^", markersize = 4)
    
    # for i in range(modes_to_plot):
    #     # color = cmap(norm_amplitudes[i])
    #     plt.plot(x, matrice[:, i], color=None, linewidth = 1., marker = "^", markersize = 2)
    plt.xscale('linear')
    plt.yscale('log')
    # plt.ylim(-0.8, 1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)


def plot_multi_frequencies(datas, labels, xlabel="Index spatial", ylabel="Value", title=None, save_name=None):

    N_points = datas[0].shape[0]
    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)
    
    nb_list = len(datas)
    print(nb_list)
    colors = plt.get_cmap("inferno")(np.linspace(0.3,0.7,nb_list))
    markers = ["s", "x", "*"]

    plt.figure(figsize=(10, 6))
    for i, data in enumerate(datas):
        sorted_data = np.sort(data)
        plt.plot(x, sorted_data, color=colors[i], label = labels[i], linestyle ='-.', linewidth = 1, marker = markers[i], markersize = 4)

    plt.xscale('linear')
    plt.yscale('log')
    # plt.ylim(-0.8, 1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, which='both', linestyle='--', alpha = 0.7)
    plt.tight_layout()
    plt.savefig(save_name)


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
    plt.yscale('log')
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
    plt.savefig(save_name)

def effective_ampl(n, eps, gamma, alpha):
    eta = eps * n
    f_eta = np.exp(-alpha * np.abs(eta) / (2 * np.cosh(gamma)))
    if n <= 0:
        val = f_eta
    else:
        val = f_eta * np.exp(-2 * n * gamma)
    return(val)



def plot_2scale_localized_modes(modes, loc_index, labels, eps_list, alpha_list, gamma_list, title = None, save_name = None):
    N_points, N_modes = modes[0].shape
    nb_list = len(modes)
    print(nb_list)
    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, mode in enumerate(modes):
        # ampl = np.array([effective_ampl(x_i - 50, eps_list[i], gamma=1.0, alpha=alpha_list[i]) * np.abs(mode[:,loc_index-1][x_i])/np.linalg.norm(mode[:,loc_index-1], np.inf)  for x_i in x])
        ampl = np.array([effective_ampl(x_i - 50, eps_list[i], gamma=gamma_list[i], alpha=alpha_list[i]) for x_i in x])
        ax[i].plot(x, np.abs(mode[:,loc_index-1])/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-', linewidth = 1, marker = markers[i%5], markersize = 4)
        ax[i].plot(x, np.abs(ampl), 'k', label = r'$f((\varepsilon - \varepsilon_c)n)$', linestyle = '--', linewidth = 1)
        # ax[i].plot(x, -ampl, 'k', linestyle = '--', linewidth = 1)
        # ax[i].set_ylim(-1.5, 1.5)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("Index")
        ax[i].set_ylabel("Mode value")
        ax[i].legend(loc="best")
        ax[i].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)





# n_list = np.arange(-50, 50, 1)
# ampl = [effective_ampl(n, 0.1, 1, 1) for n in n_list]
# plt.plot(n_list+50, ampl)
# plt.show()

# Exécuter l'exploration
# explorer_hdf5("eval_skin_N_25_R02_eps_m005.mat")


# =================== Load datas =====================
#
# ====================================================

# ==================== eps = -0.1 ======================
# # frequencies complex
# eig_R02_eps_m01_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m01.mat', 'eval_skin', 'none')
# eig_R03_eps_m01_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m01.mat', 'eval_skin', 'none')
# eig_R04_eps_m01_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m01.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_m01_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m01.mat', 'eval_skin', 'real')
# eig_R03_eps_m01_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m01.mat', 'eval_skin', 'real')
# eig_R04_eps_m01_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m01.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_m01 = read_hdf5_modes('modes_skin_N_25_R02_eps_m01.mat', 'modes_skin','real')
mat_R03_eps_m01 = read_hdf5_modes('modes_skin_N_25_R03_eps_m01.mat', 'modes_skin','real')
mat_R04_eps_m01 = read_hdf5_modes('modes_skin_N_25_R04_eps_m01.mat', 'modes_skin','real')

# ==================== eps = -0.05 ======================
# # frequencies complex
# eig_R02_eps_m005_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m005.mat', 'eval_skin', 'none')
# eig_R03_eps_m005_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m005.mat', 'eval_skin', 'none')
# eig_R04_eps_m005_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m005.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_m005_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m005.mat', 'eval_skin', 'real')
# eig_R03_eps_m005_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m005.mat', 'eval_skin', 'real')
# eig_R04_eps_m005_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m005.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_m005 = read_hdf5_modes('modes_skin_N_25_R02_eps_m005.mat', 'modes_skin','real')
mat_R03_eps_m005 = read_hdf5_modes('modes_skin_N_25_R03_eps_m005.mat', 'modes_skin','real')
mat_R04_eps_m005 = read_hdf5_modes('modes_skin_N_25_R04_eps_m005.mat', 'modes_skin','real')

# ==================== eps = -0.04 ======================
# # frequencies complex
# eig_R02_eps_m004_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m004.mat', 'eval_skin', 'none')
# eig_R03_eps_m004_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m004.mat', 'eval_skin', 'none')
# eig_R04_eps_m004_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m004.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_m004_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m004.mat', 'eval_skin', 'real')
# eig_R03_eps_m004_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m004.mat', 'eval_skin', 'real')
# eig_R04_eps_m004_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m004.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_m004 = read_hdf5_modes('modes_skin_N_25_R02_eps_m004.mat', 'modes_skin','real')
mat_R03_eps_m004 = read_hdf5_modes('modes_skin_N_25_R03_eps_m004.mat', 'modes_skin','real')
mat_R04_eps_m004 = read_hdf5_modes('modes_skin_N_25_R04_eps_m004.mat', 'modes_skin','real')


# ==================== eps = -0.03 ======================
# # frequencies complex
# eig_R02_eps_m003_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m003.mat', 'eval_skin', 'none')
# eig_R03_eps_m003_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m003.mat', 'eval_skin', 'none')
# eig_R04_eps_m003_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m003.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_m003_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m003.mat', 'eval_skin', 'real')
# eig_R03_eps_m003_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m003.mat', 'eval_skin', 'real')
# eig_R04_eps_m003_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m003.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_m003 = read_hdf5_modes('modes_skin_N_25_R02_eps_m003.mat', 'modes_skin','real')
mat_R03_eps_m003 = read_hdf5_modes('modes_skin_N_25_R03_eps_m003.mat', 'modes_skin','real')
mat_R04_eps_m003 = read_hdf5_modes('modes_skin_N_25_R04_eps_m003.mat', 'modes_skin','real')


# ==================== eps = -0.02 ======================
# # frequencies complex
# eig_R02_eps_m002_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m002.mat', 'eval_skin', 'none')
# eig_R03_eps_m002_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m002.mat', 'eval_skin', 'none')
# eig_R04_eps_m002_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m002.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_m002_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m002.mat', 'eval_skin', 'real')
# eig_R03_eps_m002_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m002.mat', 'eval_skin', 'real')
# eig_R04_eps_m002_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m002.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_m002 = read_hdf5_modes('modes_skin_N_25_R02_eps_m002.mat', 'modes_skin','real')
mat_R03_eps_m002 = read_hdf5_modes('modes_skin_N_25_R03_eps_m002.mat', 'modes_skin','real')
mat_R04_eps_m002 = read_hdf5_modes('modes_skin_N_25_R04_eps_m002.mat', 'modes_skin','real')


# ==================== eps = -0.01 ======================
# # frequencies complex
# eig_R02_eps_m001_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m001.mat', 'eval_skin', 'none')
# eig_R03_eps_m001_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m001.mat', 'eval_skin', 'none')
# eig_R04_eps_m001_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m001.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_m001_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_m001.mat', 'eval_skin', 'real')
# eig_R03_eps_m001_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_m001.mat', 'eval_skin', 'real')
# eig_R04_eps_m001_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_m001.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_m001 = read_hdf5_modes('modes_skin_N_25_R02_eps_m001.mat', 'modes_skin','real')
mat_R03_eps_m001 = read_hdf5_modes('modes_skin_N_25_R03_eps_m001.mat', 'modes_skin','real')
mat_R04_eps_m001 = read_hdf5_modes('modes_skin_N_25_R04_eps_m001.mat', 'modes_skin','real')

# ==================== eps = 0.0 ======================
# # frequencies complex
# eig_R02_eps_00_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_00.mat', 'eval_skin', 'none')
# eig_R03_eps_00_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_00.mat', 'eval_skin', 'none')
# eig_R04_eps_00_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_00.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_00_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_00.mat', 'eval_skin', 'real')
# eig_R03_eps_00_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_00.mat', 'eval_skin', 'real')
# eig_R04_eps_00_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_00.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_00 = read_hdf5_modes('modes_skin_N_25_R02_eps_00.mat', 'modes_skin','real')
mat_R03_eps_00 = read_hdf5_modes('modes_skin_N_25_R03_eps_00.mat', 'modes_skin','real')
mat_R04_eps_00 = read_hdf5_modes('modes_skin_N_25_R04_eps_00.mat', 'modes_skin','real')


# ==================== eps = +0.01 ======================
# # frequencies complex
# eig_R02_eps_p001_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_p001.mat', 'eval_skin', 'none')
# eig_R03_eps_p001_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_p001.mat', 'eval_skin', 'none')
# eig_R04_eps_p001_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_p001.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_p001_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_p001.mat', 'eval_skin', 'real')
# eig_R03_eps_p001_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_p001.mat', 'eval_skin', 'real')
# eig_R04_eps_p001_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_p001.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_p001 = read_hdf5_modes('modes_skin_N_25_R02_eps_p001.mat', 'modes_skin','real')
mat_R03_eps_p001 = read_hdf5_modes('modes_skin_N_25_R03_eps_p001.mat', 'modes_skin','real')
mat_R04_eps_p001 = read_hdf5_modes('modes_skin_N_25_R04_eps_p001.mat', 'modes_skin','real')

# ==================== eps = +0.05 ======================
# # frequencies complex
# eig_R02_eps_p005_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_p005.mat', 'eval_skin', 'none')
# eig_R03_eps_p005_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_p005.mat', 'eval_skin', 'none')
# eig_R04_eps_p005_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_p005.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_p005_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_p005.mat', 'eval_skin', 'real')
# eig_R03_eps_p005_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_p005.mat', 'eval_skin', 'real')
# eig_R04_eps_p005_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_p005.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_p005 = read_hdf5_modes('modes_skin_N_25_R02_eps_p005.mat', 'modes_skin','real')
mat_R03_eps_p005 = read_hdf5_modes('modes_skin_N_25_R03_eps_p005.mat', 'modes_skin','real')
mat_R04_eps_p005 = read_hdf5_modes('modes_skin_N_25_R04_eps_p005.mat', 'modes_skin','real')

# ==================== eps = +0.1 ======================
# # frequencies complex
# eig_R02_eps_p01_c = read_hdf5_frequencies('eval_skin_N_25_R02_eps_p01.mat', 'eval_skin', 'none')
# eig_R03_eps_p01_c = read_hdf5_frequencies('eval_skin_N_25_R03_eps_p01.mat', 'eval_skin', 'none')
# eig_R04_eps_p01_c = read_hdf5_frequencies('eval_skin_N_25_R04_eps_p01.mat', 'eval_skin', 'none')
# # frequencies real
# eig_R02_eps_p01_r = read_hdf5_frequencies('eval_skin_N_25_R02_eps_p01.mat', 'eval_skin', 'real')
# eig_R03_eps_p01_r = read_hdf5_frequencies('eval_skin_N_25_R03_eps_p01.mat', 'eval_skin', 'real')
# eig_R04_eps_p01_r = read_hdf5_frequencies('eval_skin_N_25_R04_eps_p01.mat', 'eval_skin', 'real')

# modes
mat_R02_eps_p01 = read_hdf5_modes('modes_skin_N_25_R02_eps_p01.mat', 'modes_skin','real')
mat_R03_eps_p01 = read_hdf5_modes('modes_skin_N_25_R03_eps_p01.mat', 'modes_skin','real')
mat_R04_eps_p01 = read_hdf5_modes('modes_skin_N_25_R04_eps_p01.mat', 'modes_skin','real')

# =========================== other datas =============================
# =========================== Searching for transition eps =============================
# R01
mat_N101_R01_eps_m000007 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000007.mat', 'modes_skin','real')
mat_N101_R01_eps_m000008 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000008.mat', 'modes_skin','real')
mat_N101_R01_eps_m00001 = read_hdf5_modes('modes_skin_N_101_R01_eps_m00001.mat', 'modes_skin','real')


# R015
mat_N101_R015_eps_m000035 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000035.mat', 'modes_skin','real')
mat_N101_R015_eps_m00004 = read_hdf5_modes('modes_skin_N_101_R015_eps_m00004.mat', 'modes_skin','real')
mat_N101_R015_eps_m000045 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000045.mat', 'modes_skin','real')


# R02
mat_N101_R02_eps_m0001 = read_hdf5_modes('modes_skin_N_101_R02_eps_m0001.mat', 'modes_skin','real')
mat_N101_R02_eps_m00012 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00012.mat', 'modes_skin','real')
mat_N101_R02_eps_m00013 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00013.mat', 'modes_skin','real')
mat_N101_R02_eps_m00015 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00015.mat', 'modes_skin','real')
mat_N101_R02_eps_m0002 = read_hdf5_modes('modes_skin_N_101_R02_eps_m0002.mat', 'modes_skin','real')
mat_N101_R02_eps_m0003 = read_hdf5_modes('modes_skin_N_101_R02_eps_m0003.mat', 'modes_skin','real')
mat_N101_R02_eps_m00032 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00032.mat', 'modes_skin','real')
mat_N101_R02_eps_m0004 = read_hdf5_modes('modes_skin_N_101_R02_eps_m0004.mat', 'modes_skin','real')


# R025
mat_N101_R025_eps_m0001 = read_hdf5_modes('modes_skin_N_101_R025_eps_m0001.mat', 'modes_skin','real')
mat_N101_R025_eps_m0002 = read_hdf5_modes('modes_skin_N_101_R025_eps_m0002.mat', 'modes_skin','real')
mat_N101_R025_eps_m0003 = read_hdf5_modes('modes_skin_N_101_R025_eps_m0003.mat', 'modes_skin','real')
mat_N101_R025_eps_m00035 = read_hdf5_modes('modes_skin_N_101_R025_eps_m00035.mat', 'modes_skin','real')
mat_N101_R025_eps_m0004 = read_hdf5_modes('modes_skin_N_101_R025_eps_m0004.mat', 'modes_skin','real')
mat_N101_R025_eps_m0005 = read_hdf5_modes('modes_skin_N_101_R025_eps_m0005.mat', 'modes_skin','real')
mat_N101_R025_eps_m001 = read_hdf5_modes('modes_skin_N_101_R025_eps_m001.mat', 'modes_skin','real')
mat_N101_R025_eps_m002 = read_hdf5_modes('modes_skin_N_101_R025_eps_m002.mat', 'modes_skin','real')
mat_N101_R025_eps_m003 = read_hdf5_modes('modes_skin_N_101_R025_eps_m003.mat', 'modes_skin','real')
mat_N101_R025_eps_m004 = read_hdf5_modes('modes_skin_N_101_R025_eps_m004.mat', 'modes_skin','real')
mat_N25_R025_eps_m0003 = read_hdf5_modes('modes_skin_N_25_R025_eps_m0003.mat', 'modes_skin','real')
mat_N25_R025_eps_m00035 = read_hdf5_modes('modes_skin_N_25_R025_eps_m00035.mat', 'modes_skin','real')
mat_N25_R025_eps_m0004 = read_hdf5_modes('modes_skin_N_25_R025_eps_m0004.mat', 'modes_skin','real')


# R03
mat_N101_R03_eps_m0005 = read_hdf5_modes('modes_skin_N_101_R03_eps_m0005.mat', 'modes_skin','real')
mat_N101_R03_eps_m0006 = read_hdf5_modes('modes_skin_N_101_R03_eps_m0006.mat', 'modes_skin','real')
mat_N101_R03_eps_m0007 = read_hdf5_modes('modes_skin_N_101_R03_eps_m0007.mat', 'modes_skin','real')
mat_N101_R03_eps_m0008 = read_hdf5_modes('modes_skin_N_101_R03_eps_m0008.mat', 'modes_skin','real')

# R035
mat_N101_R035_eps_m001 = read_hdf5_modes('modes_skin_N_101_R035_eps_m001.mat', 'modes_skin','real')
mat_N101_R035_eps_m0015 = read_hdf5_modes('modes_skin_N_101_R035_eps_m0015.mat', 'modes_skin','real')
mat_N101_R035_eps_m0016 = read_hdf5_modes('modes_skin_N_101_R035_eps_m0016.mat', 'modes_skin','real')
mat_N101_R035_eps_m0017 = read_hdf5_modes('modes_skin_N_101_R035_eps_m0017.mat', 'modes_skin','real')
mat_N101_R035_eps_m002 = read_hdf5_modes('modes_skin_N_101_R035_eps_m002.mat', 'modes_skin','real')


# R04
mat_N101_R04_eps_m003 = read_hdf5_modes('modes_skin_N_101_R04_eps_m003.mat', 'modes_skin','real')
mat_N101_R04_eps_m0035 = read_hdf5_modes('modes_skin_N_101_R04_eps_m0035.mat', 'modes_skin','real')
mat_N101_R04_eps_m004 = read_hdf5_modes('modes_skin_N_101_R04_eps_m004.mat', 'modes_skin','real')
mat_N101_R04_eps_m005 = read_hdf5_modes('modes_skin_N_101_R04_eps_m005.mat', 'modes_skin','real')

# R045
mat_N101_R045_eps_m006 = read_hdf5_modes('modes_skin_N_101_R045_eps_m006.mat', 'modes_skin','real')
mat_N101_R045_eps_m007 = read_hdf5_modes('modes_skin_N_101_R045_eps_m007.mat', 'modes_skin','real')
mat_N101_R045_eps_m0075 = read_hdf5_modes('modes_skin_N_101_R045_eps_m0075.mat', 'modes_skin','real')
mat_N101_R045_eps_m0078 = read_hdf5_modes('modes_skin_N_101_R045_eps_m0078.mat', 'modes_skin','real')
mat_N101_R045_eps_m008 = read_hdf5_modes('modes_skin_N_101_R045_eps_m008.mat', 'modes_skin','real')



# labels_R = [r"$R = 0.1$, $\varepsilon_c = -0.00008$", r"$R = 0.15$, $\varepsilon_c = -0.00035$", r"$R = 0.2$, $\varepsilon_c = -0.0013$", r"$R = 0.25$, $\varepsilon_c = -0.0035$"]
# plot_localized_modes([mat_N101_R01_eps_m000008, mat_N101_R015_eps_m000035, mat_N101_R02_eps_m00013, mat_N101_R025_eps_m00035], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")

# labels_R = [r"$R = 0.3$, $\varepsilon_c = -0.008$", r"$R = 0.35$, $\varepsilon_c = -0.016$", r"$R = 0.4$, $\varepsilon_c = -0.035$", r"$R = 0.45$, $\varepsilon_c = -0.078$"]
# plot_localized_modes([mat_N101_R03_eps_m0008, mat_N101_R035_eps_m0016, mat_N101_R04_eps_m0035, mat_N101_R045_eps_m0078], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")

plot_2scale_localized_modes([mat_N101_R01_eps_m000008, mat_N101_R015_eps_m000035], loc_index=101, labels=[r"$R=0.1$", r"$R=0.15$"], eps_list=[0,0], alpha_list=[-1,-1], gamma_list=[0.1,0.1], title=r'localized mode, $d = \varepsilon_c$', save_name=None)
plot_2scale_localized_modes([mat_N101_R02_eps_m00013, mat_N101_R025_eps_m00035], loc_index=101, labels=[r"$R=0.2$", r"$R=0.25$"], eps_list=[0,0], alpha_list=[-1,-1], gamma_list=[1,1], title=r'localized mode, $d = \varepsilon_c$', save_name=None)
plot_2scale_localized_modes([mat_N101_R03_eps_m0008, mat_N101_R035_eps_m0016], loc_index=101, labels=[r"$R=0.3$", r"$R=0.35$"], eps_list=[0,0], alpha_list=[-1,-1], gamma_list=[1,1], title=r'localized mode, $d = \varepsilon_c$', save_name=None)
plot_2scale_localized_modes([mat_N101_R04_eps_m0035, mat_N101_R045_eps_m0078], loc_index=101, labels=[r"$R=0.4$", r"$R=0.45$"], eps_list=[0,0], alpha_list=[-1,-1], gamma_list=[1,1], title=r'localized mode, $d = \varepsilon_c$', save_name=None)


labels_R = [r"$R = 0.1$, $\varepsilon_c = -0.00008$", r"$R = 0.15$, $\varepsilon_c = -0.00035$"]
plot_localized_modes([mat_N101_R01_eps_m000008, mat_N101_R015_eps_m000035], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")
labels_R = [r"$R = 0.2$, $\varepsilon_c = -0.0013$", r"$R = 0.25$, $\varepsilon_c = -0.0035$"]
plot_localized_modes([mat_N101_R02_eps_m00013, mat_N101_R025_eps_m00035], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")
labels_R = [r"$R = 0.3$, $\varepsilon_c = -0.008$", r"$R = 0.35$, $\varepsilon_c = -0.016$"]
plot_localized_modes([mat_N101_R03_eps_m0008, mat_N101_R035_eps_m0016], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")
labels_R = [r"$R = 0.4$, $\varepsilon_c = -0.035$", r"$R = 0.45$, $\varepsilon_c = -0.078$"]
plot_localized_modes([mat_N101_R04_eps_m0035, mat_N101_R045_eps_m0078], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")





# ======================= plot datas ========================= 
# ==================== eps = -0.1 ======================
# plot modes
# plot_modes(mat_R02_eps_m01, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = -0.1$", save_name="modes_R02_eps_m01.pdf", max_modes=25)
# plot_modes(mat_R03_eps_m01, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = -0.1$", save_name="modes_R03_eps_m01.pdf", max_modes=25)
# plot_modes(mat_R04_eps_m01, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = -0.1$", save_name="modes_R04_eps_m01.pdf", max_modes=25)
# plot frequencies
# plot_frequencies(eig_R02_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $R = 0.2$, $\varepsilon = -0.1$", save_name=None)
# plot_frequencies(eig_R03_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $R = 0.3$, $\varepsilon = -0.1$", save_name=None)
# plot_frequencies(eig_R04_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $R = 0.4$, $\varepsilon = -0.1$", save_name=None)
# eigs_eps_m01_r = [eig_R02_eps_m01_r, eig_R03_eps_m01_r, eig_R04_eps_m01_r]
# eigs_eps_m01_c = [eig_R02_eps_m01_c, eig_R03_eps_m01_c, eig_R04_eps_m01_c]
# labels_eigs_eps_m01 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_m01_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = -0.1$", save_name="frequencies_real_N25_eps_m01.pdf")
# plot_multi_complex_frequencies(eigs_eps_m01_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = -0.1$", save_name="frequencies_complex_N25_eps_m01.pdf")

# ==================== eps = -0.05 ======================
# plot modes
# plot_modes(mat_R02_eps_m005, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = -0.05$", save_name="modes_R02_eps_m005.pdf", max_modes=25)
# plot_modes(mat_R03_eps_m005, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = -0.05$", save_name="modes_R03_eps_m005.pdf", max_modes=25)
# plot_modes(mat_R04_eps_m005, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = -0.05$", save_name="modes_R04_eps_m005.pdf", max_modes=25)
# plot frequencies
# eigs_eps_m005_r = [eig_R02_eps_m005_r, eig_R03_eps_m005_r, eig_R04_eps_m005_r]
# eigs_eps_m005_c = [eig_R02_eps_m005_c, eig_R03_eps_m005_c, eig_R04_eps_m005_c]
# labels_eigs_eps_m01 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_m005_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = -0.05$", save_name="frequencies_real_N25_eps_m005.pdf")
# plot_multi_complex_frequencies(eigs_eps_m005_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = -0.05$", save_name="frequencies_complex_N25_eps_m005.pdf")

# ==================== eps = -0.01 ======================
# plot modes
# plot_modes(mat_R02_eps_m001, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = -0.01$", save_name="modes_R02_eps_m001.pdf", max_modes=25)
# plot_modes(mat_R03_eps_m001, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = -0.01$", save_name="modes_R03_eps_m001.pdf", max_modes=25)
# plot_modes(mat_R04_eps_m001, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = -0.01$", save_name="modes_R04_eps_m001.pdf", max_modes=25)
# plot frequencies
# eigs_eps_m001_r = [eig_R02_eps_m001_r, eig_R03_eps_m001_r, eig_R04_eps_m001_r]
# eigs_eps_m001_c = [eig_R02_eps_m001_c, eig_R03_eps_m001_c, eig_R04_eps_m001_c]
# labels_eigs_eps_m01 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_m001_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = -0.01$", save_name="frequencies_real_N25_eps_m001.pdf")
# plot_multi_complex_frequencies(eigs_eps_m001_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = -0.01$", save_name="frequencies_complex_N25_eps_m001.pdf")

# ==================== eps = 0.0 ======================
# plot modes
# plot_modes(mat_R02_eps_00, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = 0.0$", save_name="modes_R02_eps_00.pdf", max_modes=25)
# plot_modes(mat_R03_eps_00, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = 0.0$", save_name="modes_R03_eps_00.pdf", max_modes=25)
# plot_modes(mat_R04_eps_00, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = 0.0$", save_name="modes_R04_eps_00.pdf", max_modes=25)
# plot frequencies
# eigs_eps_00_r = [eig_R02_eps_00_r, eig_R03_eps_00_r, eig_R04_eps_00_r]
# eigs_eps_00_c = [eig_R02_eps_00_c, eig_R03_eps_00_c, eig_R04_eps_00_c]
# labels_eigs_eps_00 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_00_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = 0.0$", save_name="frequencies_real_N25_eps_00.pdf")
# plot_multi_complex_frequencies(eigs_eps_00_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = 0.0$", save_name="frequencies_complex_N25_eps_00.pdf")

# ==================== eps = +0.01 ======================
# plot modes
# plot_modes(mat_R02_eps_p001, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = +0.01$", save_name="modes_R02_eps_p001.pdf", max_modes=25)
# plot_modes(mat_R03_eps_p001, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = +0.01$", save_name="modes_R03_eps_p001.pdf", max_modes=25)
# plot_modes(mat_R04_eps_p001, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = +0.01$", save_name="modes_R04_eps_p001.pdf", max_modes=25)
# plot frequencies
# eigs_eps_p001_r = [eig_R02_eps_p001_r, eig_R03_eps_p001_r, eig_R04_eps_p001_r]
# eigs_eps_p001_c = [eig_R02_eps_p001_c, eig_R03_eps_p001_c, eig_R04_eps_p001_c]
# labels_eigs_eps_m01 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_p001_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = +0.01$", save_name="frequencies_real_N25_eps_p001.pdf")
# plot_multi_complex_frequencies(eigs_eps_p001_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = +0.01$", save_name="frequencies_complex_N25_eps_p001.pdf")

# ==================== eps = +0.05 ======================
# plot modes
# plot_modes(mat_R02_eps_p005, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = +0.05$", save_name="modes_R02_eps_p005.pdf", max_modes=25)
# plot_modes(mat_R03_eps_p005, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = +0.05$", save_name="modes_R03_eps_p005.pdf", max_modes=25)
# plot_modes(mat_R04_eps_p005, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = +0.05$", save_name="modes_R04_eps_p005.pdf", max_modes=25)
# plot frequencies
# eigs_eps_p005_r = [eig_R02_eps_p005_r, eig_R03_eps_p005_r, eig_R04_eps_p005_r]
# eigs_eps_p005_c = [eig_R02_eps_p005_c, eig_R03_eps_p005_c, eig_R04_eps_p005_c]
# labels_eigs_eps_m01 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_p005_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = +0.05$", save_name="frequencies_real_N25_eps_p005.pdf")
# plot_multi_complex_frequencies(eigs_eps_p005_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = +0.05$", save_name="frequencies_complex_N25_eps_p005.pdf")

# ==================== eps = +0.1 ======================
# plot modes
# plot_modes(mat_R02_eps_p01, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.2$, $\varepsilon = +0.1$", save_name="modes_R02_eps_p01.pdf", max_modes=25)
# plot_modes(mat_R03_eps_p01, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.3$, $\varepsilon = +0.1$", save_name="modes_R03_eps_p01.pdf", max_modes=25)
# plot_modes(mat_R04_eps_p01, xlabel="Position", ylabel="Valeur du mode", title=r"Modes $R = 0.4$, $\varepsilon = +0.1$", save_name="modes_R04_eps_p01.pdf", max_modes=25)
# plot frequencies
# eigs_eps_p01_r = [eig_R02_eps_p01_r, eig_R03_eps_p01_r, eig_R04_eps_p01_r]
# eigs_eps_p01_c = [eig_R02_eps_p01_c, eig_R03_eps_p01_c, eig_R04_eps_p01_c]
# labels_eigs_eps_m01 = [r"$R=0.2$", r"$R=0.3$", r"$R=0.4$"]
# plot_multi_frequencies(eigs_eps_p01_r, labels=labels_eigs_eps_m01, xlabel="Index", ylabel="Value", title=r"Real Frequencies $N=25$, $\varepsilon = +0.1$", save_name="frequencies_real_N25_eps_p01.pdf")
# plot_multi_complex_frequencies(eigs_eps_p01_c, labels=labels_eigs_eps_m01, title=r"Complex Frequencies $N=25$, $\varepsilon = +0.1$", save_name="frequencies_complex_N25_eps_p01.pdf")



#==================== plot localized modes ===================

#==================== R = 0.2 ======================
# modes_R02 = [mat_R02_eps_m01, mat_R02_eps_m005, mat_R02_eps_m001 ,mat_R02_eps_00, mat_R02_eps_p001, mat_R02_eps_p005, mat_R02_eps_p01]
# labels_R = [r"$\varepsilon = -0.1$", r"$\varepsilon = -0.05$", r"$\varepsilon = -0.01$", r"$\varepsilon = 0.0$", r"$\varepsilon = +0.01$", r"$\varepsilon = +0.05$", r"$\varepsilon = +0.1$"]
modes_R02 = [mat_R02_eps_m01, mat_R02_eps_m005, mat_R02_eps_m004, mat_R02_eps_m003, mat_R02_eps_m002, mat_R02_eps_m001 ,mat_R02_eps_00]
labels_R = [r"$\varepsilon = -0.1$", r"$\varepsilon = -0.05$", r"$\varepsilon = -0.04$", r"$\varepsilon = -0.03$", r"$\varepsilon = -0.02$", r"$\varepsilon = -0.01$", r"$\varepsilon = 0.0$"]
# plot_localized_modes(modes_R02, loc_index=25, labels=labels_R, title = r"25th mode, $R=0.2$", save_name = "localized_mode_last_R02.pdf")
# plot_localized_modes(modes_R02, loc_index=1, labels=labels_R, title = r"1st mode, $R=0.2$", save_name = "localized_mode_first_R02.pdf")

#==================== R = 0.3 ======================
# modes_R03 = [mat_R03_eps_m01, mat_R03_eps_m005, mat_R03_eps_m001 ,mat_R03_eps_00, mat_R03_eps_p001, mat_R03_eps_p005, mat_R03_eps_p01]
# labels_R = [r"$\varepsilon = -0.1$", r"$\varepsilon = -0.05$", r"$\varepsilon = -0.01$", r"$\varepsilon = 0.0$", r"$\varepsilon = +0.01$", r"$\varepsilon = +0.05$", r"$\varepsilon = +0.1$"]
modes_R03 = [mat_R03_eps_m01, mat_R03_eps_m005, mat_R03_eps_m004, mat_R03_eps_m003, mat_R03_eps_m002, mat_R03_eps_m001 ,mat_R03_eps_00]
labels_R = [r"$\varepsilon = -0.1$", r"$\varepsilon = -0.05$", r"$\varepsilon = -0.04$", r"$\varepsilon = -0.03$", r"$\varepsilon = -0.02$", r"$\varepsilon = -0.01$", r"$\varepsilon = 0.0$"]
# plot_localized_modes(modes_R03, loc_index=25, labels=labels_R, title = r"25th mode, $R=0.3$", save_name = "localized_mode_last_R03.pdf")
# plot_localized_modes(modes_R03, loc_index=1, labels=labels_R, title = r"1st mode, $R=0.3$", save_name = "localized_mode_first_R03.pdf")

#==================== R = 0.4 ======================
# modes_R04 = [mat_R04_eps_m01, mat_R04_eps_m005, mat_R04_eps_m001 ,mat_R04_eps_00, mat_R04_eps_p001, mat_R04_eps_p005, mat_R04_eps_p01]
modes_R04 = [mat_R04_eps_m01, mat_R04_eps_m005, mat_R04_eps_m004, mat_R04_eps_m003, mat_R04_eps_m002, mat_R04_eps_m001 ,mat_R04_eps_00]
# modes_R04 = [mat_R03_eps_m001]
# labels_R = [r"$\varepsilon = -0.1$", r"$\varepsilon = -0.05$", r"$\varepsilon = -0.01$", r"$\varepsilon = 0.0$", r"$\varepsilon = +0.01$", r"$\varepsilon = +0.05$", r"$\varepsilon = +0.1$"]
# labels_R = [r"$\varepsilon = -0.1$", r"$\varepsilon = -0.05$", r"$\varepsilon = -0.04$", r"$\varepsilon = -0.03$", r"$\varepsilon = -0.02$", r"$\varepsilon = -0.01$", r"$\varepsilon = 0.0$"]
labels_R = [r"$\varepsilon = -0.01$", r"$\varepsilon = -0.02$", r"$\varepsilon = -0.03$"]
# plot_localized_modes(modes_R04, loc_index=25, labels=labels_R, title = r"25th mode, $R=0.4$", save_name = "localized_mode_last_R04.pdf")
# plot_localized_modes([mat_R03_eps_m001, mat_R03_eps_m002, mat_R03_eps_m003], loc_index=25, labels=labels_R, title = r"25th mode, $R=0.3$", save_name = "localized_mode_last_R04.pdf")
