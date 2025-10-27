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
        # ax[i].plot(x, np.abs(mode[:,loc_index-1])/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
        ax[i].plot(x, mode[:,loc_index-1]/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
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
    # plt.savefig(save_name)


def check_localisation(modes,index):
    mode = np.abs(modes[:,index-1])
    ind = np.argmax(mode)
    print(f'maximum mode à l index:{ind}')
    return(0)






# =========================== other datas =============================
# =========================== Searching for transition eps =============================

# R045
mat_N101_R045_eps0 = read_hdf5_modes('modes_skin_N_101_R045_eps0.mat', 'modes_skin','real') 
mat_N101_R045_eps_m005 = read_hdf5_modes('modes_skin_N_101_R045_eps_m005.mat', 'modes_skin','real')
mat_N101_R045_eps_m0055 = read_hdf5_modes('modes_skin_N_101_R045_eps_m0055.mat', 'modes_skin','real')
mat_N101_R045_eps_m0058 = read_hdf5_modes('modes_skin_N_101_R045_eps_m0058.mat', 'modes_skin','real')
mat_N101_R045_eps_m0059 = read_hdf5_modes('modes_skin_N_101_R045_eps_m0059.mat', 'modes_skin','real')
mat_N101_R045_eps_m006 = read_hdf5_modes('modes_skin_N_101_R045_eps_m006.mat', 'modes_skin','real')


# R04
mat_N101_R04_eps0 = read_hdf5_modes('modes_skin_N_101_R04_eps0.mat', 'modes_skin','real')
mat_N101_R04_eps_m0019 = read_hdf5_modes('modes_skin_N_101_R04_eps_m0019.mat', 'modes_skin','real')
mat_N101_R04_eps_m00195 = read_hdf5_modes('modes_skin_N_101_R04_eps_m00195.mat', 'modes_skin','real')
mat_N101_R04_eps_m00197 = read_hdf5_modes('modes_skin_N_101_R04_eps_m00197.mat', 'modes_skin','real')
mat_N101_R04_eps_m00198 = read_hdf5_modes('modes_skin_N_101_R04_eps_m00198.mat', 'modes_skin','real')
mat_N101_R04_eps_m002 = read_hdf5_modes('modes_skin_N_101_R04_eps_m002.mat', 'modes_skin','real')
mat_N101_R04_eps_m0021 = read_hdf5_modes('modes_skin_N_101_R04_eps_m0021.mat', 'modes_skin','real')


# R035
mat_N101_R035_eps0 = read_hdf5_modes('modes_skin_N_101_R035_eps0.mat', 'modes_skin','real')
mat_N101_R035_eps_m0008 = read_hdf5_modes('modes_skin_N_101_R035_eps_m0008.mat', 'modes_skin','real')
mat_N101_R035_eps_m00085 = read_hdf5_modes('modes_skin_N_101_R035_eps_m00085.mat', 'modes_skin','real')
mat_N101_R035_eps_m000853 = read_hdf5_modes('modes_skin_N_101_R035_eps_m000853.mat', 'modes_skin','real')
mat_N101_R035_eps_m000855 = read_hdf5_modes('modes_skin_N_101_R035_eps_m000855.mat', 'modes_skin','real')
mat_N101_R035_eps_m00086 = read_hdf5_modes('modes_skin_N_101_R035_eps_m00086.mat', 'modes_skin','real')
mat_N101_R035_eps_m00087 = read_hdf5_modes('modes_skin_N_101_R035_eps_m00087.mat', 'modes_skin','real')
mat_N101_R035_eps_m0009 = read_hdf5_modes('modes_skin_N_101_R035_eps_m0009.mat', 'modes_skin','real')
mat_N101_R035_eps_m00095 = read_hdf5_modes('modes_skin_N_101_R035_eps_m00095.mat', 'modes_skin','real')
mat_N101_R035_eps_m001 = read_hdf5_modes('modes_skin_N_101_R035_eps_m001.mat', 'modes_skin','real')


# R03
mat_N101_R03_eps0 = read_hdf5_modes('modes_skin_N_101_R03_eps0.mat', 'modes_skin','real')
mat_N101_R03_eps_m00039 = read_hdf5_modes('modes_skin_N_101_R03_eps_m00039.mat', 'modes_skin','real')
mat_N101_R03_eps_m000393 = read_hdf5_modes('modes_skin_N_101_R03_eps_m000393.mat', 'modes_skin','real')
mat_N101_R03_eps_m000394 = read_hdf5_modes('modes_skin_N_101_R03_eps_m000394.mat', 'modes_skin','real')
mat_N101_R03_eps_m0003945 = read_hdf5_modes('modes_skin_N_101_R03_eps_m0003945.mat', 'modes_skin','real')
mat_N101_R03_eps_m000395 = read_hdf5_modes('modes_skin_N_101_R03_eps_m000395.mat', 'modes_skin','real')
mat_N101_R03_eps_m0004 = read_hdf5_modes('modes_skin_N_101_R03_eps_m0004.mat', 'modes_skin','real')

# R025
mat_N101_R025_eps0 = read_hdf5_modes('modes_skin_N_101_R025_eps0.mat', 'modes_skin','real')
mat_N101_R025_eps_m00015 = read_hdf5_modes('modes_skin_N_101_R025_eps_m00015.mat', 'modes_skin','real')
mat_N101_R025_eps_m00016 = read_hdf5_modes('modes_skin_N_101_R025_eps_m00016.mat', 'modes_skin','real')
mat_N101_R025_eps_m00017 = read_hdf5_modes('modes_skin_N_101_R025_eps_m00017.mat', 'modes_skin','real')
# mat_N101_R025_eps_m000172 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000172.mat', 'modes_skin','real')
mat_N101_R025_eps_m000174 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000174.mat', 'modes_skin','real')
mat_N101_R025_eps_m000176 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000176.mat', 'modes_skin','real')
mat_N101_R025_eps_m000178 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000178.mat', 'modes_skin','real')
mat_N101_R025_eps_m00018 = read_hdf5_modes('modes_skin_N_101_R025_eps_m00018.mat', 'modes_skin','real')
mat_N101_R025_eps_m000182 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000182.mat', 'modes_skin','real')
mat_N101_R025_eps_m000183 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000183.mat', 'modes_skin','real')
mat_N101_R025_eps_m000185 = read_hdf5_modes('modes_skin_N_101_R025_eps_m000185.mat', 'modes_skin','real')
mat_N101_R025_eps_m00019 = read_hdf5_modes('modes_skin_N_101_R025_eps_m00019.mat', 'modes_skin','real')
mat_N101_R025_eps_m0002 = read_hdf5_modes('modes_skin_N_101_R025_eps_m0002.mat', 'modes_skin','real')


# R02
mat_N101_R02_eps0 = read_hdf5_modes('modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps_m00005 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00005.mat', 'modes_skin','real')
mat_N101_R02_eps_m00006 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00006.mat', 'modes_skin','real')
mat_N101_R02_eps_m00007 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00007.mat', 'modes_skin','real')
mat_N101_R02_eps_m000073 = read_hdf5_modes('modes_skin_N_101_R02_eps_m000073.mat', 'modes_skin','real')
mat_N101_R02_eps_m0000735 = read_hdf5_modes('modes_skin_N_101_R02_eps_m0000735.mat', 'modes_skin','real')
mat_N101_R02_eps_m000074 = read_hdf5_modes('modes_skin_N_101_R02_eps_m000074.mat', 'modes_skin','real')
mat_N101_R02_eps_m000075 = read_hdf5_modes('modes_skin_N_101_R02_eps_m000075.mat', 'modes_skin','real')
mat_N101_R02_eps_m00008 = read_hdf5_modes('modes_skin_N_101_R02_eps_m00008.mat', 'modes_skin','real')


# R015
mat_N101_R015_eps0 = read_hdf5_modes('modes_skin_N_101_R015_eps0.mat', 'modes_skin','real')
mat_N101_R015_eps_m00001 = read_hdf5_modes('modes_skin_N_101_R015_eps_m00001.mat', 'modes_skin','real')
mat_N101_R015_eps_m000015 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000015.mat', 'modes_skin','real')
mat_N101_R015_eps_m00002 = read_hdf5_modes('modes_skin_N_101_R015_eps_m00002.mat', 'modes_skin','real')
mat_N101_R015_eps_m000021 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000021.mat', 'modes_skin','real')
mat_N101_R015_eps_m000022 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000022.mat', 'modes_skin','real')
mat_N101_R015_eps_m000024 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000024.mat', 'modes_skin','real')
mat_N101_R015_eps_m0000245 = read_hdf5_modes('modes_skin_N_101_R015_eps_m0000245.mat', 'modes_skin','real')
mat_N101_R015_eps_m000025 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000025.mat', 'modes_skin','real')
mat_N101_R015_eps_m000026 = read_hdf5_modes('modes_skin_N_101_R015_eps_m000026.mat', 'modes_skin','real')


# R01
mat_N101_R01_eps0 = read_hdf5_modes('modes_skin_N_101_R01_eps0.mat', 'modes_skin','real')
mat_N101_R01_eps_m000001 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000001.mat', 'modes_skin','real')
mat_N101_R01_eps_m000003 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000003.mat', 'modes_skin','real')
mat_N101_R01_eps_m000004 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000004.mat', 'modes_skin','real')
mat_N101_R01_eps_m0000045 = read_hdf5_modes('modes_skin_N_101_R01_eps_m0000045.mat', 'modes_skin','real')
mat_N101_R01_eps_m0000047 = read_hdf5_modes('modes_skin_N_101_R01_eps_m0000047.mat', 'modes_skin','real')
mat_N101_R01_eps_m0000048 = read_hdf5_modes('modes_skin_N_101_R01_eps_m0000048.mat', 'modes_skin','real')
mat_N101_R01_eps_m00000485 = read_hdf5_modes('modes_skin_N_101_R01_eps_m00000485.mat', 'modes_skin','real')
mat_N101_R01_eps_m0000049 = read_hdf5_modes('modes_skin_N_101_R01_eps_m0000049.mat', 'modes_skin','real')
mat_N101_R01_eps_m000005 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000005.mat', 'modes_skin','real')
mat_N101_R01_eps_m000007 = read_hdf5_modes('modes_skin_N_101_R01_eps_m000007.mat', 'modes_skin','real')

# labels_R = [r"$R = 0.45$, $\varepsilon = -0.1$", r"$R = 0.45$, $\varepsilon = -0.105$", r"$R = 0.45$, $\varepsilon = -0.11$"]
labels_R = [r"$R = 0.45$, $\varepsilon = -0.058$", r"$R = 0.45$, $\varepsilon = -0.059$", r"$R = 0.45$, $\varepsilon = -0.06$", r"$R = 0.45$, $\varepsilon = -0.06$"]
plot_localized_modes([mat_N101_R01_eps0, mat_N101_R01_eps_m000004, mat_N101_R01_eps_m00000485,mat_N101_R01_eps_m000007], loc_index=101, labels=labels_R, title = r"101th mode, $\varepsilon=\varepsilon_c$", save_name = "localized_mode_last_R04.pdf")

# check_localisation(mat_N101_R04_eps_m00195, 101)
# check_localisation(mat_N101_R04_eps_m00197, 101)
# check_localisation(mat_N101_R04_eps_m00198, 101)



