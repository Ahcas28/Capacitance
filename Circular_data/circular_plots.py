import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import glob
from scipy.linalg import eig
import matplotlib
matplotlib.rcParams['text.usetex'] = True


def extract_params(filename):
    # Expression régulière mise à jour : accepte 'eps' ou 'epsm'
    match = re.search(r'N(\d+)_R(\d+)_r(\d+)_eps(m?\d+)', filename)
    if match:
        N = int(match.group(1))
        R = int(match.group(2))

        # def to_decimal(val_str):
        #     # Valeur négative si commence par 'm'
        #     if val_str.startswith('m'):
        #         sign = -1
        #         digits = val_str[1:]
        #     else:
        #         sign = 1
        #         digits = val_str
        #     # digits = digits.lstrip("0") or "0"
        #     return sign * float("0." + digits)

        def to_decimal(val_str):
            sign = -1 if val_str.startswith('m') else 1
            digits = val_str[1:] if sign == -1 else val_str

            # Cas particulier : tout est zéro
            if set(digits) == {"0"}:
                return 0.0

            # Règle générale
            value = int(digits) / (10 ** (len(digits) - 1))
            return sign * value  
          
        r = to_decimal(match.group(3))
        eps = to_decimal(match.group(4))

        return [N, R, r, eps]
    else:
        raise ValueError("Le nom de fichier ne correspond pas au format attendu.")


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
        

def plot_complex_eigenfrequencies(eigs,params):
    size = len(eigs)
    plt.figure(figsize=(10, 6))
    x = eigs.real
    y = eigs.imag
    plt.scatter(x, y, s=8, marker="^")
    for j, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(j), fontsize=8, ha='left', va='bottom')     

    plt.xscale('linear')
    plt.yscale('linear')
    # plt.xscale('symlog', linthresh=1e-15)    
    # plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel("Real part")
    plt.ylabel("Imaginary Part")
    plt.title(rf'Eigenfrequencies in complex plane, $N={size}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axis('equal')  # Pour que le cercle unité soit rond
    plt.tight_layout()
    plt.show()


def plot_real_eigenfrequencies(eigs,params):

    size = len(eigs)
    plt.figure(figsize=(10, 6))
    x = np.arange(1,size+1)
    y = eigs.real
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,size))

    for i in range(0,size):
        if (i == size-1):
            plt.scatter(x[i], y[i], color = 'k', marker="*", s=8)
        else:
            plt.scatter(x[i], y[i], color = colors[i], marker="*", s=8)
       
    plt.xscale('linear')
    plt.yscale('linear')
    # plt.xscale('symlog', linthresh=1e-15)    
    # plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"Eigenvalue $\lambda_k$")
    # plt.title(rf'Circular Eigenfrequencies, $N={size}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_all_eigenfrequencies(eigs,params):

    nb_freq = len(eigs)
    eig_0 = next(iter(eigs.values()))
    size = len(eig_0)
    params = list(params.values())
    plt.figure(figsize=(10, 6))
    x = np.arange(1,size+1)
    colors = plt.get_cmap("inferno")(np.linspace(0.1,0.9,nb_freq))
    for i, freq in enumerate(eigs.values()):
        y = freq.real
        # plt.plot(x, y, color = colors[i], label = rf"$\varepsilon = {params[i][2]}$")
        plt.plot(x, y, color = colors[i], label = rf"$R = {params[i][0]}$")

    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend(loc='best')
    # plt.title(rf'Circular Eigenfrequencies, $N={size}$, $R = {params[0][0]}$, $r = {params[0][1]}$')
    plt.title(rf'Circular Eigenfrequencies, $N={size}$, $r = {params[0][1]}$, $\varepsilon = {params[0][2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_modes(matrice,params):
    N_points, N_modes = matrice.shape

    plt.figure(figsize=(10, 6))
    
    x = np.arange(1,N_points+1) 
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,N_modes))

    for i in range(N_modes):
        mode = matrice[:, i]
        norm = np.linalg.norm(mode, ord=2)
        if (i==N_modes-1):
            plt.plot(x, mode, color='k', linewidth = 1.)
        else:
            plt.plot(x, mode, color=colors[i], linewidth = 1.)

    # y = [4-x_i for x_i in x[1:10]]
    # plt.plot(x[1:10], y, "k")
    plt.xlabel(r"Site index $n$")
    plt.ylabel(r"$\psi_k(n)$")
    plt.xscale("linear")
    plt.yscale("linear")
    # plt.title(rf'Circular Modes, $N={N_modes}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def compute_average_distribution(modes):
    N = len(modes)
    avg = []
    for n in range(0,N):
        avg_n = 0
        for k in range(0,N):
            avg_n += np.abs(modes[n,k])**2
        avg.append(avg_n/N)
    return(avg)


def plot_avgerage_distribution(modes,params):

    nb_modes = len(modes)

    plt.figure(figsize=(10, 6))
    # params = list(params.values())
    arr = np.array(list(modes.keys()), dtype=float)
    print(arr)
    # colors = plt.get_cmap("inferno")(np.linspace(0.3,0.9,nb_modes))

    cmap = plt.get_cmap("inferno")

    # Créer la colormap tronquée en précisant positions ET couleurs
    colors_trunc = cmap(np.linspace(0.2, 0.9, nb_modes))
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno",colors_trunc)
    norm = plt.Normalize(vmin=arr.min(), vmax=arr.max())
    sm = cm.ScalarMappable(norm=norm, cmap=trunccmap)
    # colors = trunccmap(np.linspace(0, 1, nb_modes))

    for i, modes_i in enumerate(modes.values()):
        size = len(modes_i)
        x = np.arange(1, size+1)
        avg = compute_average_distribution(modes_i)
        # plt.plot(x, avg, color = colors[i], label = rf"$\varepsilon = {params[i][2]}$")
        plt.plot(x, np.ones(size)/size, color = colors_trunc[i], linestyle = '-.')
        plt.plot(x, avg, color = colors_trunc[i])

    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(r"Site index $n$")
    plt.ylabel(r"$\rho(n)$")
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'$R$', rotation=0)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def plot_matrix_entries_heatmap(matrix):

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("inferno")
    im = plt.imshow(matrix, cmap=cmap, aspect='auto', origin='upper')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title("")
    plt.colorbar(im, label="Value")

    # if show_values:
    #     for i in range(matrix.shape[0]):
    #         for j in range(matrix.shape[1]):
    #             plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")

    plt.tight_layout()
    plt.show()

def get_mat_entries(mat):

    rows, cols = mat.shape
    min_offset = -(rows - 1)
    max_offset = cols - 1

    diag_means = []
    offsets = []

    for offset in range(min_offset, max_offset + 1):
        diag = np.diagonal(mat, offset=offset)
        mean_val = np.mean(diag)
        diag_means.append(mean_val)
        offsets.append(offset)

    return diag_means, offsets

def plot_matrix_entries(mats,params):

    nb_mats = len(mats)
    mat_0 = next(iter(mats.values()))
    size = len(mat_0)
    params = list(params.values())
    plt.figure(figsize=(10,6))
    colors = plt.get_cmap("inferno")(np.linspace(0.3,0.9,nb_mats))

    for i, mat in enumerate(mats.values()):
        entries, offsets = get_mat_entries(mat)
        plt.plot(offsets, np.abs(entries), color = colors[i], linestyle = "-.", label = rf"$R = {params[i][0]}$")

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel("off-diagonal distance")
    plt.ylabel("Average matrix entries")
    plt.legend(loc = 'best')
    plt.title(rf'Matrix mean diagonal entries, $N={size}$, $r = {params[0][1]}$, $\varepsilon = {params[0][2]}$')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def compute_IPR(mode):
    N = len(mode)
    num = 0
    denom = 0
    for i in range(0,N):
        num += np.abs(mode[i])**4
        denom += np.abs(mode[i])**2
    IPR = num/(denom**2)
    return(IPR)

def  plot_IPR(modes, list):
    nb_modes = len(modes)
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,nb_modes))
    sizes = []
    avgs = []
    for i, mode in enumerate(modes.values()):
        size = len(mode)
        sizes.append(size)
        IPRs = []
        for j in range(0,size):
            mod = mode[:,j]
            IPR = compute_IPR(mod)
            IPRs.append(IPR)
            plt.scatter(list[i], IPR, color = colors[i], marker="*", s=10)
        avg = np.mean(IPRs)
        avgs.append(avg)

    print(avgs)
    print(list)
    # sizes_list = np.arange(1, np.max(sizes)+1)
    plt.plot(list, 1./np.array(sizes), color="k", linestyle = "-.", linewidth = 1.2, label = r'$f(N)=1/N$')
    plt.plot(list, avgs, color = "k", linestyle = "-", linewidth = 1.2, label = r'$\langle IPR(\psi_k)\rangle_k$')
    plt.xscale('linear')
    plt.yscale('log')
    # plt.yscale('symlog', linthresh=1e-15)    
    # plt.xlabel(r"Size $N$")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$IPR(\psi_k)$")
    plt.legend(loc='best')
    # plt.title(rf'Inverse participation ratio')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_matrix_entries_heatmap(matrix):

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("inferno")
    im = plt.imshow(matrix, cmap=cmap, aspect='auto', origin='upper')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title("")
    plt.colorbar(im, label="Value")

    # if show_values:
    #     for i in range(matrix.shape[0]):
    #         for j in range(matrix.shape[1]):
    #             plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")

    plt.tight_layout()
    plt.show()