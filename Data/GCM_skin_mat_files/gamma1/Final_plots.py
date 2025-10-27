import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import scipy
import h5py
from scipy.linalg import eig
import matplotlib
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
# matplotlib.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"""
#         \usepackage{lmodern}
#         \usepackage[T1]{fontenc}
#     """
# })
def extract_params(filename):
    # Expression régulière mise à jour : accepte 'eps' ou 'epsm'
    match = re.search(r'_N_(\d+)_R(\d+)_eps_(m?\d+)', filename)
    if match:
        N = int(match.group(1))

        def to_decimal(val_str):
            # Valeur négative si commence par 'm'
            if val_str.startswith('m'):
                sign = -1
                digits = val_str[1:]
            else:
                sign = 1
                digits = val_str
            # digits = digits.lstrip("0") or "0"
            return sign * float(digits) / (10 ** (len(digits)-1))

        r = to_decimal(match.group(2))
        eps = to_decimal(match.group(3))
        # digits = match.group(3)
        # eps = float(digits) / (10 ** (len(digits)-1))

        # if digits.startswith('0'):  # cas fraction
        #     eps = float(digits) / (10 ** (len(digits)-1))
        # else:  # cas entier direct
        #     eps = float(digits)

        return [N, r, eps]
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
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9, size))
    for i in range(0,size):
        plt.scatter(x[i], y[i], color = colors[i], s=8, marker="*")
    for j, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(j+1), fontsize=8, ha='left', va='bottom')     

    plt.xscale('linear')
    plt.yscale('linear')
    # plt.xscale('symlog', linthresh=1e-15)    
    # plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel("Real part")
    plt.ylabel("Imaginary Part")
    # plt.title(rf'Eigenfrequencies in complex plane, $N={size}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axis('equal')  # Pour que le cercle unité soit rond
    plt.tight_layout()
    plt.show()


def plot_real_eigenfrequencies(eigs,list):

    nb_freq = len(eigs)
    # eig_0 = next(iter(eigs.values()))
    # size = len(eig_0)
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,nb_freq))
    x = np.arange(1,nb_freq+1)
    for i in range(0, nb_freq):
        if (i==nb_freq-1):
            plt.scatter(x[i], eigs[i], color = colors[i], marker="*", s=8)
        else:
            plt.scatter(x[i], eigs[i], color = colors[i], marker="*", s=8)

    plt.xscale('linear')
    plt.yscale('linear')
    # plt.xscale('symlog', linthresh=1e-15)    
    # plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"$\lambda_k$")
    # plt.title(rf'Circular Eigenfrequencies, $N={size}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_expansion_eigenfrequencies(eigs_list, prediction_list, lists):

    nb_R = len(lists)
    print(nb_R)
    eig_0 = next(iter(eigs_list[0].values()))
    size = len(eig_0)
    print(size)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_R))
    # x = np.arange(1,size+1)
    radius = [0.1, 0.25, 0.4]
    fig, ax = plt.subplots(nb_R, 1, figsize=(10, 6))
    for i in range(nb_R):
        Eig = []
        for j, eig in enumerate(eigs_list[i].values()):
            Eig.append(eig[size-1])

        ax[i].plot(lists[i], Eig, color = colors[i], label = rf"$R = {radius[i]}$", linestyle = "-.", marker = "*", markersize = 6)
        ax[i].plot(lists[i], prediction_list[i], color = colors[i], linestyle = "-", marker = "*", markersize = 6)
        ax[i].set_xscale("symlog", linthresh=1e-15)    
        ax[i].set_yscale("log")
        ax[i].set_xlabel(r"$\varepsilon$")
        ax[i].set_ylim(0.0001, 0.01)
        # ax[i].set_ylabel(r"Deloc eigenfrequency $E(\varepsilon)$")
        ax[i].legend(loc="best")
        ax[i].grid(True, which="both", linestyle='--', alpha=0.7)

    # fig.suptitle(title)
    # plt.xlabel(r"$\varepsilon$")
    # plt.ylabel(r"Deloc eigenfrequency $E(\varepsilon)$")
    # plt.grid(True, which='both', linestyle='--', alpha=0.7)
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


def plot_2scale_modes(modes_list, prediction_list):

    nb_R = 3
    print(nb_R)
    size = len(modes_list[0])
    print(size)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_R))
    x = np.arange(1,size+1)
    radius = [0.1, 0.25, 0.4]

    fig, ax = plt.subplots(nb_R, 1, figsize=(10, 6))
    for i in range(nb_R):
        modes = modes_list[i]
        mode = modes[:,size-1]
        ax[i].plot(x, np.abs(mode)/np.linalg.norm(mode, ord=np.inf), color = colors[i], label = rf"$R = {radius[i]}$", linestyle = "-.")
        ax[i].plot(x, prediction_list[i], color = colors[i], linestyle = "-")
        ax[i].set_xscale("linear")
        ax[i].set_yscale("log")
        ax[i].set_xlabel(r"Site index $n$")
        ax[i].set_ylabel(r"Mode value $|\psi(n)|$")
        ax[i].legend(loc="upper right")
        ax[i].grid(True)

    # fig.suptitle(title)
    # plt.xlabel(r"$\varepsilon$")
    # plt.ylabel(r"Deloc eigenfrequency $E(\varepsilon)$")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




def plot_modes(matrice,params):
    N_points, N_modes = matrice.shape

    plt.figure(figsize=(10, 6))
    
    x = np.arange(1,N_points+1) 
    # colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,N_modes+1))
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,N_modes))

    for i in range(N_modes):
        mode = matrice[:, i]
        norm = np.linalg.norm(mode, ord=2)
        if (i == N_modes-1):
            plt.plot(x, mode, color=colors[i], linestyle = "-", linewidth = 1.)
            # plt.plot(x, np.abs(mode), color=colors[2], linewidth = 1.)
            # plt.plot(x, -np.abs(mode), color=colors[2], linewidth = 1.)
        else: 
            plt.plot(x, mode, color=colors[i], linewidth = 1.)

    plt.xlabel(r"Site index $n$")
    plt.ylabel(r"$\psi_k(n)$")
    plt.xscale("linear")
    plt.yscale("linear")
    # plt.title(rf'Circular Modes, $N={N_modes}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_Nmodes(matrices,params):
    N_points, N_modes = matrices[0].shape

    num_modes = len(matrices)
    fig, ax = plt.subplots(num_modes, 1, figsize = (10, 6))
    x = np.arange(1,N_points+1) 
    # colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,N_modes+1))
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,num_modes))

    for i, matrice in enumerate(matrices):
        mode = matrice[:, N_modes-1]
        norm = np.linalg.norm(mode, ord=2)
        ax[i].plot(x, mode, color = colors[i], label = rf'$R = {params[i]}$')
        ax[i].plot(x, np.abs(mode), color = colors[i], linewidth = 1.)
        ax[i].plot(x, -np.abs(mode), color = colors[i], linewidth = 1.)
        ax[i].set_xscale("linear")
        ax[i].set_yscale("linear")
        ax[i].set_ylabel(r"$\psi(n)$")
        ax[i].grid(True, which='both', linestyle='--', alpha=0.7)
        ax[i].legend(loc="best")


    plt.xlabel(r"Site index $n$")
    # plt.ylabel(r"$\psi_k(n)$")
    # plt.grid(True, which='both', linestyle='--', alpha=0.7)
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
    positions = np.linspace(0, 1, nb_modes)
    colors_trunc = cmap(np.linspace(0.2, 0.9, nb_modes))
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno",colors_trunc)
    norm = plt.Normalize(vmin=arr.min(), vmax=arr.max())
    sm = cm.ScalarMappable(norm=norm, cmap=trunccmap)
    colors = trunccmap(np.linspace(0, 1, nb_modes))

    for i, modes_i in enumerate(modes.values()):
        size = len(modes_i)
        x = np.arange(1, size+1)
        avg = compute_average_distribution(modes_i)
        # plt.plot(x, avg, color = colors[i], label = rf"$\varepsilon = {params[i][2]}$")
        plt.plot(x, avg, color = colors[i])

    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(r"Site index $n$")
    plt.ylabel(r"$\rho(n)$")
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'$\gamma$', rotation=0)
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


# def plot_modes(matrice,params):
#     N_points, N_modes = matrice.shape

#     plt.figure(figsize=(10, 6))
    
#     x = np.arange(1,N_points+1) 
#     colors = plt.get_cmap("inferno")(np.linspace(0.1,0.9,N_modes+1))

#     for i in range(N_modes):
#         mode = matrice[:, i]
#         norm = np.linalg.norm(mode, ord=2)
#         plt.plot(x, np.abs(mode), color=colors[i], linewidth = 1.)

#     plt.xlabel("Site index")
#     plt.ylabel("Mode value")
#     plt.xscale("linear")
#     plt.yscale("linear")
#     # plt.title(rf'Circular Modes, $N={N_modes}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
#     plt.grid(True, which='both', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()



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
    IPRs = []
    for i, mode in enumerate(modes.values()):
        size = len(mode)
        sizes.append(size)
        print(i)
        # IPRs.append(IPR)

        for j in range(0,size):
            mod = mode[:,j]
            IPR = compute_IPR(mod)
            IPRs.append(IPR)
            plt.scatter(list[i], IPR, color = colors[-i-1], marker="*", s=8)
        
        # mod = mode[:,size-1]
        # IPR = compute_IPR(mod)
        # plt.scatter(list[i], IPR, color = "k", marker="*", s=8)

        avg = np.mean(IPRs)
        avgs.append(avg)

    # plt.plot(list, IPRs, color = colors[2])
    x_val = -0.003425
    plt.axvline(x_val, color='k', linestyle = ':', linewidth = 1.5)
    # plt.text(x_val-0.0003, 0.032, r"$\varepsilon=\varepsilon_c$", rotation = 90)    # sizes_list = np.arange(1, np.max(sizes)+1)
    plt.plot(list, 1./np.array(sizes), color="k", linestyle = "-.", linewidth = 1.2, label = r'$f(N)=1/N$')
    plt.plot(list, avgs, color = "k", linestyle = "-", linewidth = 1.2, label = r'$\langle IPR(\psi_k)\rangle_k$')
    # plt.xscale('linear')
    plt.yscale('log')
    # plt.xscale('linear')
    plt.xscale('symlog', linthresh=1e-6)    
    plt.xlim(-0.015, -6.5e-4)
    # plt.xlabel(r"Size $N$")
    plt.xlabel(r"Defect strenght $\varepsilon$")
    plt.ylabel(r"$IPR(\psi)$")
    plt.legend(loc='best')
    # plt.title(rf'Inverse participation ratio')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()



def compute_approx_mat(mat):
    m = np.array(mat)
    # Diagonale principale
    diag_principale = np.diag(m)
    moy_principale = diag_principale.mean()
    
    # Diagonale supérieure (au-dessus de la principale)
    diag_sup = np.diag(m, k=1)  # décalage +1
    moy_sup = diag_sup.mean()
    
    # Diagonale inférieure (en dessous de la principale)
    diag_inf = np.diag(m, k=-1)  # décalage -1
    moy_inf = diag_inf.mean()
    
    return [moy_principale, moy_sup, moy_inf]

def compute_deriv(mat, derivmat, eps):
    derivative = (mat - derivmat)/eps
    return(derivative)



def compute_approx_eigs(approx_mat, deriv, eps_list):
    E = []
    C = approx_mat[0]
    Cp = approx_mat[1]
    Cm = approx_mat[2]
    for eps in eps_list:
        val = C - (Cp+Cm) - eps * deriv * (Cp-Cm)/(Cp+Cm) - (eps**2)*(deriv**2)/(2*(Cp+Cm))
        E.append(val)
    return(E)


def compute_approx_mode(approx_mat, size):
    mode = np.zeros(size)
    Cp = approx_mat[1]
    Cm = approx_mat[2]
    print(Cm/Cp)
    n0 = size//2
    for i in range(0, size):
        if (i <= n0):
            mode[i] = 1
        else:
            mode[i] = np.pow(np.sqrt(Cm/Cp), i-n0)
    return(mode)


def plot_defect_threshold(R_list, lists):
    nb_list = len(lists)
    colors = plt.get_cmap("inferno")(np.linspace(0.2, 0.9, nb_list))
    labels = ["IPR measure", "Eye looking", "Toeplitz approx", "Tridiagonal approx"]
    plt.figure(figsize=(10,6))
    for i in range(nb_list):
        plt.plot(R_list, lists[i], color = colors[i], label = labels[i], marker = "*", markersize = 4)

    plt.xscale("linear")
    plt.yscale("symlog", linthresh=1e-7)    
    plt.xlabel(r"Radius $R$")
    plt.ylabel(r"Defect threshold $\varepsilon_c$")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc="best")
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def best_Toeplitz_approx(A):
    """
    Remplace chaque diagonale de A par la moyenne de ses éléments,
    et renvoie la matrice Toeplitz ainsi construite.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    T = np.zeros((m, n))
    
    # Parcourir toutes les diagonales identifiées par k = j - i
    for k in range(-(m-1), n):
        diag_vals = np.diag(A, k=k)
        mean_val = diag_vals.mean()
        # Remplacer la diagonale correspondante par la moyenne
        T += np.diag(np.full(len(diag_vals), mean_val), k=k)
    
    return T



def compute_critical_eigenval(mat):

    mat = np.asarray(mat, dtype=float)
    m, n = mat.shape
    # k parcourt toutes les diagonales présentes
    total = 0
    for k in range(-(m - 1), n):
        diag_vals = np.diag(mat, k=k)
        # moyenne de la diagonale k
        mean_k = diag_vals.mean()
        total += mean_k * ((-1)**k)
    return(total)


def compute_full_critical(mat, deriv):
    m = len(mat)
    # best_approx = best_Toeplitz_approx(mat)
    # print(np.linalg.norm(mat-best_approx, ord='fro')/np.linalg.norm(mat, ord='fro'))
    E_c = compute_critical_eigenval(mat)
    # print(E_c)
    res = np.linalg.inv(mat - E_c*np.eye(m,m))
    denom = np.trace(res @ deriv)
    print(denom)
    return(1/denom)

def compute_tridiagonal_critical(mat,deriv):
    N = len(mat)
    diag_p = np.diag(mat, k=1)
    Cp = diag_p.mean()
    diag_m = np.diag(mat, k=-1)
    Cm = diag_m.mean()
    diag = np.diag(mat, k=0)
    C0 = diag.mean()
    return((Cp - Cm)/deriv[N//2][N//2])
