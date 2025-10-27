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

def plot_localized_modes(modes, loc_index, title = None, save_name = None):
    N_points, N_modes = modes[0].shape
    nb_list = len(modes)
    print(nb_list)
    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, (mode, index) in enumerate(zip(modes, loc_index)):
        # ax[i].plot(x, np.abs(mode[:,index-1])/np.linalg.norm(mode[:,index-1], np.inf), color=colors[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
        ax[i].plot(x, np.abs(mode[:,index-1]), color=colors[i], linestyle ='-.', linewidth = 1, marker = markers[i%5], markersize = 4)
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


def plot_multi_complex_frequencies(datas, eps_list, title=None, save_name=None):
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
    markers = ["s", "x", "*", "o", "^"]
    plt.figure(figsize=(10, 6))
    for i, (data,eps) in enumerate(zip(datas,eps_list)):
        # data = np.asarray(data)
        eigvals = np.linalg.eigvals(data)
        idx = np.argsort(eigvals.real)
        eigvals = eigvals[idx]

        x = eigvals.real
        y = eigvals.imag
        plt.scatter(x, y, color=colors[i], label = rf'$\varepsilon = {eps}$', marker=markers[i%len(markers)], s = 8)
        for j, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, str(j), fontsize=8, ha='right', va='bottom')     

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

def compute_IPR_expension_coefs(skin_mat, deriv_mat, loc_index):
    eigvals, right_eigvecs = eig(skin_mat, left=False, right=True)
    eigvals_left, left_eigvecs = eig(skin_mat, left=True, right=False)
    idx = np.argsort(eigvals.real)
    # eigvals_sorted = eigvals[idx]
    eigvals_sorted = eigvals[:]
    # right_sorted = right_eigvecs[:, idx]
    right_sorted = right_eigvecs[:]
    # left_sorted = left_eigvecs[:, idx]
    left_sorted = left_eigvecs[:]

    N = skin_mat.shape[0]
    loc_mode_0 = right_sorted[:,loc_index]
    A_num = B_num = C_num = D_num = E_num = 0
    A = B = C = 0
    for i in range(N):
        a_i = loc_mode_0[i]
        b_i = 0
        for l in range(N):
            if l != loc_index:
                Ll = left_sorted[:, l]
                Rl = right_sorted[:, l]
                num = np.vdot(Ll.conj(), deriv_mat @ loc_mode_0)
                denom = (eigvals_sorted[loc_index] - eigvals_sorted[l]) * np.vdot(Ll,Rl)
                c_l = num/denom
                b_i += c_l * Rl[i]
        Re_ab_i = (a_i.conj()*b_i).real
        A_num += (b_i ** 4)
        B_num += 4 * (b_i ** 3)*Re_ab_i
        C_num += 4*Re_ab_i + 2*(a_i**2)*(b_i**2)
        D_num += 4*Re_ab_i * (a_i**2)
        E_num += (a_i**4)
        A += (a_i ** 2)
        B += 2*Re_ab_i
        C += (b_i ** 2)
    A_denom = C**2
    B_denom = 2*B*C
    C_denom = (B**2) + 2*A*C
    D_denom = 2*A*B
    E_denom = A**2
    return([A_num,B_num,C_num,D_num,E_num], [A_denom, B_denom, C_denom, D_denom, E_denom])


def plot_IPR_expension(skin_mat, deriv_mat, loc_index, eps_list):
    plt.figure(figsize=(10,6))
    [A1,B1,C1,D1,E1], [A2,B2,C2,D2,E2]= compute_IPR_expension_coefs(skin_mat, deriv_mat, loc_index)
    IPR = [(A1 * (eps**4) + B1 * (eps**3) + C1 * (eps**2) + D1 * eps + E1)/(A2 * (eps**4) + B2 * (eps**3) + C2 * (eps**2) + D2 * eps + E2) for eps in eps_list]
    plt.plot(eps_list, IPR, color = "b")
    plt.xscale('linear')
    plt.xscale('linear')
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$IPR(\varepsilon)$")
    plt.legend(loc = 'best')
    plt.title('Inverse participation ratio perturbative expension')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_IPR_SR(skin_mat, deriv_mat, loc_index, eps_list):
    plt.figure(figsize=(10,6))
    IPR = []
    for i, eps in enumerate(eps_list):
        predicted_SR_mode = compute_SR_modes(skin_mat,deriv_mat,eps)
        IPR.append(compute_IPR(predicted_SR_mode[:,loc_index]))

    plt.plot(eps_list, IPR, color = "b")
    plt.xscale('linear')
    plt.xscale('linear')
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$IPR(\varepsilon)$")
    plt.legend(loc = 'best')
    plt.title('Inverse participation ratio perturbative expension')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()




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
        plt.plot(index, IPRS, label = rf'$\varepsilon = {gamma}$', color = colors[i], linestyle = "-.")
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(r"Mode index k")
    plt.ylabel(r"$IPR(\psi_k(\varepsilon))$")
    plt.legend(loc = 'best')
    plt.title(r'Inverse participation ratio defect, $R=0.2$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_data_loc_IPR_eps(data_modes, skin_mat, deriv_mat, loc_index, eps_list):
    plt.figure(figsize=(10,6))
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,4))
    IPRS_data = []
    IPRS_predicted = []
    loc_modes = [mode[:,loc_index-1] for mode in data_modes]
    for i, (eps,loc_mode) in enumerate(zip(eps_list,loc_modes)):
        IPR_data = compute_IPR(loc_mode)
        IPRS_data.append(IPR_data)
        predicted_SR_loc_mode = compute_SR_modes(skin_mat,deriv_mat,eps)
        IPR_predicted = compute_IPR(predicted_SR_loc_mode[:,loc_index-1])
        IPRS_predicted.append(IPR_predicted)

    plt.plot(eps_list, IPRS_data, color = colors[1], label = "data", linestyle = "-", marker = "^", markersize = 3)
    plt.plot(eps_list, IPRS_predicted, color = colors[2], label = "SR predicted", linestyle = "-", marker = "s", markersize = 3)

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$IPR(\psi_N(\varepsilon))$")
    plt.legend(loc = 'best')
    plt.title('Inverse participation ratio')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()



def plot_data_loc_IPR_eps_all_modes(data_modes, eps_list):
    """
    Trace l'IPR en fonction de ε pour chaque mode k, en couleur selon k.
    """
    plt.figure(figsize=(10, 6))

    num_modes = data_modes[0].shape[1]  # colonnes = modes
    # Colormap pour epsilon
    cmap = plt.get_cmap("inferno")
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno", cmap(np.linspace(0.2, 0.8, 256)))
    norm = mcolors.Normalize(vmin=0, vmax=num_modes - 1)
    sm = cm.ScalarMappable(norm=norm, cmap=trunccmap)

    for k in range(num_modes):
        IPRS_data = []
        for i, mode in enumerate(data_modes):
            loc_mode_k = mode[:, k]
            IPR_k = compute_IPR(loc_mode_k)
            IPRS_data.append(IPR_k)

        plt.plot(eps_list, IPRS_data/np.max(IPRS_data),
                 color=trunccmap(k / (num_modes - 1)),
                 linestyle="-", marker='o', markersize=2)

    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$IPR(\psi_k(\varepsilon))$")
    plt.title('Normalized IPR for all modes, $\gamma = 1.0$')
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r"Index of mode $k$", rotation=270, labelpad=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
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


mat_N101_R02_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps_m00002 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00002.mat', 'modes_skin','real')
mat_N101_R02_eps_m00003 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00003.mat', 'modes_skin','real')
mat_N101_R02_eps_m00004 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00004.mat', 'modes_skin','real')
mat_N101_R02_eps_m000045 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000045.mat', 'modes_skin','real')
mat_N101_R02_eps_m00005 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00005.mat', 'modes_skin','real')
mat_N101_R02_eps_m000055 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000055.mat', 'modes_skin','real')
mat_N101_R02_eps_m00006 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00006.mat', 'modes_skin','real')
mat_N101_R02_eps_m00007 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00007.mat', 'modes_skin','real')
mat_N101_R02_eps_m000073 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000073.mat', 'modes_skin','real')
mat_N101_R02_eps_m0000735 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0000735.mat', 'modes_skin','real')
mat_N101_R02_eps_m000074 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000074.mat', 'modes_skin','real')
mat_N101_R02_eps_m000075 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000075.mat', 'modes_skin','real')
mat_N101_R02_eps_m00008 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00008.mat', 'modes_skin','real')
mat_N101_R02_eps_m00009 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00009.mat', 'modes_skin','real')
mat_N101_R02_eps_m0001 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0001.mat', 'modes_skin','real')
modes_defect_gamma05 = [mat_N101_R02_eps0, mat_N101_R02_eps_m00002, mat_N101_R02_eps_m00003, mat_N101_R02_eps_m00004, mat_N101_R02_eps_m000045, mat_N101_R02_eps_m00005, mat_N101_R02_eps_m000055, mat_N101_R02_eps_m00006, mat_N101_R02_eps_m00007, mat_N101_R02_eps_m000073, mat_N101_R02_eps_m000074, mat_N101_R02_eps_m000075, mat_N101_R02_eps_m00008, mat_N101_R02_eps_m00009, mat_N101_R02_eps_m0001]
eps_defect_gamma05 = [0, -0.0002, -0.0003, -0.0004, -0.00045, -0.0005, -0.00055, -0.0006, -0.0007, -0.00073, -0.00074, -0.00075, -0.0008, -0.0009, -0.001]

GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps0.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00002 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00002.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00003 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00003.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00004 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00004.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000045 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000045.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00005.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000055 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000055.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00006 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00006.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00007 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00007.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000073 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000073.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m0000735 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m0000735.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000074 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000074.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000075 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000075.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00008 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00008.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00009 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00009.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m0001 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m0001.mat', 'GCM_skin','real')
GCM_defect_gamma05 = [GCM_N101_R02_eps0, GCM_N101_R02_eps_m00002, GCM_N101_R02_eps_m00003, GCM_N101_R02_eps_m00004, GCM_N101_R02_eps_m000045, GCM_N101_R02_eps_m00005, GCM_N101_R02_eps_m000055, GCM_N101_R02_eps_m00006, GCM_N101_R02_eps_m00007, GCM_N101_R02_eps_m000073, GCM_N101_R02_eps_m000074, GCM_N101_R02_eps_m000075, GCM_N101_R02_eps_m00008, GCM_N101_R02_eps_m00009, GCM_N101_R02_eps_m0001]





GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','none')

GCM_N101_R02_eps_m00005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00005.mat', 'GCM_skin','none')
deriv_mat_R02_gamma05 = (GCM_N101_R02_eps0 - GCM_N101_R02_eps_m00005)/(0.0005)



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


mat_N101_R02_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps_m0001 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0001.mat', 'modes_skin','real')
mat_N101_R02_eps_m00012 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00012.mat', 'modes_skin','real')
mat_N101_R02_eps_m00013 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00013.mat', 'modes_skin','real')
mat_N101_R02_eps_m000145 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000145.mat', 'modes_skin','real')
mat_N101_R02_eps_m000146 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000146.mat', 'modes_skin','real')
mat_N101_R02_eps_m000147 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000147.mat', 'modes_skin','real')
mat_N101_R02_eps_m000148 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000148.mat', 'modes_skin','real')
mat_N101_R02_eps_m00015 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00015.mat', 'modes_skin','real')
mat_N101_R02_eps_m0002 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0002.mat', 'modes_skin','real')
mat_N101_R02_eps_m0003 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0003.mat', 'modes_skin','real')
mat_N101_R02_eps_m00032 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00032.mat', 'modes_skin','real')
mat_N101_R02_eps_m0004 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0004.mat', 'modes_skin','real')
modes_defect_gamma1 = [mat_N101_R02_eps0, mat_N101_R02_eps_m0001, mat_N101_R02_eps_m00012, mat_N101_R02_eps_m00013, mat_N101_R02_eps_m000145, mat_N101_R02_eps_m000146, mat_N101_R02_eps_m000147, mat_N101_R02_eps_m000148, mat_N101_R02_eps_m00015, mat_N101_R02_eps_m0002, mat_N101_R02_eps_m0003, mat_N101_R02_eps_m00032, mat_N101_R02_eps_m0004]
eps_defect_gamma1 = [0, -0.001, -0.0012, -0.0013, -0.00145, -0.00146, -0.00147, -0.00148, -0.0015, -0.002, -0.003,-0.0032, -0.004]


GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps0.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m00013 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00013.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000145 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000145.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000146 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000146.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000147 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000147.mat', 'GCM_skin','real')
GCM_N101_R02_eps_m000148 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m000148.mat', 'GCM_skin','real')
GCM_defect_gamma1 = [GCM_N101_R02_eps0, GCM_N101_R02_eps_m00013, GCM_N101_R02_eps_m000145, GCM_N101_R02_eps_m000146, GCM_N101_R02_eps_m000147, GCM_N101_R02_eps_m000148]
eps_defect_gamma1 = [0, -0.0013, -0.00145, -0.00146, -0.00147, -0.00148]


###=========================================GAMMA2===============================================
GCM_folder ='./GCM_skin_files/gamma2/'
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

mat_N101_R02_eps0 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps0.mat', 'modes_skin','real')
mat_N101_R02_eps_m00025 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00025.mat', 'modes_skin','real')
mat_N101_R02_eps_m00027 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00027.mat', 'modes_skin','real')
mat_N101_R02_eps_m0003 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m0003.mat', 'modes_skin','real')

modes_defect_gamma2 = [mat_N101_R02_eps0, mat_N101_R02_eps_m00025, mat_N101_R02_eps_m00027, mat_N101_R02_eps_m0003]
eps_defect_gamma2 = [0, -0.0025, -0.0027, -0.003]




# gammas = [0.5, 1, 2]
# plot_IPR([modes_skin_gamma05[2], modes_skin_gamma1[2], modes_skin_gamma2[2]], gammas)

# plot_localized_modes([modes_defect_gamma05[0], modes_defect_gamma05[1], modes_defect_gamma05[2], modes_defect_gamma05[3]], [101, 101, 101, 101], None, None)
# plot_IPR(modes_defect_gamma05, eps_defect_gamma05)

# plot_localized_modes([modes_defect_gamma1[0], modes_defect_gamma1[1], modes_defect_gamma1[2], modes_defect_gamma1[3]], [101, 101, 101, 101], None, None)
plot_IPR(modes_defect_gamma1, eps_defect_gamma1)

# plot_localized_modes([modes_defect_gamma2[0], modes_defect_gamma2[1], modes_defect_gamma2[2], modes_defect_gamma2[3]], [101, 101, 101, 101], None, None)
# plot_IPR(modes_defect_gamma2, eps_defect_gamma2)

# plot_data_loc_IPR_eps_all_modes(modes_defect_gamma05, eps_defect_gamma05)
# plot_data_loc_IPR_eps_all_modes(modes_defect_gamma1, eps_defect_gamma1)
# plot_data_loc_IPR_eps(modes_defect_gamma05, GCM_N101_R02_eps0, deriv_mat_R02_gamma05, 101, eps_defect_gamma05)
# predicted_modes = compute_SR_modes(GCM_N101_R02_eps0, deriv_mat_R02_gamma05, -0.001) 
# plot_localized_modes([modes_defect_gamma05[14],predicted_modes],[101,101], None, None)



# plot_data_loc_IPR_eps(modes_defect_gamma1, 100, eps_defect_gamma1)

# plot_IPR_SR(GCM_N101_R02_eps0, deriv_mat_R02_gamma05, 100, np.linspace(-0.001, 0, 100))
# plot_data_loc_IPR_eps(modes_defect_gamma2, 100, eps_defect_gamma2)
# plot_IPR_expension(GCM_N101_R02_eps0, deriv_mat_R02_gamma05, 100, np.linspace(-0.001, 0, 100))
# plot_loc_IPR_eps(modes_defect_gamma1, 100, eps_defect_gamma1)

# plot_multi_complex_frequencies(GCM_defect_gamma05, eps_defect_gamma05, None, None)
# plot_multi_complex_frequencies(GCM_defect_gamma1, eps_defect_gamma1, None, None)