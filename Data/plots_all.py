import h5py
import numpy as np
import matplotlib.pyplot as plt


def self_interaction_derivative(delta, v, R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = delta * v * v * gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)



def epsP_threshold4(R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = 2 * np.sinh(gamma) * (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    denom = (gR - coshgR * sinhgR) * gamma * gamma * gamma
    return(num / denom)

def epsM_threshold4(R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = -2 * np.sinh(gamma) * (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    denom = (gR - coshgR * sinhgR) * gamma * gamma * gamma
    return(num / denom)



def eps_tridiagonal_denom(delta, v, gamma, R):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = delta * v * v * gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)




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

def compute_mean_interaction_terms(mat):
    N = len(mat)
    mean0 = 0
    for i in range(0, N):
        mean0 += mat[i][i]

    mean_m1 = 0
    for i in range(1, N):
        mean_m1 += mat[i][i-1]

    mean_m2 = 0
    for i in range(2, N):
        mean_m2 += mat[i][i-2]

    mean_m3 = 0
    for i in range(3, N):
        mean_m3 += mat[i][i-3]

    mean_p1 = 0
    for i in range(0, N-1):
        mean_p1 += mat[i][i+1]

    mean_p2 = 0
    for i in range(0, N-2):
        mean_p2 += mat[i][i+2]

    mean_p3 = 0
    for i in range(0, N-3):
        mean_p3 += mat[i][i+3]

    return([mean_m3/(N-3), mean_m2 / (N-2), mean_m1/(N-1), mean0/N, mean_p1/(N-1), mean_p2/(N-2), mean_p3/(N+3)])


def compute_log_decay_rate(mat):
    N = len(mat)
    mean_m1 = 1
    for i in range(1, N):
        mean_m1 = mean_m1 * mat[i-1][i]/ mat[i][i-1]
    return(np.log(mean_m1)/2)




def compute_approx_threshold(R, gamma, delta, v, mat):
    [Cm3,Cm2,Cm1,C0,Cp1,Cp2,Cp3] = compute_mean_interaction_terms(mat)
    deriv = self_interaction_derivative(delta, v, R, gamma)
    return( -(Cp1 - Cm1)/deriv )




def effective_ampl_tri(mat, R, gamma, n, eps, alpha):
    eta = eps * n
    dC = self_interaction_derivative(1e-5, 1, R, gamma)
    [Cm3, Cm2, Cm1, C0, Cp1, Cp2, Cp3] = compute_mean_interaction_terms(mat)
    # rate = compute_log_decay_rate(mat)
    # Cm = (Cm1 + Cm2 + Cm3)
    # Cp = (Cp1 + Cp2 + Cp3)
    Cm = Cm1
    Cp = Cp1
    f_eta = np.exp(-(eps/np.abs(eps)) * dC * np.abs(eta) / (Cm + Cp))
    # f_eta = np.exp(dC * np.abs(eta) / (Cm1 + Cp1))
    ratio = (np.sqrt(1/((eps * dC) * (eps * dC)) + 4 * Cm1 * Cp1) + 1/(eps * dC))/(2*Cp1)
    if n <= 0:
        val = f_eta
    else:
        ev = eps * dC
        ratio = (np.sqrt(ev * ev + 4 * Cp1 * Cm1) + np.sqrt(ev * ev))/(2*Cp1)
        ratio = np.sqrt(Cm1/Cp1)
        val = f_eta * np.exp(-ratio * n)
        # val = np.pow(ratio, n)
    return(val)

def effective_ampl_quint(mat, n, eps, alpha):
    eta = eps * n
    [Cm3, Cm2, Cm1, C0, Cp1, Cp2, Cp3] = compute_mean_interaction_terms(mat)
    f_eta = np.exp(-alpha * np.abs(eta) / (Cm1 + Cp1))
    if n <= 0:
        val = f_eta
    else:
        rates = np.roots([1, (Cp1/Cp2 - 1), (Cm1 - Cm2)/Cp2, Cm2/Cp2])
        # print(rates)
        val = f_eta * (np.pow(rates[1], n) + np.pow(rates[2], n))
        # val = f_eta * (np.pow(rates[0], n) + np.pow(rates[1], n) + np.pow(rates[2], n))
    return(val)





def effective_ampl_approx(gamma, n, eps, alpha):
    eta = eps * n
    f_eta = np.exp(-alpha * np.abs(eta) / 2 * np.cosh(gamma))
    if n <= 0:
        val = f_eta
    else:
        val = f_eta * np.exp(-2 * n * gamma)
    return(val)


def plot_2scale_localized_modes(modes, GC_matrices, loc_index, labels, R_list, gamma_list, eps_list, alpha_list, title = None, save_name = None):
    N_points, N_modes = modes[0].shape
    nb_list = len(modes)
    print(nb_list)
    x = np.arange(N_points)  # Indices des points (0, 1, 2, ...)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    markers = ["s", "x", "*", "o", "^"]

    fig, ax = plt.subplots(nb_list, 1, figsize=(10, 6))
    for i, (mode, mat, gamma) in enumerate(zip(modes, GC_matrices, gamma_list)):
        mode_ponderated = np.exp(0.5*gamma * x)*np.abs(mode[:,loc_index-1])
        ampl_tri = np.array([effective_ampl_tri(mat, R_list[i], gamma, x_i - 50, eps_list[i], alpha=alpha_list[i]) for x_i in x])
        ampl_quint = np.array([effective_ampl_quint(mat, x_i - 50, eps_list[i], alpha=alpha_list[i]) for x_i in x])
        # ampl_approx = np.array([effective_ampl_approx(1, x_i - 50, eps_list[i], alpha=alpha_list[i]) for x_i in x])
        # ax[i].plot(x[40:61], (np.exp(gamma * x)*np.abs(mode[:,loc_index-1]))[40:61]/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-', linewidth = 1, marker = markers[i%5], markersize = 4)
        # ax[i].plot(x, np.exp(0.5 * gamma * x)*np.abs(mode[:,loc_index-1])/np.linalg.norm(mode[:,loc_index-1], np.inf), color=colors[i], label = labels[i], linestyle ='-', linewidth = 1, marker = markers[i%5], markersize = 4)
        ax[i].plot(x, mode_ponderated/np.linalg.norm(mode_ponderated, np.inf), color=colors[i], label = labels[i], linestyle ='-', linewidth = 1, marker = markers[i%5], markersize = 4)
        # ax[i].plot(x[40:61], np.abs(ampl_tri)[40:61], 'k', label = r'$f(\varepsilon n)|U(n)|$', linestyle = '-', linewidth = 1)
        # ax[i].plot(x, np.abs(ampl_tri), 'k', label = r'$f(\varepsilon n)|U(n)|$', linestyle = '-', linewidth = 1)
        # ax[i].plot(x, np.abs(ampl_quint), 'g', label = r'$f(\varepsilon n)|U(n)|$, $\mathcal{C}^\gamma_{\pm 1}$ numerical quint', linestyle = '-', linewidth = 1)
        # ax[i].plot(x[:58], np.abs(ampl_approx)[:58], 'k', label = r'$f(\varepsilon n)|U(n)|$, $\mathcal{C}^\gamma_{\pm 1} = e^{\pm \gamma}$', linestyle = '-.', linewidth = 1)
        # ax[i].plot(x, -ampl, 'k', linestyle = '--', linewidth = 1)
        # ax[i].set_ylim(-1.5, 1.5)
        ax[i].set_xscale("linear")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("Index")
        ax[i].set_ylabel("Mode value")
        ax[i].legend(loc="best")
        ax[i].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()






GCM_folder ='./GCM_skin_mat_files/gamma1/'
mode_folder ='./Modes_mat_files/gamma1/'

# R = 0.1
GCM_N101_R01_eps_m00000985 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m00000985.mat", 'GCM_skin','real')
mat_N101_R01_eps_m00000985 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R01_eps_m00000985.mat', 'modes_skin','real')

# R = 0.15
GCM_N101_R015_eps_m0000483 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m0000483.mat", 'GCM_skin','real')
mat_N101_R015_eps_m0000483 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R015_eps_m0000483.mat', 'modes_skin','real')

# R = 0.2
GCM_N101_R02_eps_m000148 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m000148.mat", 'GCM_skin','real')
mat_N101_R02_eps_m000148 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000148.mat', 'modes_skin','real')

# R = 0.25
GCM_N101_R025_eps_m000348 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m000348.mat", 'GCM_skin','real')
mat_N101_R025_eps_m000348 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R025_eps_m000348.mat', 'modes_skin','real')

# R = 0.3
GCM_N101_R03_eps_m0008 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0008.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0008 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R03_eps_m0008.mat', 'modes_skin','real')

# R = 0.35
GCM_N101_R035_eps_m00163 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m00163.mat", 'GCM_skin','real')
mat_N101_R035_eps_m00163 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R035_eps_m00163.mat', 'modes_skin','real')

# R = 0.4
GCM_N101_R04_eps_m0035 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m0035.mat", 'GCM_skin','real')
mat_N101_R04_eps_m0035 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R04_eps_m0035.mat', 'modes_skin','real')

# R = 0.45
GCM_N101_R045_eps_m0079 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0079.mat", 'GCM_skin','real')
mat_N101_R045_eps_m0079 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R045_eps_m0079.mat', 'modes_skin','real')

GC_matrices_gamma1 = [GCM_N101_R01_eps_m00000985, GCM_N101_R015_eps_m0000483, GCM_N101_R02_eps_m000148, GCM_N101_R025_eps_m000348, GCM_N101_R03_eps_m0008, GCM_N101_R035_eps_m00163, GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0079]

# modes_R1 = [mat_N101_R01_eps_m000008, mat_N101_R015_eps_m000035]
# GC_matrices_R1 = [GCM_N101_R01_eps_m000008, GCM_N101_R015_eps_m000035]
# labels_R1 = [r"$R = 0.1$, $\varepsilon_c = -0.00008$", r"$R = 0.15$, $\varepsilon_c = -0.00035$"]
# plot_2scale_localized_modes(modes = modes_R1,
#                             GC_matrices=GC_matrices_R1,
#                             loc_index = 101, 
#                             labels = labels_R1,
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $\varepsilon = 0$",
#                             save_name=None)

# modes_R2 = [mat_N101_R02_eps_m00013, mat_N101_R025_eps_m00035]
# GC_matrices_R2 = [GCM_N101_R02_eps_m00013, GCM_N101_R025_eps_m00035]
# labels_R2 = [r"$R = 0.2$, $\varepsilon_c = -0.0013$", r"$R = 0.25$, $\varepsilon_c = -0.0035$"]
# plot_2scale_localized_modes(modes = modes_R2,
#                             GC_matrices=GC_matrices_R2,
#                             loc_index = 101, 
#                             labels = labels_R2,
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $\varepsilon = 0$",
#                             save_name=None)

# modes_R3 = [mat_N101_R03_eps_m0008, mat_N101_R035_eps_m0016]
# GC_matrices_R3 = [GCM_N101_R03_eps_m0008, GCM_N101_R035_eps_m0016]
# labels_R3 = [r"$R = 0.3$, $\varepsilon_c = -0.008$", r"$R = 0.35$, $\varepsilon_c = -0.016$"]
# plot_2scale_localized_modes(modes = modes_R3,
#                             GC_matrices=GC_matrices_R3,
#                             loc_index = 101, 
#                             labels = labels_R3,
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $\varepsilon = 0$",
#                             save_name=None)

# modes_R4 = [mat_N101_R04_eps_m0035, mat_N101_R045_eps_m0078]
# GC_matrices_R4 = [GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0078]
# labels_R4 = [r"$R = 0.4$, $\varepsilon_c = -0.035$", r"$R = 0.45$, $\varepsilon_c = -0.078$"]
# plot_2scale_localized_modes(modes = modes_R4,
#                             GC_matrices=GC_matrices_R4,
#                             loc_index = 101, 
#                             labels = labels_R4,
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $\varepsilon = 0$",
#                             save_name=None)


GCM_folder ='./GCM_skin_mat_files/gamma2/'
mode_folder ='./Modes_mat_files/gamma2/'

# R = 0.1
GCM_N101_R01_eps_m0000195 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m0000195.mat", 'GCM_skin','real')
mat_N101_R01_eps_m0000195 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R01_eps_m0000195.mat', 'modes_skin','real')

# R = 0.15
GCM_N101_R015_eps_m000097 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m000097.mat", 'GCM_skin','real')
mat_N101_R015_eps_m000097 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R015_eps_m000097.mat', 'modes_skin','real')

# R = 0.2
GCM_N101_R02_eps_m00027 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m00027.mat", 'GCM_skin','real')
mat_N101_R02_eps_m00027 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m00027.mat', 'modes_skin','real')

# R = 0.25
GCM_N101_R025_eps_m0007 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m0007.mat", 'GCM_skin','real')
mat_N101_R025_eps_m0007 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R025_eps_m0007.mat', 'modes_skin','real')

# R = 0.3
GCM_N101_R03_eps_m0015 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0015.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0015 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R03_eps_m0015.mat', 'modes_skin','real')

# R = 0.35
GCM_N101_R035_eps_m0031 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m0031.mat", 'GCM_skin','real')
mat_N101_R035_eps_m0031 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R035_eps_m0031.mat', 'modes_skin','real')

# R = 0.4
GCM_N101_R04_eps_m0058 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m0058.mat", 'GCM_skin','real')
mat_N101_R04_eps_m0058 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R04_eps_m0058.mat', 'modes_skin','real')

# R = 0.45
GCM_N101_R045_eps_m0107 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0107.mat", 'GCM_skin','real')
mat_N101_R045_eps_m0107 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R045_eps_m0107.mat', 'modes_skin','real')

GC_matrices_gamma2 = [GCM_N101_R01_eps_m0000195, GCM_N101_R015_eps_m000097, GCM_N101_R02_eps_m00027, GCM_N101_R025_eps_m0007, GCM_N101_R03_eps_m0015, GCM_N101_R035_eps_m0031, GCM_N101_R04_eps_m0058, GCM_N101_R045_eps_m0107]




# modes_R1 = [mat_N101_R01_eps_m000008, mat_N101_R01_eps_m000013]
# GC_matrices_R1 = [GCM_N101_R01_eps_m000008, GCM_N101_R01_eps_m000013]
# labels_R1 = [r"$\gamma = 1$, $\varepsilon_c = -0.00008$", r"$\gamma=2$, $\varepsilon_c = -0.00013$", ]
# plot_2scale_localized_modes(modes = modes_R1,
#                             GC_matrices=GC_matrices_R1,
#                             loc_index = 101, 
#                             labels = labels_R1,
#                             R_list = [0.1, 0.15],
#                             gamma_list = [1, 2],
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.1$, $\varepsilon = 0$",
#                             save_name=None)



# modes_R2 = [mat_N101_R02_eps_m00013, mat_N101_R02_eps_m00027]
# GC_matrices_R2 = [GCM_N101_R02_eps_m00013, GCM_N101_R02_eps_m00027]
# labels_R2 = [r"$\gamma = 1$, $\varepsilon_c = -0.00013$", r"$\gamma=2$, $\varepsilon_c = -0.0027$", ]
# plot_2scale_localized_modes(modes = modes_R2,
#                             GC_matrices=GC_matrices_R2,
#                             loc_index = 101, 
#                             labels = labels_R2,
#                             R_list = [0.2, 0.25],
#                             gamma_list = [1, 2],
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.2$, $\varepsilon = 0$",
#                             save_name=None)

# modes_R3 = [mat_N101_R03_eps_m0008, mat_N101_R03_eps_m0015]
# GC_matrices_R3 = [GCM_N101_R03_eps_m0008, GCM_N101_R03_eps_m0015]
# labels_R3 = [r"$\gamma = 1$, $\varepsilon_c = -0.008$", r"$\gamma = 2$, $\varepsilon_c = -0.015$"]
# plot_2scale_localized_modes(modes = modes_R3,
#                             GC_matrices=GC_matrices_R3,
#                             loc_index = 101, 
#                             labels = labels_R3,
#                             R_list = [0.3, 0.35],
#                             gamma_list = [1, 2],
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.3$, $\varepsilon = 0$",
#                             save_name=None)



# modes_R4 = [mat_N101_R04_eps_m0035, mat_N101_R04_eps_m0058]
# GC_matrices_R4 = [GCM_N101_R04_eps_m0035, GCM_N101_R04_eps_m0058]
# labels_R4 = [r"$\gamma = 1$, $\varepsilon_c = -0.035$", r"$\gamma = 2$, $\varepsilon_c = -0.058$"]
# plot_2scale_localized_modes(modes = modes_R4,
#                             GC_matrices=GC_matrices_R4,
#                             loc_index = 101, 
#                             labels = labels_R4,
#                             R_list = [0.4, 0.45],
#                             gamma_list = [1, 2],
#                             eps_list=[0,0],
#                             alpha_list=[-1, -1],
#                             title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.4$, $\varepsilon = 0$",
#                             save_name=None)






GCM_folder ='./GCM_skin_mat_files/gamma05/'
mode_folder ='./Modes_mat_files/gamma05/'

# R = 0.1
GCM_N101_R01_eps_m00000485 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m00000485.mat", 'GCM_skin','real')
mat_N101_R01_eps_m00000485 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R01_eps_m00000485.mat', 'modes_skin','real')

# R = 0.15
GCM_N101_R015_eps_m0000245 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m0000245.mat", 'GCM_skin','real')
mat_N101_R015_eps_m0000245 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R015_eps_m0000245.mat', 'modes_skin','real')

# R = 0.2
GCM_N101_R02_eps_m000074 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m000074.mat", 'GCM_skin','real')
mat_N101_R02_eps_m000074 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R02_eps_m000074.mat', 'modes_skin','real')

# R = 0.25
GCM_N101_R025_eps_m00018 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m00018.mat", 'GCM_skin','real')
mat_N101_R025_eps_m00018 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R025_eps_m00018.mat', 'modes_skin','real')

# R = 0.3
GCM_N101_R03_eps_m0003945 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0003945.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0003945 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R03_eps_m0003945.mat', 'modes_skin','real')

# R = 0.35
GCM_N101_R035_eps_m000853 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps_m000853.mat", 'GCM_skin','real')
mat_N101_R035_eps_m000853 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R035_eps_m000853.mat', 'modes_skin','real')

# R = 0.4
GCM_N101_R04_eps_m00198 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps_m00198.mat", 'GCM_skin','real')
mat_N101_R04_eps_m00198 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R04_eps_m00198.mat', 'modes_skin','real')

# R = 0.45
GCM_N101_R045_eps_m0059 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps_m0059.mat", 'GCM_skin','real')
mat_N101_R045_eps_m0059 = read_hdf5_modes(mode_folder + 'modes_skin_N_101_R045_eps_m0059.mat', 'modes_skin','real')

GC_matrices_gamma05 = [GCM_N101_R01_eps_m00000485, GCM_N101_R015_eps_m0000245, GCM_N101_R02_eps_m000074, GCM_N101_R025_eps_m00018, GCM_N101_R03_eps_m0003945, GCM_N101_R035_eps_m000853, GCM_N101_R04_eps_m00198, GCM_N101_R045_eps_m0059]



gamma_list = [0.5, 1, 2]
modes_R1 = [mat_N101_R01_eps_m00000485, mat_N101_R01_eps_m00000985, mat_N101_R01_eps_m0000195]
GC_matrices_R1 = [GCM_N101_R01_eps_m00000485, GCM_N101_R01_eps_m00000985, GCM_N101_R01_eps_m0000195]
eps_c_list_R1 = [-0.0000485, -0.0000985, -0.000195]
eps_list_R1 = [eps - compute_approx_threshold(0.1,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R1,gamma_list,GC_matrices_R1)]
labels_R1 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R1)]
print(eps_list_R1)
plot_2scale_localized_modes(modes = modes_R1,
                            GC_matrices=GC_matrices_R1,
                            loc_index = 101, 
                            labels = labels_R1,
                            R_list = [0.1, 0.1, 0.1],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R1,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.1$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)



gamma_list = [0.5, 1, 2]
modes_R15 = [mat_N101_R015_eps_m0000245, mat_N101_R015_eps_m0000483, mat_N101_R015_eps_m000097]
GC_matrices_R15 = [GCM_N101_R015_eps_m0000245, GCM_N101_R015_eps_m0000483, GCM_N101_R015_eps_m000097]
eps_c_list_R15 = [-0.000245, -0.000483, -0.00097]
eps_list_R15 = [eps - compute_approx_threshold(0.15,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R15,gamma_list,GC_matrices_R15)]
labels_R15 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R15)]
print(eps_list_R15)
plot_2scale_localized_modes(modes = modes_R15,
                            GC_matrices=GC_matrices_R15,
                            loc_index = 101, 
                            labels = labels_R15,
                            R_list = [0.15, 0.15, 0.15],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R15,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.15$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)




modes_R2 = [mat_N101_R02_eps_m000074, mat_N101_R02_eps_m000148, mat_N101_R02_eps_m00027]
GC_matrices_R2 = [GCM_N101_R02_eps_m000074, GCM_N101_R02_eps_m000148, GCM_N101_R02_eps_m00027]
eps_c_list_R2 = [-0.00074, -0.00148, -0.0027]
eps_list_R2 = [eps - compute_approx_threshold(0.2,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R2,gamma_list,GC_matrices_R2)]
labels_R2 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R2)]
print(eps_list_R2)
plot_2scale_localized_modes(modes = modes_R2,
                            GC_matrices=GC_matrices_R2,
                            loc_index = 101, 
                            labels = labels_R2,
                            R_list = [0.2, 0.2, 0.2],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R2,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.2$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)




modes_R25 = [mat_N101_R025_eps_m00018, mat_N101_R025_eps_m000348, mat_N101_R025_eps_m0007]
GC_matrices_R25 = [GCM_N101_R025_eps_m00018, GCM_N101_R025_eps_m000348, GCM_N101_R025_eps_m0007]
eps_c_list_R25 = [-0.0018, -0.00348, -0.007]
eps_list_R25 = [eps - compute_approx_threshold(0.25,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R25,gamma_list,GC_matrices_R25)]
labels_R25 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R25)]
print(eps_list_R25)
plot_2scale_localized_modes(modes = modes_R25,
                            GC_matrices=GC_matrices_R25,
                            loc_index = 101, 
                            labels = labels_R25,
                            R_list = [0.25, 0.25, 0.25],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R25,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.25$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)




modes_R3 = [mat_N101_R03_eps_m0003945, mat_N101_R03_eps_m0008, mat_N101_R03_eps_m0015]
GC_matrices_R3 = [GCM_N101_R03_eps_m0003945, GCM_N101_R03_eps_m0008, GCM_N101_R03_eps_m0015]
eps_c_list_R3 = [-0.003945, -0.008, -0.015]
eps_list_R3 = [eps - compute_approx_threshold(0.3,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R3,gamma_list,GC_matrices_R3)]
labels_R3 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R3)]
print(eps_list_R3)
plot_2scale_localized_modes(modes = modes_R3,
                            GC_matrices=GC_matrices_R3,
                            loc_index = 101, 
                            labels = labels_R3,
                            R_list = [0.3, 0.3, 0.3],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R3,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.3$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)



modes_R35 = [mat_N101_R035_eps_m000853, mat_N101_R035_eps_m00163, mat_N101_R035_eps_m0031]
GC_matrices_R35 = [GCM_N101_R035_eps_m000853, GCM_N101_R035_eps_m00163, GCM_N101_R035_eps_m0031]
eps_c_list_R35 = [-0.00853, -0.0163, -0.031]
eps_list_R35 = [eps - compute_approx_threshold(0.35,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R35,gamma_list,GC_matrices_R35)]
labels_R35 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R35)]
print(eps_list_R35)
plot_2scale_localized_modes(modes = modes_R35,
                            GC_matrices=GC_matrices_R35,
                            loc_index = 101, 
                            labels = labels_R35,
                            R_list = [0.35, 0.35, 0.35],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R35,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.35$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)


modes_R4 = [mat_N101_R04_eps_m00198, mat_N101_R04_eps_m0035, mat_N101_R04_eps_m0058]
GC_matrices_R4 = [GCM_N101_R04_eps_m00198, GCM_N101_R04_eps_m0035, GCM_N101_R04_eps_m0058]
eps_c_list_R4 = [-0.0198, -0.035, -0.058]
eps_list_R4 = [eps - compute_approx_threshold(0.4,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R4,gamma_list,GC_matrices_R4)]
labels_R4 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R4)]
print(eps_list_R4)
plot_2scale_localized_modes(modes = modes_R4,
                            GC_matrices=GC_matrices_R4,
                            loc_index = 101, 
                            labels = labels_R4,
                            R_list = [0.4, 0.4, 0.4],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R4,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.4$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)


modes_R45 = [mat_N101_R045_eps_m0059, mat_N101_R045_eps_m0079, mat_N101_R045_eps_m0107]
GC_matrices_R45 = [GCM_N101_R045_eps_m0059, GCM_N101_R045_eps_m0079, GCM_N101_R045_eps_m0107]
eps_c_list_R45 = [-0.059, -0.079, -0.107]
eps_list_R45 = [eps - compute_approx_threshold(0.45,gamma,1e-5,1,mat) for (eps,gamma,mat) in zip(eps_c_list_R45,gamma_list,GC_matrices_R45)]
labels_R45 = [r"$\gamma = $" + f'{gamma_i},'+ r'$\Delta\varepsilon =$'+ f'{delta_eps:.5e}' for (gamma_i,delta_eps) in zip(gamma_list, eps_list_R45)]
print(eps_list_R45)
plot_2scale_localized_modes(modes = modes_R45,
                            GC_matrices=GC_matrices_R45,
                            loc_index = 101, 
                            labels = labels_R45,
                            R_list = [0.45, 0.45, 0.45],
                            gamma_list = [0.5, 1, 2],
                            eps_list=eps_list_R45,
                            # eps_list=[0,0,0],
                            alpha_list=[-1, -1, -1],
                            title=r"$R_d = R + \varepsilon_c + \alpha \varepsilon$, $R = 0.45$, $\varepsilon = \varepsilon(data)-\varepsilon_c(theoretical)$",
                            save_name=None)



# GCM = [GCM_N101_R01_eps_m000008, GCM_N101_R015_eps_m000035, GCM_N101_R02_eps_m00013, GCM_N101_R025_eps_m00035, GCM_N101_R03_eps_m0008, GCM_N101_R035_eps_m0016, GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0078]
# R_list = np.arange(0.1, 0.5, 0.05)
# print(R_list)
# eps_denom_list = eps_tridiagonal_denom(1e-5, 1, 1, R_list)
# Delta_Cmp1_list = np.array([compute_mean_interaction_terms(matR)[2] for matR in GCM]) - np.array([compute_mean_interaction_terms(matR)[1] for matR in GCM])
# print(Delta_Cmp1_list)
# eps_threshold_list = [Delta_Cmp1_list[i] / eps_denom_list[i] for i in range(0,len(R_list))]
R_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
approx_threshold_gamma05 = [compute_approx_threshold(R, 0.5, 1e-5, 1, mat) for (R, mat) in zip(R_list,GC_matrices_gamma05)]
approx_threshold_gamma1 = [compute_approx_threshold(R, 1, 1e-5, 1, mat) for (R, mat) in zip(R_list,GC_matrices_gamma1)]
approx_threshold_gamma2 = [compute_approx_threshold(R, 2, 1e-5, 1, mat) for (R, mat) in zip(R_list,GC_matrices_gamma2)]
eps_c_data_gamma05 = [-0.0000485, -0.000245, -0.00074, -0.0018, -0.003945, -0.00853, -0.0198, -0.059]
eps_c_data_gamma1 = [-0.0000985, -0.000483, -0.00148, -0.00348, -0.00785, -0.0165, -0.035, -0.079]
eps_c_data_gamma2 = [-0.000195, -0.00097, -0.0027, -0.007, -0.015, -0.031, -0.058, -0.107]

plt.plot(R_list, approx_threshold_gamma05, color = 'r', linestyle = '-.', label = r"$\gamma = 0.5$, approx")
plt.plot(R_list, approx_threshold_gamma1, color = 'b', linestyle = '-.', label = r"$\gamma = 1$, approx")
plt.plot(R_list, approx_threshold_gamma2, color = 'k', linestyle = '-.', label = r"$\gamma = 2$, approx")
plt.plot(R_list, eps_c_data_gamma05, color = 'r', linestyle = '-', label = r"$\gamma = 0.5$, data")
plt.plot(R_list, eps_c_data_gamma1, color = 'b', linestyle = '-', label = r"$\gamma = 1$, data")
plt.plot(R_list, eps_c_data_gamma2, color = 'k', linestyle = '-', label = r"$\gamma = 2$, data")
plt.legend(loc = "best")
plt.xlabel(r'$R$')
plt.ylabel(r'$\varepsilon_c$')
plt.title(r'Defect threshold $\varepsilon_c$ at first order')
plt.grid(True, which='both', linestyle='--', alpha = 0.7)
plt.minorticks_on()
# plt.yscale('log')
plt.yscale('symlog', linthresh = 1e-6)
plt.show()

