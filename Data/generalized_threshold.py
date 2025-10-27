import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
from scipy.optimize import curve_fit
from scipy.linalg import eig
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib
matplotlib.rcParams['text.usetex'] = True

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


def power_law(n, alpha, C):
    return C / (n**alpha)

def exp_law(n, alpha, C):
    return(C * np.exp(-n*alpha))

def self_interaction_derivative(delta, v, R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = delta * v * v * gamma * gamma * gamma * (gR - coshgR * sinhgR)
    denom = (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    return(num/denom)


def compute_resolvante_nn(mat, freq, index):
    G = np.linalg.inv(mat - freq * np.eye(mat.shape[0]))
    return(G[index,index])


def get_mat_entries(mat,order):
    means = []
    for k in range(-order, order+1):
        diag = np.diagonal(mat, offset=k)
        mean = np.mean(diag)
        means.append(mean)
    return means

def band_approx(A, k):
    """
    Renvoie l'approximation bande de largeur k de la matrice A.
    """
    n = A.shape[0]
    B = np.zeros_like(A)
    for offset in range(-k, k + 1):
        B += np.diag(np.diagonal(A, offset=offset), k=offset)
    return B

def get_mat_approximation(mat,order):
    n = mat.shape[0]
    approx_mat = np.zeros_like(mat)

    for k in range(-order, order + 1):
        diag = np.diagonal(mat, offset=k)
        approx_mat += np.diag(diag, k)

    return approx_mat       

def compute_Toeplitz_approx(mat):
    n = mat.shape[0]
    approx_mat = np.zeros_like(mat)
    for k in range(-n + 1, n):
        diag_vals = np.diagonal(mat, offset=k)
        avg = np.mean(diag_vals)
        # Remplit la diagonale correspondante avec la moyenne
        approx_mat += np.diag([avg]*len(diag_vals), k)    
    return(approx_mat)

def plot_eigenvectors_kapprox_error(skin_mats, eig_index, gammas):
    nb_list = len(skin_mats)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    plt.figure(figsize=(10,6))
    for i, mat in enumerate(skin_mats):
        n = mat.shape[0]
        error = []
        eigvals, right_eigvecs = eig(mat, left=False, right=True)
        for k in range(0,n-1):
            kband_approx = band_approx(compute_Toeplitz_approx(mat), k)
            vec = kband_approx @ right_eigvecs[:,eig_index] - eigvals[eig_index] * right_eigvecs[:,eig_index]
            # error.append(np.linalg.norm(vec, ord=2))
            error.append(np.linalg.norm(vec, ord=2)/np.linalg.norm(right_eigvecs[:,eig_index], ord=2))

        order = np.arange(0,n-1)
        # params, _ = curve_fit(power_law, order, error, p0=[3.0, 1e-4])
        # alpha, C= params
        # plt.plot(order/n, error, color = colors[i], linestyle = "-.", label = r"$\gamma =$" + f'{gammas[i]}')
        plt.plot(order/n, error, color = colors[i], linestyle = "-.", label = r"$N =$" + f'{gammas[i]}')
        # plt.plot(order/n, power_law(order,alpha,C), color = colors[i], linestyle = "-", label = rf'fit $f(n)=({C:.2e})/n^{{{alpha:.2f}}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Relative approx $k/N$")
    plt.ylabel(r"$\|(C^\gamma_k - \lambda I)\psi\|_2$")
    plt.legend(loc = 'best')
    plt.title(r'k-order approximation eigen problem relative error, $\gamma=1.0$, $R=0.2$, '+ rf'$\psi = \psi_{{{eig_index}}}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_mat_entries(mats, gammas):
    nb_list = len(mats)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,nb_list))
    plt.figure(figsize=(10,6))
    for i, mat in enumerate(mats):
        entries = get_mat_entries(mat, 100)[101:201]
        index = np.arange(1,101)
        params, _ = curve_fit(power_law, index, entries, p0=[1.0, -1e-3])
        alpha, C = params
        plt.plot(index, np.abs(entries), color = colors[i], linestyle = "-.", label = r"$\gamma =$" + f'{gammas[i]}')
        plt.plot(index, np.abs(C/(index**alpha)), color = colors[i], linestyle = "-", label = rf'fit $f(n)=({C:.2e})/n^{{{alpha:.2f}}}$')
    # plt.xscale('symlog', linthresh=1e0)    
    # plt.xscale('symlog', linthresh=0)    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("off-diagonal distance")
    plt.ylabel("Average mat entries")
    plt.legend(loc = 'best')
    # plt.title(r'Matrices mean diagonal entries')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_Toeplitz_approx_error(mats, sizes):
    nb_list = len(mats)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,3))
    plt.figure(figsize=(10,6))
    toeplitz_distance_Fro = []
    toeplitz_distance_2 = []
    toeplitz_distance_inf = []
    for i, mat in enumerate(mats):
        Toeplitz_approx = compute_Toeplitz_approx(mat)
        # print(Toeplitz_approx)
        toeplitz_rel_error_Fro = np.linalg.norm(mat - Toeplitz_approx, "fro")/np.linalg.norm(mat, 'fro')
        # toeplitz_rel_error_Fro = np.linalg.norm(mat - Toeplitz_approx, "fro")
        # toeplitz_rel_error_2 = np.linalg.norm(mat - Toeplitz_approx, 2)/np.linalg.norm(mat, 2)
        toeplitz_rel_error_2 = np.linalg.norm(mat - Toeplitz_approx, 2)
        # toeplitz_rel_error_inf = np.linalg.norm(mat - Toeplitz_approx, np.inf)/np.linalg.norm(mat, np.inf)
        toeplitz_rel_error_inf = np.linalg.norm(mat - Toeplitz_approx, np.inf)
        toeplitz_distance_Fro.append(toeplitz_rel_error_Fro)
        toeplitz_distance_2.append(toeplitz_rel_error_2)
        toeplitz_distance_inf.append(toeplitz_rel_error_inf)
    
    plt.plot(sizes, toeplitz_distance_Fro, color = colors[0], linestyle = "-.", label = r"$\|\cdot \|_{Fr}$")
    # plt.plot(sizes, toeplitz_distance_2, color = colors[2], linestyle = "-.", label = r"$\|\cdot \|_2$")
    # plt.plot(sizes, toeplitz_distance_inf, color = colors[2], linestyle = "-.", label = r"$\|\cdot \|_{\infty}$")
    # plt.xscale('symlog', linthresh=1e0)    
    # plt.xscale('symlog', linthresh=0)    
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"Size $N$")
    plt.ylabel("Relative error")
    plt.legend(loc = 'best')
    # plt.title(r'Toeplitz approximation error')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()



def plot_diag_variance(mats, gammas):
    nb_list = len(mats)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.8,nb_list))
    plt.figure(figsize=(10,6))
    for i, mat in enumerate(mats):
        n = len(mat)
        offsets = range(-n + 1, n)
        variances = []
        for k in offsets:
            diag = np.diagonal(mat, offset=k)
            var = np.var(diag)
            variances.append(var)
        plt.plot(offsets, variances, color = colors[i], label = rf'$\gamma ={gammas[i]} $', marker = '*', linestyle = "-.") 


    # plt.xscale('symlog', linthresh=1e0)    
    # plt.xscale('symlog', linthresh=0)   
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"off diagonal $k$")
    plt.ylabel("Variance")
    plt.legend(loc = 'best')
    # plt.title(r'Variance of diagonal coefficients')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_diag_variance_size(mats):
    nb_list = len(mats)
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,nb_list))
    plt.figure(figsize=(10,6))
    for i, mat in enumerate(mats):
        n = mat.shape[0]
        max_offset = n - 1
        offsets = range(-max_offset, max_offset + 1)
        variances = []
        distances = []
        for k in offsets:
            diag = np.diagonal(mat, offset=k)
            if len(diag) > 1:  # éviter des artefacts avec une seule valeur
                var = np.var(diag)
                variances.append(var)
                distances.append(k / max_offset)  # distance normalisée

        plt.plot(distances, variances, color=colors[i], label=rf'$N = {n}$', linestyle='-.')

    # plt.xscale('symlog', linthresh=1e0)    
    # plt.xscale('symlog', linthresh=0)   
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"Relative distance $(i-j)/N$")
    plt.ylabel("Variance")
    plt.legend(loc = 'best')
    # plt.title(r'Variance of diagonal coefficients')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_size_convegence_multi_kapprox(mats_gamma, sizes, k_orders, eig_index):
    nb_list = len(k_orders)
    # Créer une colormap tronquée entre 0.25 et 0.75
    colormap = plt.get_cmap("inferno")
    truncated_colormap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno", colormap(np.linspace(0.2, 0.8, 100)))
    norm = mcolors.Normalize(vmin=np.min(k_orders), vmax=np.max(k_orders))
    colors = truncated_colormap(norm(k_orders))
    norm = mcolors.Normalize(vmin=k_orders.min(), vmax=k_orders.max())
    plt.figure(figsize=(10,6))
    for i, k_order in enumerate(k_orders):
        error = []
        for (n,mat) in zip(sizes,mats_gamma):
            eigvals, right_eigvecs = eig(mat, left=False, right=True)
            mat_approx = band_approx(mat, k_order)
            vec = mat_approx @ right_eigvecs[:,eig_index] - eigvals[eig_index] * right_eigvecs[:,eig_index]
            error.append(np.linalg.norm(vec, ord=2))

        # plt.plot(sizes, error, color = colors[i], linestyle = "-.", label = r"$k=$" + f'{k_order}')
        plt.plot(sizes, error, color = colors[i], linestyle = "-.")
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel("Problem size N")
    plt.ylabel(r"$\|(C^\gamma_k - \lambda I)\psi\|_2$")
    sm = cm.ScalarMappable(cmap=truncated_colormap, norm=norm)    # sm.set_array([])
    colorbar = plt.colorbar(sm, ax=plt.gca())
    colorbar.set_ticks(np.arange(min(k_orders), max(k_orders)+1, 5))
    colorbar.set_label('k', rotation=0, labelpad=15)
    # plt.legend(loc = 'center left')
    plt.title(r'k-order approximation eigen problem error, $R=0.2$, '+r'$\gamma=1$, '+rf'$\psi = \psi_{{{eig_index+1}}}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def plot_size_convergence_all_k(mats_size, sizes, eig_index):
    """
    Pour chaque matrice de taille N, trace l'erreur spectrale pour toutes les bandes k possibles,
    avec une colorbar continue indiquant k/N.
    """
    plt.figure(figsize=(10, 6))
    colormap = plt.get_cmap("inferno")
    truncated_colormap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno", colormap(np.linspace(0.2, 0.9, 200)))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    # colors = truncated_colormap(norm(np.))

    for n, mat in zip(sizes, mats_size):
        eigvals, right_eigvecs = eig(mat, left=False, right=True)
        psi = right_eigvecs[:, eig_index]
        lambda_i = eigvals[eig_index]

        for k in range(n-1):  # k = 0 à n-1
            matk_approx = band_approx(mat, k)
            matToeplitz_k_approx = band_approx(compute_Toeplitz_approx(mat), k)
            resk = matk_approx @ psi - lambda_i * psi
            res_Toeplitz_k = matToeplitz_k_approx @ psi - lambda_i * psi
            errk = np.linalg.norm(resk, ord=2)/np.linalg.norm(psi, ord=2)
            err_Topelitz_k = np.linalg.norm(res_Toeplitz_k, ord=2)/np.linalg.norm(psi, ord=2)

            color = truncated_colormap(k / n)
            plt.scatter(n-1, errk, color=color, s=6)
            plt.scatter(n+1, err_Topelitz_k, color=color, s=6)

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel(r"Size $N$")
    plt.ylabel(r"$\|C_k^\gamma \psi - \lambda \psi\|_2$")
    plt.title('Relative band and Relative Toeplitz band-approximation relative error, '+rf'$\psi = \psi_{{{eig_index+1}}}$')
    sm = cm.ScalarMappable(cmap=truncated_colormap, norm=norm)    # sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'$k/N$', rotation=0, labelpad=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def compute_Toeplitz_symbol(mat,s,order):
    mat_entries = get_mat_entries(mat, order-1)
    # print(mat_entries)
    f_s = 0.0
    for idx, a_k in enumerate(mat_entries):
        k = idx - order  # car on a stocké de -order à +order
        f_s += a_k * (s ** k)
    return f_s


def compute_resolvant(mat,freq):
    G = np.linalg.inv(mat - freq * np.eye(mat.shape[0]))
    return(G)


def extract_line_and_column(A, i, j):
    N = A.shape[0]
    B = np.zeros_like(A)
    B[i, :] = A[i, :]
    B[:, j] = A[:, j]
    return B[i,j]

def plot_Toeplitz_symbol(mat, eig, order, N_theta=500):
    thetas = np.linspace(0, 2*np.pi, N_theta)
    symbols = []

    # eig = np.array(eig)
    x_eig = eig.real
    y_eig = eig.imag
    print(len(x_eig))
    print(len(y_eig))
    for theta in thetas:
        s = np.exp(1j * theta)
        f_s = compute_Toeplitz_symbol(mat, s, order)
        symbols.append(f_s)

    symbols = np.array(symbols)
    
    plt.figure(figsize=(10,6))
    plt.plot(symbols.real, symbols.imag, 'b-', lw=2)
    plt.scatter(x_eig, y_eig, color='r', marker = 'o')
    plt.title("Toeplitz symbol and eigenvalues")
    plt.xscale('linear')
    plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel("Re(f(s))")
    plt.ylabel("Im(f(s))")
    plt.grid(True)
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()


def get_winding_region_edge(mat, order, N_theta = 500, tol = 1e-10):
    thetas = np.linspace(0, 2*np.pi, N_theta)
    symbols = []

    for theta in thetas:
        s = np.exp(1j * theta)
        f_s = compute_Toeplitz_symbol(mat, s, order)
        symbols.append(f_s)

    symbols = np.array(symbols)
    mask = np.abs(symbols.imag) < tol
    symbols_real_axis = np.max(symbols[mask].real)

    return(symbols_real_axis)


def plot_modes(matrice):
    N_points, N_modes = matrice.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(N_points)

    # Colormap sur l'indice du mode
    cmap = plt.get_cmap("inferno")
    colors_trunc = cmap(np.linspace(0.1, 0.9, N_modes))
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno",list(colors_trunc))

    norm = Normalize(vmin=1, vmax=N_modes)
    sm = ScalarMappable(cmap=trunccmap, norm=norm)
    sm.set_array([])

    # Tracer les modes
    for i in range(N_modes):
        color = trunccmap(norm(i))
        mode = matrice[:, i]
        norm_factor = np.linalg.norm(matrice, ord=np.inf)
        ax.plot(x, mode / norm_factor, color=color, linewidth=1.)

    ax.set_xlabel("Site index")
    ax.set_ylabel("Mode amplitude")
    ax.set_title("Non-Hermitian Skin Effect")
    ax.grid(True)

    # Ajouter une colorbar liée à l'axe actuel
    cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(1, N_modes, min(N_modes, 6), dtype=int))
    cbar.set_label("Mode index")

    plt.tight_layout()
    plt.show()


def plot_complex_frequencies(data):
    """
    Affiche des nombres complexes dans le plan complexe (Argand),
    chaque point coloré selon son index dans la liste.
    """
    N = len(data)
    data = np.asarray(data)
    x = data.real
    y = data.imag
    indices = np.arange(len(data))

    # Colormap sur les indices
    cmap = plt.get_cmap("inferno")
    colors_trunc = cmap(np.linspace(0.1, 0.9, N))
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno",list(colors_trunc))

    norm = Normalize(vmin=1, vmax=N)
    sm = ScalarMappable(cmap=trunccmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(x, y, c=indices, cmap=trunccmap, norm=norm, s=8, marker = "x")

    # Axes
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    # ax.set_yscale('symlog', linthresh=1e-15)
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("Eigenfrequencies in complex plane")
    ax.grid(True, linestyle='--', alpha=0.7)

    # Colorbar liée aux indices
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=trunccmap), ax=ax)
    cbar.set_label("Frequency index", rotation = 270, labelpad = 10)

    plt.tight_layout()
    plt.show()

def plot_winding_edge_order(mats, eigs, labels, Rs, orders, N_theta = 500, tol = 1e-10):
    Rs = np.array(Rs)
    nb_list = len(Rs)

    # Créer une colormap tronquée entre 0.25 et 0.75
    colormap = plt.get_cmap("inferno")
    truncated_colormap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno", colormap(np.linspace(0.2, 0.8, 100)))
    norm = mcolors.Normalize(vmin=Rs.min(), vmax=Rs.max())
    colors = truncated_colormap(norm(Rs))
    plt.figure(figsize=(10,6))
    for i, (mat, eig, label) in enumerate(zip(mats, eigs, labels)):
        edge_order = [get_winding_region_edge(mat, order, N_theta, tol) for order in orders]
        plt.plot(orders, edge_order, color = colors[i], linestyle = "-.")
        # plt.plot(orders, [np.max(eig.real) for i in range(0, len(orders))], color = colors[i], label = label)
    # plt.yscale('symlog', linthresh=1e-15)    
    plt.yscale('log')
    plt.xlabel("k-order")
    plt.ylabel("Winding region edge order k symbol")
    # plt.legend(loc = 'best')
    sm = cm.ScalarMappable(cmap=truncated_colormap, norm=norm)
    colorbar = plt.colorbar(sm, ax=plt.gca())
    colorbar.set_ticks(Rs)
    colorbar.set_label('R', rotation=0, labelpad=15)

    plt.title(r'Expected transition eigenvalue, $\gamma = 0.5$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_all_winding_regions_order(mat, eig, orders, N_theta = 500, title=None):
    
    nb_order = len(orders)
    colors = plt.get_cmap("inferno")(np.linspace(0.25,0.75,nb_order))
    thetas = np.linspace(0, 2*np.pi, N_theta)
    plt.figure(figsize=(10,6))

    for i, order in enumerate(orders):
        symbols = []
        for theta in thetas:
            s = np.exp(1j * theta)
            f_s = compute_Toeplitz_symbol(mat, s, order)
            symbols.append(f_s)

        symbols = np.array(symbols)
        plt.plot(symbols.real, symbols.imag, color=colors[i], label = f"{order}-approx", lw=1)

    # eig = np.array(eig)
    x_eig = eig.real
    y_eig = eig.imag
    
    plt.scatter(x_eig, y_eig, label = r'$\sigma(C^\gamma_N)$', color='b', marker = 'o', s=1.2)
    plt.title("Toeplitz symbol and eigenvalues")
    plt.xscale('linear')
    plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel("Re(f(s))")
    plt.ylabel("Im(f(s))")
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, which="major", alpha = 0.7)
    plt.minorticks_on()
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()


def estimate_self_interaction_derivative(skin_mats, deriv_mats, derivs_eps,index):
    dC_R = [(deriv_mat - skin_mat)[index,index]/deriv_eps for (skin_mat, deriv_mat, deriv_eps) in zip(skin_mats, deriv_mats, derivs_eps)]
    return(dC_R)



def compute_generalized_threshold(skin_mat, deriv_mat, deriv_eps, approx_order, R, gamma):
    omega_eps = get_winding_region_edge(skin_mat, approx_order, 500, 1e-10)
    resolvant = compute_resolvant(skin_mat, omega_eps)
    dC = -(deriv_mat - skin_mat)/deriv_eps
    theoretical_dC = self_interaction_derivative(1e-5,1,R,gamma)
    ddC = extract_line_and_column(dC,50,50)
    print(ddC)
    tr = np.trace(resolvant @ dC)
    eps = -1/tr
    return(eps)

def plot_generalized_threshold(skin_mats_gamma, deriv_mats_gamma, deriv_eps_gamma, eps_datas, Rs, gammas, approx_order):
    nb_gamma = len(gammas)
    colors = plt.get_cmap("inferno")(np.linspace(0.25,0.75,nb_gamma))
    plt.figure(figsize=(10,6))
    
    for i, (skin_mats,deriv_mats,deriv_eps,eps_measured) in enumerate(zip(skin_mats_gamma, deriv_mats_gamma, deriv_eps_gamma,eps_datas)):
        eps_i = []
        for (skin_mat, deriv_mat, deriv_eps, R) in zip(skin_mats,deriv_mats, deriv_eps, Rs):
            eps = compute_generalized_threshold(skin_mat, deriv_mat, deriv_eps, approx_order, R, gammas[i])
            eps_i.append(eps)
        plt.plot(Rs, eps_i, color = colors[i], linestyle = "-.", label = r"predicted, $\gamma =$" + f'{gammas[i]}')
        plt.plot(Rs, eps_measured, color = colors[i], label = r"measured, $\gamma =$" + f'{gammas[i]}')
    
    plt.title(r"Generalized threshold $\varepsilon_c$")
    plt.xscale('linear')
    plt.yscale('symlog', linthresh = 1e-6)
    plt.xlabel("R")
    plt.ylabel(r"$\varepsilon_c$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

        
def plot_selfint_derivative(skin_mats_gamma, deriv_mats_gamma, deriv_eps_gamma, Rs, gammas):
    nb_gamma = len(gammas)
    colors = plt.get_cmap("inferno")(np.linspace(0.25,0.75,nb_gamma))
    plt.figure(figsize=(10,6))
    
    for i, (skin_mats,deriv_mats,deriv_eps,gamma) in enumerate(zip(skin_mats_gamma, deriv_mats_gamma, deriv_eps_gamma,gammas)):
        dC_estimate = estimate_self_interaction_derivative(skin_mats, deriv_mats, deriv_eps,50)
        dC_computed = [self_interaction_derivative(1e-5, 1, R, gamma) for R in Rs]
        plt.plot(Rs, dC_estimate, color = colors[i], linestyle = "-.", label = r"data estimate; $\gamma =$"+f"{gamma}")
        plt.plot(Rs, dC_computed, color = colors[i], linestyle = "-", label = r"theoretical approx; $\gamma =$"+f"{gamma}")

    plt.legend(loc="best")
    plt.title("Self interaction derivative: model vs data")
    plt.xlabel(r"$R$")
    plt.ylabel(r"$dC^{\gamma}/d\varepsilon$")
    plt.xscale("linear")
    plt.yscale('symlog', linthresh=1e-7)    
    plt.minorticks_on()
    # Définir le minor locator manuellement pour le symlog
    minor_locator = SymmetricalLogLocator(base=10, linthresh=1e-7, subs=np.arange(2, 10)*0.1)
    plt.gca().yaxis.set_minor_locator(minor_locator)    
    plt.grid(which="major")
    plt.grid(True, which="minor", linestyle = ":", alpha = 0.8)
    plt.tight_layout()
    plt.show()






###===================GAMMA05===================
GCM_folder ='./GCM_skin_mat_files/gamma05/'
freq_folder ='./Eigvals_mat_files/gamma05/'

# R = 0.1
#eps0
GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
mat_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')

GCM_N101_R01_eps_m000001 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R01_eps_m000001.mat', 'GCM_skin','real')
mat_N101_R01_eps_m000001 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m000001.mat', 'eval_skin','none')

GCM_N101_R01_eps_m00000485 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m00000485.mat", 'GCM_skin','real')
mat_N101_R01_eps_m00000485 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m00000485.mat', 'eval_skin','none')

# R = 0.15
#eps0
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')

GCM_N101_R015_eps_m00001 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R015_eps_m00001.mat', 'GCM_skin','real')
mat_N101_R015_eps_m00001 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m00001.mat', 'eval_skin','none')

GCM_N101_R015_eps_m0000245 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m0000245.mat", 'GCM_skin','real')
mat_N101_R015_eps_m0000245 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m0000245.mat', 'eval_skin','none')

# R = 0.2
#eps0
GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')

GCM_N101_R02_eps_m00005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00005.mat', 'GCM_skin','real')
mat_N101_R02_eps_m00005 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m00005.mat', 'eval_skin','none')

GCM_N101_R02_eps_m000074 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m000074.mat", 'GCM_skin','real')
mat_N101_R02_eps_m000074 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m000074.mat', 'eval_skin','none')

# R = 0.25
#eps0
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')

GCM_N101_R025_eps_m00015 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R025_eps_m00015.mat', 'GCM_skin','real')
mat_N101_R025_eps_m00015 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m00015.mat', 'eval_skin','none')

GCM_N101_R025_eps_m00018 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps_m00018.mat", 'GCM_skin','real')
mat_N101_R025_eps_m00018 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m00018.mat', 'eval_skin','none')

# R = 0.3
#eps0
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')

GCM_N101_R03_eps_m00039 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R03_eps_m00039.mat', 'GCM_skin','real')
mat_N101_R03_eps_m00039 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m00039.mat', 'eval_skin','none')

GCM_N101_R03_eps_m0003945 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps_m0003945.mat", 'GCM_skin','real')
mat_N101_R03_eps_m0003945 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m0003945.mat', 'eval_skin','none')

# R = 0.35
#eps0
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')

GCM_N101_R035_eps_m0008 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R035_eps_m0008.mat', 'GCM_skin','real')
mat_N101_R035_eps_m0008 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m0008.mat', 'eval_skin','none')

GCM_N101_R035_eps_m000853 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R035_eps_m000853.mat', 'GCM_skin','real')
mat_N101_R035_eps_m000853 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m000853.mat', 'eval_skin','none')

# R = 0.4
#eps0
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')

GCM_N101_R04_eps_m0019 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R04_eps_m0019.mat', 'GCM_skin','real')
mat_N101_R04_eps_m0019 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0019.mat', 'eval_skin','none')

GCM_N101_R04_eps_m00198 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R04_eps_m00198.mat', 'GCM_skin','real')
mat_N101_R04_eps_m00198 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m00198.mat', 'eval_skin','none')

# R = 0.45
#eps0
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')

GCM_N101_R045_eps_m005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R045_eps_m005.mat', 'GCM_skin','real')
mat_N101_R045_eps_m005 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m005.mat', 'eval_skin','none')

GCM_N101_R045_eps_m0059 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R045_eps_m0059.mat', 'GCM_skin','real')
mat_N101_R045_eps_m0059 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0059.mat', 'eval_skin','none')


GCM_skins_gamma05 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]
deriv_mats_gamma05 = [GCM_N101_R01_eps_m000001, GCM_N101_R015_eps_m00001, GCM_N101_R02_eps_m00005, GCM_N101_R025_eps_m00015, GCM_N101_R03_eps_m00039, GCM_N101_R035_eps_m0008, GCM_N101_R04_eps_m0019, GCM_N101_R045_eps_m005]
# deriv_mats_gamma05 = [GCM_N101_R01_eps_m00000485, GCM_N101_R015_eps_m0000245, GCM_N101_R02_eps_m000074, GCM_N101_R025_eps_m00018, GCM_N101_R03_eps_m0003945, GCM_N101_R035_eps_m000853, GCM_N101_R04_eps_m00198, GCM_N101_R045_eps_m0059]
deriv_eps_gamma05 = [-0.00001, -0.0001, -0.0005, -0.0015, -0.0039, -0.008, -0.019, -0.05]
# deriv_eps_gamma05 = [-0.0000485, -0.000245, -0.00074, -0.0018, -0.003945, -0.00853, -0.0198, -0.059]
eps_data_gamma05 = [-0.0000485, -0.000245, -0.00074, -0.0018, -0.003945, -0.00853, -0.0198, -0.059]
GCM_eigs = [mat_N101_R01_eps0, mat_N101_R015_eps0, mat_N101_R02_eps0, mat_N101_R025_eps0, mat_N101_R03_eps0, mat_N101_R035_eps0, mat_N101_R04_eps0, mat_N101_R045_eps0]
labels_skins = ["R=0.1", "R=0.15", "R=0.2", "R=0.25", "R=0.3", "R=0.35", "R=0.4", "R=0.45"]
Rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]



# plot_winding_edge_order(GCM_skins_gamma05, GCM_eigs, labels_skins, Rs, np.arange(0,101), 500, 1e-10)


# print(get_mat_entries(GCM_N101_R03_eps0, 0))
# plot_all_winding_regions_order(GCM_N101_R02_eps0, mat_N101_R02_eps0, np.arange(0,101,20), 500, title=r"Winding regions for $R=0.2$ skin GCM")


###===================GAMMA1===================
GCM_folder ='./GCM_skin_mat_files/gamma1/'
freq_folder ='./Eigvals_mat_files/gamma1/'

# R = 0.1
#eps0
GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
mat_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')

GCM_N101_R01_eps_m000008 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R01_eps_m000008.mat', 'GCM_skin','real')
mat_N101_R01_eps_m000008 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m000008.mat', 'eval_skin','none')

GCM_N101_R01_eps_m00000985 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m00000985.mat", 'GCM_skin','real')
mat_N101_R01_eps_m00000985 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m00000985.mat', 'eval_skin','none')

# R = 0.15
#eps0
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')

GCM_N101_R015_eps_m000035 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R015_eps_m000035.mat', 'GCM_skin','real')
mat_N101_R015_eps_m000035 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m000035.mat', 'eval_skin','none')

GCM_N101_R015_eps_m0000483 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m0000483.mat", 'GCM_skin','real')
mat_N101_R015_eps_m0000483 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m0000483.mat', 'eval_skin','none')

# R = 0.2
#eps0
GCM_N25_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_25_R02_eps0.mat", 'GCM_skin','real')
GCM_N50_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_50_R02_eps0.mat", 'GCM_skin','real')
GCM_N75_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_75_R02_eps0.mat", 'GCM_skin','real')
GCM_N100_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_100_R02_eps0.mat", 'GCM_skin','real')
GCM_N150_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_150_R02_eps0.mat", 'GCM_skin','real')
GCM_N200_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_200_R02_eps0.mat", 'GCM_skin','real')
GCM_N250_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_250_R02_eps0.mat", 'GCM_skin','real')
GCM_N300_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_300_R02_eps0.mat", 'GCM_skin','real')

GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')

GCM_N101_R02_eps_m00013 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00013.mat', 'GCM_skin','real')
mat_N101_R02_eps_m00013 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m00013.mat', 'eval_skin','none')

GCM_N101_R02_eps_m000148 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m000148.mat", 'GCM_skin','real')
mat_N101_R02_eps_m000148 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m000148.mat', 'eval_skin','none')

GCM_N101_R02_eps_m0002 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m0002.mat', 'GCM_skin','real')
mat_N101_R02_eps_m0002 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m0002.mat', 'eval_skin','none')

GCM_N101_R02_eps_m0003 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m0003.mat', 'GCM_skin','none')
mat_N101_R02_eps_m0003 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m0003.mat', 'eval_skin','none')

# R = 0.25
#eps0
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')

GCM_N101_R025_eps_m00034 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R025_eps_m00034.mat', 'GCM_skin','real')
mat_N101_R025_eps_m00034 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m00034.mat', 'eval_skin','none')

GCM_N101_R025_eps_m000348 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R025_eps_m000348.mat', 'GCM_skin','real')
mat_N101_R025_eps_m000348 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m000348.mat', 'eval_skin','none')

# R = 0.3
#eps0
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')

GCM_N101_R03_eps_m00075 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R03_eps_m00075.mat', 'GCM_skin','real')
mat_N101_R03_eps_m00075 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m00075.mat', 'eval_skin','none')

GCM_N101_R03_eps_m0008 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R03_eps_m0008.mat', 'GCM_skin','real')
mat_N101_R03_eps_m0008 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m0008.mat', 'eval_skin','none')

# R = 0.35
#eps0
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')

GCM_N101_R035_eps_m0016 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R035_eps_m0016.mat', 'GCM_skin','real')
mat_N101_R035_eps_m0016 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m0016.mat', 'eval_skin','none')

GCM_N101_R035_eps_m00163 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R035_eps_m00163.mat', 'GCM_skin','real')
mat_N101_R035_eps_m00163 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m00163.mat', 'eval_skin','none')

# R = 0.4
#eps0
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')

GCM_N101_R04_eps_m0035 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R04_eps_m0035.mat', 'GCM_skin','real')
mat_N101_R04_eps_m0035 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0035.mat', 'eval_skin','none')

GCM_N101_R04_eps_m0035 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R04_eps_m0035.mat', 'GCM_skin','real')
mat_N101_R04_eps_m0035 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0035.mat', 'eval_skin','none')

# R = 0.45
#eps0
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')

GCM_N101_R045_eps_m0078 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R045_eps_m0078.mat', 'GCM_skin','real')
mat_N101_R045_eps_m0078 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0078.mat', 'eval_skin','none')

GCM_N101_R045_eps_m0079 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R045_eps_m0079.mat', 'GCM_skin','real')
mat_N101_R045_eps_m0079 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0079.mat', 'eval_skin','none')


GCM_skins_gamma1 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]
deriv_mats_gamma1 = [GCM_N101_R01_eps_m000008, GCM_N101_R015_eps_m000035, GCM_N101_R02_eps_m00013, GCM_N101_R025_eps_m00034, GCM_N101_R03_eps_m00075, GCM_N101_R035_eps_m0016, GCM_N101_R04_eps_m0035, GCM_N101_R045_eps_m0078]
deriv_eps_gamma1 = [-0.00008, -0.00035, -0.0013, -0.0034, -0.0075, -0.016, -0.035, -0.078]
eps_data_gamma1 = [-0.0000985, -0.000483, -0.00148, -0.00348, -0.008, -0.0163, -0.035, -0.079]

Skin_mats_R02_gamma1 = [GCM_N25_R02_eps0, GCM_N50_R02_eps0, GCM_N75_R02_eps0, GCM_N100_R02_eps0, GCM_N150_R02_eps0, GCM_N200_R02_eps0, GCM_N250_R02_eps0, GCM_N300_R02_eps0]

###===================GAMMA2===================
GCM_folder ='./GCM_skin_mat_files/gamma2/'
freq_folder ='./Eigvals_mat_files/gamma2/'

# R = 0.1
#eps0
GCM_N101_R01_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps0.mat", 'GCM_skin','real')
mat_N101_R01_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps0.mat', 'eval_skin','none')

GCM_N101_R01_eps_m00001 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R01_eps_m00001.mat', 'GCM_skin','real')
mat_N101_R01_eps_m00001 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m00001.mat', 'eval_skin','none')

GCM_N101_R01_eps_m0000195 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R01_eps_m0000195.mat", 'GCM_skin','real')
mat_N101_R01_eps_m0000195 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R01_eps_m0000195.mat', 'eval_skin','none')

# R = 0.15
#eps0
GCM_N101_R015_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps0.mat", 'GCM_skin','real')
mat_N101_R015_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps0.mat', 'eval_skin','none')

GCM_N101_R015_eps_m00005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R015_eps_m00005.mat', 'GCM_skin','real')
mat_N101_R015_eps_m00005 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m00005.mat', 'eval_skin','none')

GCM_N101_R015_eps_m000097 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R015_eps_m000097.mat", 'GCM_skin','real')
mat_N101_R015_eps_m000097 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R015_eps_m000097.mat', 'eval_skin','none')

# R = 0.2
#eps0
GCM_N101_R02_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps0.mat", 'GCM_skin','real')
mat_N101_R02_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps0.mat', 'eval_skin','none')

GCM_N101_R02_eps_m00025 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R02_eps_m00025.mat', 'GCM_skin','real')
mat_N101_R02_eps_m00025 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m00025.mat', 'eval_skin','none')

GCM_N101_R02_eps_m00027 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R02_eps_m00027.mat", 'GCM_skin','real')
mat_N101_R02_eps_m00027 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R02_eps_m00027.mat', 'eval_skin','none')

# R = 0.25
#eps0
GCM_N101_R025_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R025_eps0.mat", 'GCM_skin','real')
mat_N101_R025_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps0.mat', 'eval_skin','none')

GCM_N101_R025_eps_m0005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R025_eps_m0005.mat', 'GCM_skin','real')
mat_N101_R025_eps_m0005 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m0005.mat', 'eval_skin','none')

GCM_N101_R025_eps_m0007 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R025_eps_m0007.mat', 'GCM_skin','real')
mat_N101_R025_eps_m0007 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R025_eps_m0007.mat', 'eval_skin','none')

# R = 0.3
#eps0
GCM_N101_R03_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R03_eps0.mat", 'GCM_skin','real')
mat_N101_R03_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps0.mat', 'eval_skin','none')

GCM_N101_R03_eps_m001 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R03_eps_m001.mat', 'GCM_skin','real')
mat_N101_R03_eps_m001 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m001.mat', 'eval_skin','none')

GCM_N101_R03_eps_m0015 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R03_eps_m0015.mat', 'GCM_skin','real')
mat_N101_R03_eps_m0015 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R03_eps_m0015.mat', 'eval_skin','none')

# R = 0.35
#eps0
GCM_N101_R035_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R035_eps0.mat", 'GCM_skin','real')
mat_N101_R035_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps0.mat', 'eval_skin','none')

GCM_N101_R035_eps_m002 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R035_eps_m002.mat', 'GCM_skin','real')
mat_N101_R035_eps_m002 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m002.mat', 'eval_skin','none')

GCM_N101_R035_eps_m0031 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R035_eps_m0031.mat', 'GCM_skin','real')
mat_N101_R035_eps_m0031 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R035_eps_m0031.mat', 'eval_skin','none')

# R = 0.4
#eps0
GCM_N101_R04_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R04_eps0.mat", 'GCM_skin','real')
mat_N101_R04_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps0.mat', 'eval_skin','none')

GCM_N101_R04_eps_m005 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R04_eps_m005.mat', 'GCM_skin','real')
mat_N101_R04_eps_m005 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m005.mat', 'eval_skin','none')

GCM_N101_R04_eps_m0058 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R04_eps_m0058.mat', 'GCM_skin','real')
mat_N101_R04_eps_m0058 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R04_eps_m0058.mat', 'eval_skin','none')

# R = 0.45
#eps0
GCM_N101_R045_eps0 = read_hdf5_modes(GCM_folder + "GCM_skin_N_101_R045_eps0.mat", 'GCM_skin','real')
mat_N101_R045_eps0 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps0.mat', 'eval_skin','none')

GCM_N101_R045_eps_m01 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R045_eps_m01.mat', 'GCM_skin','real')
mat_N101_R045_eps_m01 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m01.mat', 'eval_skin','none')

GCM_N101_R045_eps_m0107 = read_hdf5_modes(GCM_folder + 'GCM_skin_N_101_R045_eps_m0107.mat', 'GCM_skin','real')
mat_N101_R045_eps_m0107 = read_hdf5_modes(freq_folder + 'eval_skin_N_101_R045_eps_m0107.mat', 'eval_skin','none')


GCM_skins_gamma2 = [GCM_N101_R01_eps0, GCM_N101_R015_eps0, GCM_N101_R02_eps0, GCM_N101_R025_eps0, GCM_N101_R03_eps0, GCM_N101_R035_eps0, GCM_N101_R04_eps0, GCM_N101_R045_eps0]
deriv_mats_gamma2 = [GCM_N101_R01_eps_m00001, GCM_N101_R015_eps_m00005, GCM_N101_R02_eps_m00025, GCM_N101_R025_eps_m0005, GCM_N101_R03_eps_m001, GCM_N101_R035_eps_m002, GCM_N101_R04_eps_m005, GCM_N101_R045_eps_m01]
deriv_eps_gamma2 = [-0.0001, -0.0005, -0.0025, -0.005, -0.01, -0.02, -0.05, -0.1]
eps_data_gamma2 = [-0.000195, -0.00097, -0.0027, -0.007, -0.015, -0.031, -0.058, -0.107]




#================================================================================================================================


GCM_skins_gammas = [GCM_skins_gamma05, GCM_skins_gamma1, GCM_skins_gamma2]
deriv_mats_gammas = [deriv_mats_gamma05, deriv_mats_gamma1, deriv_mats_gamma2]
deriv_eps_gammas = [deriv_eps_gamma05, deriv_eps_gamma1, deriv_eps_gamma2]
eps_datas = [eps_data_gamma05, eps_data_gamma1, eps_data_gamma2]
Rs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
gammas = [0.5, 1.0, 2.0]
# plot_generalized_threshold(GCM_skins_gammas,deriv_mats_gammas,deriv_eps_gammas,eps_datas,Rs,gammas,10)

# plot_selfint_derivative(GCM_skins_gammas, deriv_mats_gammas, deriv_eps_gammas, Rs, gammas)




Mats = [GCM_skins_gamma05[2], GCM_skins_gamma1[2], GCM_skins_gamma2[2]]
plot_mat_entries(Mats,gammas)

# plot_eigenvectors_kapprox_error(Mats, 75, gammas)

Sizes = [25, 50, 75, 100, 150, 200, 250, 300]
# plot_eigenvectors_kapprox_error(Skin_mats_R02_gamma1, 10, Sizes)

# plot_size_convegence_multi_kapprox(Skin_mats_R02_gamma1, Sizes, np.arange(0,24,1), 9)
# plot_size_convergence_all_k(Skin_mats_R02_gamma1, Sizes, 0)
# plot_Toeplitz_approx_error(Skin_mats_R02_gamma1, Sizes)

# plot_diag_variance(Mats, gammas)
# plot_diag_variance_size(Skin_mats_R02_gamma1)



eigs, right_eigenvectors = eig(GCM_N101_R02_eps_m0003)
# plot_modes(right_eigenvectors)
# plot_complex_frequencies(eigs)