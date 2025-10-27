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
    match = re.search(r'_N(\d+)_R(\d+)_eta_(m?\d+)', filename)
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
        


def plot_delocalized_modes(modes, params):

    plt.figure(figsize=(10,6))
    N_modes = len(modes)
    N_points = len(modes[0])
    x = np.arange(1,N_points+1) 
    # colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,N_modes))
    cmap = plt.get_cmap("inferno")

    colors_trunc = cmap(np.linspace(0.2, 0.9, N_modes))
    trunccmap = mcolors.LinearSegmentedColormap.from_list("trunc_inferno",colors_trunc)
    norm = plt.Normalize(vmin=params.min(), vmax=params.max())
    sm = cm.ScalarMappable(norm=norm, cmap=trunccmap)
    colors = trunccmap(np.linspace(0, 1, N_modes))

    for i, mode in enumerate(modes):
        # norm = np.linalg.norm(mode, ord=np.inf)
        norm = mode[50]
        plt.plot(x-51, np.abs(mode)/norm, color=colors[i], linewidth = 1.)
    
    plt.xlabel(r"Site index $n$")
    plt.ylabel(r"$|\psi_k(n)|$")
    # plt.xscale("symlog", linthresh = 10e0)
    plt.yscale('log')
    # plt.xscale('log')
    plt.xscale('linear')
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'$\varepsilon$', rotation=0)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.title(r"Positive defect, mode $\psi_0$")
    # plt.title(r"Negative defect, mode $\psi_{51}$")
    plt.tight_layout()
    plt.show()





def plot_modes(matrice):
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


def plot_real_eigenfrequencies(eigs):

    nb_freq = len(eigs)
    eig_0 = next(iter(eigs.values()))
    size = len(eig_0)
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("inferno")(np.linspace(0.2,0.9,nb_freq))
    x = np.arange(1,size+1)
    for i, eig in enumerate(eigs.values()):
        for j in range(0, size):
            plt.scatter(x[j], eig[j], color = colors[i], marker = "*", s=8)
    plt.xscale('linear')
    plt.yscale('log')
    # plt.xscale('symlog', linthresh=1e-15)    
    # plt.yscale('symlog', linthresh=1e-15)    
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"$\lambda_k$")
    # plt.title(rf'Circular Eigenfrequencies, $N={size}$, $R = {params[0]}$, $r = {params[1]}$, $\varepsilon = {params[2]}$')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


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


def approximation_bandes(A, n):
    """
    Approxime une matrice carrée A en conservant seulement les n bandes
    autour de la diagonale principale.
    
    Paramètres :
    ------------
    A : numpy.ndarray
        Matrice carrée (taille m x m).
    n : int
        Nombre de bandes à conserver autour de la diagonale.
        - n=0 garde uniquement la diagonale
        - n=1 garde diagonale + bande supérieure et inférieure immédiates
    
    Retour :
    --------
    numpy.ndarray
        Matrice approximée à n bandes.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice doit être carrée.")
    
    m = A.shape[0]
    # Création d'une matrice nulle
    A_n = np.zeros_like(A, dtype=A.dtype)
    
    # On copie uniquement les coefficients à distance <= n de la diagonale
    for i in range(m):
        for j in range(max(0, i-n), min(m, i+n+1)):
            A_n[i, j] = A[i, j]
    
    return A_n



def scatter_decay(matrix):
    n, m = matrix.shape
    if n != m:
        raise ValueError("La matrice doit être carrée.")
    
    # Indices
    i, j = np.indices(matrix.shape)
    dist = np.abs(i - j)
    
    # Parties supérieures et inférieures
    upper = np.abs(matrix[i < j])   # j > i
    lower = np.abs(matrix[i > j])   # i > j
    dist_upper = dist[i < j]
    dist_lower = dist[i > j]
    
    # Tronquer à distance max
    max_dist = n
    mask_up = dist_upper <= max_dist
    mask_lo = dist_lower <= max_dist
    dist_upper, upper = dist_upper[mask_up], upper[mask_up]
    dist_lower, lower = dist_lower[mask_lo], lower[mask_lo]
    
    # Fit sur log-log : log(y) = b - alpha log(k)
    def fit_powerlaw(d, v):
        d, v = d[d > 0], v[d > 0]
        x, y = np.log(d), np.log(v)
        a, b = np.polyfit(x, y, 1)   # y ≈ a*x + b
        alpha = -a
        C = np.exp(b)
        return C, alpha
    
    C_up, alpha_up = fit_powerlaw(dist_upper, upper)
    C_lo, alpha_lo = fit_powerlaw(dist_lower, lower)
    
    # Plot
    colors = plt.get_cmap("inferno")(np.linspace(0.1, 0.9, 4))
    plt.figure(figsize=(10, 6))
    plt.scatter(dist_upper, upper, color=colors[1], marker="*", s=8, label="Upper diagonals")
    plt.scatter(dist_lower, lower, color=colors[2], marker="*", s=8, label="Lower diagonals")
    plt.xlabel(r"Distance $|i-j|$")
    plt.ylabel(r"$|(\mathcal{C}^\gamma_N)_{ij}|$")
    # Courbes ajustées
    # x_fit = np.logspace(0, np.log10(max_dist), 200)
    # plt.plot(x_fit, C_up * x_fit**(-alpha_up), color=colors[1], lw=2)
    # plt.plot(x_fit, C_lo * x_fit**(-alpha_lo), color=colors[2], lw=2)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, "both", linestyle="--", alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()



def decay_from_means(matrix, cut=20):
    n, m = matrix.shape
    if n != m:
        raise ValueError("La matrice doit être carrée.")
        
    max_dist = n - cut
    ks = np.arange(1, max_dist+1)

    upper_means, lower_means = [], []

    for k in ks:
        up = np.abs(np.diag(matrix, k=k))   # diagonale supérieure
        lo = np.abs(np.diag(matrix, k=-k))  # diagonale inférieure
        upper_means.append(up.mean() if up.size > 0 else np.nan)
        lower_means.append(lo.mean() if lo.size > 0 else np.nan)

    upper_means = np.array(upper_means)
    lower_means = np.array(lower_means)

    # Fit en log-log : log(y) = b - alpha log(k)
    def fit_powerlaw(k, y):
        mask = (y > 0) & ~np.isnan(y)
        x, z = np.log(k[mask]), np.log(y[mask])
        slope, intercept = np.polyfit(x, z, 1)
        return np.exp(intercept), -slope

    C_up, alpha_up = fit_powerlaw(ks, upper_means)
    C_lo, alpha_lo = fit_powerlaw(ks, lower_means)

    # Plot
    colors = plt.get_cmap("inferno")(np.linspace(0.1, 0.9, 4))
    plt.figure(figsize=(10,6))
    plt.scatter(ks, upper_means, s=10, color=colors[1], label=rf"Upper fit $\alpha=${alpha_up:.3f}")
    plt.scatter(ks, lower_means, s=10, color=colors[2], label=rf"Lower fit $\alpha=${alpha_lo:.3f}")

    x_fit = np.logspace(0, np.log10(max_dist), 200)
    plt.plot(x_fit, C_up * x_fit**(-alpha_up), color=colors[1], lw=2)
    plt.plot(x_fit, C_lo * x_fit**(-alpha_lo), color=colors[2], lw=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Distance $|i-j|$")
    plt.ylabel(r"$|(\mathcal{C}^\gamma_N)_{ij}|$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_complex_band_structure(mat, n_lambda=400, lambda_range=None):
    """
    Plot the complex band structure (beta vs lambda) for a banded Toeplitz matrix.
    
    Arguments
    ----------
    mat : (n,n) ndarray
        Square Toeplitz-by-bands matrix. Assumed Toeplitz: T[i,j] = a_{j-i}.
    n_lambda : int
        Number of lambda sample points (default 400).
    lambda_range : tuple (lam_min, lam_max) or None
        Range of lambda to sample. If None, use eigenvalue range of mat with a small margin.
    beta_clip : tuple (beta_min, beta_max) or None
        If provided, discard points with beta outside this interval (useful for plotting).
        Note beta = -log(|z|). Positive beta => decaying root (|z|<1).
    show : bool
        Whether to call plt.show() (default True).
    figsize : tuple
        Size of the figure.
    marker, alpha : plotting style
    max_roots_per_lambda : int or None
        If provided, only keep this many roots (by smallest |beta|) per lambda to avoid clutter.
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The created figure and axes.
    
    Notes
    -----
    - We extract symbol coefficients a_k from the Toeplitz property:
        a_k = mat[i, i+k] for any valid i (we use i = max(0,-k)).
      We then form polynomial P(z) = sum_{k=-r}^{s} a_k z^{k} - lambda.
      Multiply by z^{r} to get a polynomial in nonnegative powers and find roots with numpy.roots.
    - beta is defined as -log(|z|). If |z|==0 the root is skipped.
    """
    # --- basic checks
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat must be a square 2D numpy array.")
    n = mat.shape[0]

    # --- extract Toeplitz coefficients a_k for k in [-(n-1), ..., (n-1)]
    # a_k = mat[max(0,-k), max(0,k)]
    Kmin = -(n-1)
    Kmax = (n-1)
    a = {}  # dictionary k -> coefficient
    for k in range(Kmin, Kmax+1):
        i = max(0, -k)
        j = max(0,  k)
        a_k = mat[i, j]
        # Keep only nonzero (within machine tol) to detect band
        if np.abs(a_k) > 0:
            a[k] = a_k

    if len(a) == 0:
        raise ValueError("All matrix entries appear zero.")

    # Determine band limits r (negative side) and s (positive side)
    neg_ks = [k for k in a.keys() if k < 0]
    pos_ks = [k for k in a.keys() if k > 0]
    r = -min(neg_ks) if neg_ks else 0  # r >= 0, number of negative powers
    s = max(pos_ks) if pos_ks else 0   # s >= 0, max positive power
    # We want coefficients for k = -r ... s
    deg = r + s  # degree after multiplying by z^r
    # Build coefficient array coeffs_deg where coeffs_deg[d] is coeff for z^d (d=0..deg)
    coeffs_deg = np.zeros(deg+1, dtype=complex)
    for k in range(-r, s+1):
        coeffs_deg[k + r] = a.get(k, 0.0)

    # --- winding region: min/max of f(e^{iθ})
    thetas = np.linspace(0, 2*np.pi, 400, endpoint=False)
    z = np.exp(1j*thetas)
    fz = np.zeros_like(z, dtype=complex)
    for k, coeff in a.items():
        fz += coeff * (z**(k))
    wmin, wmax = np.real(fz).min(), np.real(fz).max()

    # --- determine lambda sampling range
    lam_min, lam_max = lambda_range
    lambdas = np.logspace(np.log(lam_min)/np.log(10), np.log(lam_max)/np.log(10), n_lambda)

    # storage for plotting
    betas_plot = []  # list of beta values
    lambdas_plot = []  # corresponding lambda for each beta

    # For each lambda solve polynomial: sum a_k z^{k+r} - lambda * z^r = 0
    # i.e. coeffs_deg with coeffs_deg[r] -= lambda, then find roots
    for lam in lambdas:
        poly = coeffs_deg.copy()
        poly[r] = poly[r] - lam  # subtract lambda * z^r term
        # Beware: if all coefficients zero (degenerate) skip
        if np.allclose(poly, 0):
            continue

        # numpy.roots expects highest-degree-first
        poly_desc = poly[::-1]
        # If leading coefficient is zero, np.roots handles but may warn. We'll trim leading zeros:
        # (np.roots will anyway accept.)
        roots = np.roots(poly_desc)

        # compute beta = -log(|z|) (skip zeros or extremely small values)
        abs_roots = np.abs(roots)
        # avoid zeros
        nz_mask = abs_roots > 0
        roots = roots[nz_mask]
        abs_roots = abs_roots[nz_mask]
        if roots.size == 0:
            continue
        betas = -np.log(abs_roots)  # positive -> decaying modes (|z|<1)

        # append
        betas_plot.append(betas)
        lambdas_plot.append(np.full_like(betas, lam, dtype=float))

    if len(betas_plot) == 0:
        raise RuntimeError("No roots found / nothing to plot.")

    betas_plot = np.concatenate(betas_plot)
    lambdas_plot = np.concatenate(lambdas_plot)

    # --- plotting
    fig, ax = plt.subplots(figsize=(10,8))

    # background regions
    eigs = np.linalg.eigvals(mat)
    eig_min, eig_max = np.real(eigs).min(), np.real(eigs).max()
    print(wmin)
    print(wmax)
    print(eig_min)
    print(eig_max)
    # Jaffard regions below and above
    ax.axhspan(lambda_range[0], wmin, facecolor="green", alpha=0.3)
    ax.axhspan(wmin, eig_min, facecolor="blue", alpha=0.3)
    ax.axhspan(eig_min, eig_max, facecolor="red", alpha=0.3)
    ax.axhspan(eig_max, wmax, facecolor="blue", alpha=0.3)
    ax.axhspan(wmax, lambda_range[1], facecolor="green", alpha=0.3)

    ax.scatter(betas_plot, lambdas_plot, s=8, marker="*", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r'$\beta = -\log|z|$')
    ax.set_ylabel(r'$\lambda$')
    ax.set_title('Complex band structure (beta vs lambda)')
    ax.grid(True)

    # Improve y-limits to show full lambda sampling
    ax.set_ylim(lam_min, lam_max)
    plt.tight_layout()
    plt.show()