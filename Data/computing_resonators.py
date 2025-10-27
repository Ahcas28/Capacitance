import h5py
import numpy as np
import matplotlib.pyplot as plt

#================= Compute threshold function ====================#

def sigma(R,num_iter):
    sum = 0
    xi = np.arcsinh(np.sqrt(1-4*R*R) / (2*R))
    for i in range(0,num_iter):
        val = np.exp(2 * (2 * i + 1) * xi) - 1
        sum += 1/val
    return(sum)


# def epsM_threshold(R, gamma,num_iter):
#     gR = gamma * R
#     sumR = sigma(R, num_iter)
#     num = 2 * np.sinh(gamma) * np.sqrt(1-4*R*R) * sumR * (np.sinh(gR)-gR*np.cosh(gR))
#     denom = np.cosh(gR) * np.sinh(gR) - gR
#     return(num / denom)

# def epsP_threshold(R, gamma,num_iter):
#     gR = gamma * R
#     sumR = sigma(R, num_iter)
#     num = -2 * np.sinh(gamma) * np.sqrt(1-4*R*R) * sumR * (np.sinh(gR)-gR*np.cosh(gR))
#     denom = np.cosh(gR) * np.sinh(gR) - gR
#     return(num / denom)


# def epsM_threshold(R, gamma):
#     gR = gamma * R
#     num = -2*np.sinh(gamma) * (np.tanh(gR)-gR)/gamma
#     return(num)

# def epsP_threshold(R, gamma):
#     gR = gamma * R
#     num = 2*np.sinh(gamma) * (np.tanh(gR)-gR)/gamma
#     return(num)



def epsM_threshold1(R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = -2*np.sinh(gamma) * sinhgR * (gR * coshgR - sinhgR) / gamma
    denom = gR - coshgR * sinhgR
    return(num / denom)

def epsP_threshold1(R, gamma):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    num = 2*np.sinh(gamma) * sinhgR * (gR * coshgR - sinhgR) / gamma
    denom = gR - coshgR * sinhgR
    return(num / denom)

def epsP_threshold2(R, gamma, num_iter):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    sigmaR = sigma(R, num_iter)
    num = 6 * np.sinh(gamma) * np.sqrt(1 - 4 * R * R) * sigmaR * (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    denom = gR * gR * gR * (gR - coshgR * sinhgR)
    return(num / denom)

def epsM_threshold2(R, gamma, num_iter):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    sigmaR = sigma(R, num_iter)
    num = -6 * np.sinh(gamma) * np.sqrt(1 - 4 * R * R) * sigmaR * (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    denom = gR * gR * gR * (gR - coshgR * sinhgR)
    return(num / denom)



def epsP_threshold3(R, gamma, num_iter):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    sigmaR = sigma(R, num_iter)
    num = 2 * np.sinh(gamma) * np.sqrt(1 - 4 * R * R) * sigmaR * (gR * coshgR - sinhgR)
    denom = gR - coshgR * sinhgR
    return(num / denom)

def epsM_threshold3(R, gamma, num_iter):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    sigmaR = sigma(R, num_iter)
    num = -2 * np.sinh(gamma) * np.sqrt(1 - 4 * R * R) * sigmaR * (gR * coshgR - sinhgR)
    denom = gR - coshgR * sinhgR
    return(num / denom)


def epsP_threshold4(R, gamma, num_iter):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    sigmaR = sigma(R, num_iter)
    num = 2 * np.sinh(gamma) * (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    denom = (gR - coshgR * sinhgR) * gamma * gamma * gamma
    return(num / denom)

def epsM_threshold4(R, gamma, num_iter):
    gR = gamma * R
    coshgR = np.cosh(gR)
    sinhgR = np.sinh(gR)
    sigmaR = sigma(R, num_iter)
    num = -2 * np.sinh(gamma) * (gR * coshgR - sinhgR) * (gR * coshgR - sinhgR)
    denom = (gR - coshgR * sinhgR) * gamma * gamma * gamma
    return(num / denom)



print(epsP_threshold1(0.4, 1))
print(epsP_threshold2(0.4, 1, 10))
print(epsP_threshold3(0.4, 1, 10)-epsP_threshold2(0.4, 1, 10))
# print(epsP_threshold1(0.25, 10))



R_data = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
eps_c_data = [0.00008, 0.00035, 0.0013, 0.0035, 0.008, 0.0165, 0.035, 0.078]


R_list = np.linspace(0.1, 0.45, 100)
# GAMMA = 1
epsP_list1_gamma1 = epsP_threshold1(R_list, 1)
epsM_list1_gamma1 = epsM_threshold1(R_list, 1)
epsP_list2_gamma1 = epsP_threshold2(R_list, 1, 10)
epsM_list2_gamma1 = epsM_threshold2(R_list, 1, 10)
epsP_list3_gamma1 = epsP_threshold3(R_list, 1, 10)
epsM_list3_gamma1 = epsM_threshold3(R_list, 1, 10)
epsP_list4_gamma1 = epsP_threshold4(R_list, 1, 10)
epsM_list4_gamma1 = epsM_threshold4(R_list, 1, 10)

# GAMMA = 2
epsP_list1_gamma2 = epsP_threshold1(R_list, 2)
epsM_list1_gamma2 = epsM_threshold1(R_list, 2)
epsP_list2_gamma2 = epsP_threshold2(R_list, 2, 10)
epsM_list2_gamma2 = epsM_threshold2(R_list, 2, 10)
epsP_list3_gamma2 = epsP_threshold3(R_list, 2, 10)
epsM_list3_gamma2 = epsM_threshold3(R_list, 2, 10)
epsP_list4_gamma2 = epsP_threshold4(R_list, 2, 10)
epsM_list4_gamma2 = epsM_threshold4(R_list, 2, 10)

# GAMMA = 0.5
epsP_list4_gamma05 = epsP_threshold4(R_list, 0.5, 10)
epsM_list4_gamma05 = epsM_threshold4(R_list, 0.5, 10)


plt.plot(R_list, R_list, 'k', linestyle = '-.', linewidth = 1, label = 'R')
plt.plot(R_data, eps_c_data, 'k+', linestyle = '--', label = "data")
# plt.plot(R_list, epsP_list1_gamma1, label = "approx 1")
# plt.plot(R_list, epsM_list1_gamma1, linestyle = '-', label = "approx 1")
# plt.plot(R_list, epsP_list2_gamma1, label = "approx 2")
# plt.plot(R_list, epsM_list2_gamma1, linestyle = '-', label = "approx 2")
# plt.plot(R_list, epsP_list3_gamma1, label = "approx 3")
# plt.plot(R_list, epsM_list3_gamma1, linestyle = '-', label = "approx 3")
# plt.plot(R_list, epsP_list4_gamma1, label = "approx 3")
plt.plot(R_list, epsM_list4_gamma1, linestyle = '-', label = r"approx 4, $\gamma = 1$")
plt.plot(R_list, epsM_list4_gamma05, linestyle = '-', label = r"approx 4, $\gamma = 0.5$")
plt.plot(R_list, epsM_list4_gamma2, linestyle = '-', label = r"approx 4, $\gamma = 2$")
plt.legend(loc = "best")
plt.xlabel(r'$R$')
plt.ylabel(r'Defect threshold $\varepsilon_c$')
# plt.title(r'$\gamma = 1$')
plt.grid(True, which='both', linestyle='--', alpha = 0.7)
plt.minorticks_on()
plt.yscale('symlog', linthresh = 1e-4)
plt.show()
