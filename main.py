# Program to calculate Dielectric Constants for Maxwellian Plasma
# Reference: Kinetic Theory of Plasma Waves - Marco Brambilla, Chapter 4

import numpy as np
import scipy.constants
import scipy.special

def pdispersion(z):
    ''' Plasma Dispersion Function '''
    print('--->', z, scipy.special.erfi(z))
    return -np.sqrt(np.pi) * np.exp(-z * z) * (scipy.special.erfi(z) - 1.0j)

def dpdispersion(z):
    ''' Plasma Dispersion Function First Derivative '''
    return -2.0 * (1.0 + z * pdispersion(z))

def d2pdispersion(z):
    ''' Plasma Dispersion Function Second Derivative '''
    return -2.0 * (pdispersion(z) + z * dpdispersion(z))

def dteps(nrho_alf, mass_alf, chrg_alf, temp_alf, b0, kperp, kprll, nperp, nprll, omg, nmax):
    ''' Dielectric Constants for Maxwellian Plasma '''
    nalf = len(nrho_alf)
    omgp_alf = np.sqrt(nrho_alf / (scipy.constants.epsilon_0 * mass_alf)) * chrg_alf
    omgc_alf = np.abs(chrg_alf * b0 / mass_alf)
    vth_alf = np.sqrt(2.0 * temp_alf * scipy.constants.Boltzmann / mass_alf)
    lmbd_alf = 0.5 * (kperp * vth_alf / omgc_alf) ** 2
    x0_alf = omg / (kprll * vth_alf)
    
    npp = nperp * nprll
    lmbdr_alf = 1.0 / lmbd_alf
    w1_alf = (omgp_alf / omg) ** 2 * np.exp(-lmbd_alf) * x0_alf
    w2_alf = (w1_alf * omg / omgc_alf) * (vth_alf / scipy.constants.speed_of_light) ** 2 * x0_alf
    te_xx_alf = np.zeros(nalf, dtype = complex)
    te_xy_alf = np.zeros(nalf, dtype = complex)
    te_xz_alf = np.zeros(nalf, dtype = complex)
    te_yy_alf = np.zeros(nalf, dtype = complex)
    te_yz_alf = np.zeros(nalf, dtype = complex)
    te_zz_alf = np.zeros(nalf, dtype = complex)
    
    for n in range(-nmax, nmax + 1):
        n2 = n * n
        xn_alf = (omg - n * omgc_alf) / (kprll * vth_alf)
        in_alf = scipy.special.iv(n, lmbd_alf)
        din_alf = scipy.special.ivp(n, lmbd_alf, 1)
        pdisp_alf = np.array([pdispersion(xn) for xn in xn_alf])
        dpdisp_alf = np.array([dpdispersion(xn) for xn in xn_alf])
        
        dii_alf = din_alf - in_alf
        te_xx_alf += w1_alf * n2 * lmbdr_alf * in_alf * pdisp_alf
        te_xy_alf += w1_alf * n * dii_alf * pdisp_alf
        te_xz_alf += w2_alf * n * lmbdr_alf * in_alf * dpdisp_alf
        te_yy_alf += w1_alf * (n2 * lmbdr_alf - 2.0 * lmbd_alf * dii_alf) * pdisp_alf
        te_yz_alf += w2_alf * dii_alf * dpdisp_alf
        te_zz_alf += w1_alf * in_alf * xn_alf * dpdisp_alf
    
    eps_xx = 1.0 + np.sum(te_xx_alf)
    eps_xy = 1.0j * np.sum(te_xy_alf)
    eps_xz = -0.5 * npp * np.sum(te_xz_alf)
    eps_yy = 1.0 + np.sum(te_yy_alf)
    eps_yz = 0.5j * npp * np.sum(te_yz_alf)
    eps_zz = 1.0 - np.sum(te_zz_alf)
    return np.array([[ eps_xx,  eps_xy, eps_xz],
                     [-eps_xy,  eps_yy, eps_yz],
                     [ eps_xz, -eps_yz, eps_zz]])

nrho_alf = np.array([1.0e19, 1.0e19])
mass_alf = np.array([scipy.constants.electron_mass, scipy.constants.proton_mass])
chrg_alf = np.array([-scipy.constants.elementary_charge, scipy.constants.elementary_charge])
temp_alf = np.array([2.0e2, 2.0e2]) * 11604.5221
b0 = 2.0
kperp = 1.0e3
kprll = 1.0e3
nperp = 1.0
nprll = 1.0
omg = 2.0e8
nmax = 1
print(dteps(nrho_alf, mass_alf, chrg_alf, temp_alf, b0, kperp, kprll, nperp, nprll, omg, nmax))

