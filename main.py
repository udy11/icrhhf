# Program to calculate Dielectric Constants for Maxwellian Plasma
# Reference: Kinetic Theory of Plasma Waves - Marco Brambilla, Chapter 4

# All units in CGS, except Temperature in keV and Magnetic Field in Tesla

import numpy as np
import scipy.special

def constants_cgs():
    ''' Defines fundamental constants in CGS units '''
    global mass_elec, mass_amu, bltz_k, speed_light, chrg_elem
    mass_elec = 9.10938215e-28
    mass_amu = 1.660538782e-24
    bltz_k = 1.38064852e-16
    speed_light = 2.99792458e10
    chrg_elem = 4.80320427e-10

def pdispersion(z):
    ''' (complex) -> complex
        Plasma Dispersion Function '''
    print('--->', z, scipy.special.erfi(z))
    return -np.sqrt(np.pi) * np.exp(-z * z) * (scipy.special.erfi(z) - 1.0j)

def dpdispersion(z):
    ''' (complex) -> complex
        Plasma Dispersion Function First Derivative '''
    return -2.0 * (1.0 + z * pdispersion(z))

def d2pdispersion(z):
    ''' (complex) -> complex
        Plasma Dispersion Function Second Derivative '''
    return -2.0 * (pdispersion(z) + z * dpdispersion(z))

def dteps(nrho_alf, mass_alf, chrg_alf, temp_alf, b0, kperp, kprll, omg, nmax):
    ''' Dielectric Constants for Maxwellian Plasma

        INPUT:
        nrho_alf = number density species alpha [float array of size nalf]
        mass_alf = particle mass of species alpha [float array of size nalf]
        chrg_alf = particle charge of species alpha [float array of size nalf]
        temp_alf = temperature of species alpha [float array of size nalf]
        b0 = magnetic field [float]
        kperp = perpendicular component of wave-vector [float]
        kprll = parallel component of wave-vector [float]
        omg = angular frequency of wave [float]
        
        OUTPUT:
        eps_** = tensor components of dielectric tensor
                 [full output is complex numpy array of size 3x3]
        
        OTHER:
        nalf = number of species [integer]
        k0 = wave-vector [float]
        nperp = [float]
        nprll = [float]
        omgp_alf = plasma angular frequency of species alpha [float array of size nalf]
        omgc_alf = cylcotron angular frequency of species alpha [float array of size nalf]
        vth_alf = thermal velocity of species alpha [float array of size nalf]
        lmbd_alf = lambda of species alpha [float array of size nalf]
        x0_alf = x0 of species alpha [float array of size nalf] '''
    
    nalf = len(nrho_alf)
    k0 = omg / speed_light
    nperp = kperp / k0
    nprll = kprll / k0
    omgp_alf = np.sqrt(4.0 * np.pi * nrho_alf / mass_alf) * chrg_alf
    omgc_alf = np.abs(chrg_alf * b0 * 1.0e4 / (mass_alf * speed_light))
    vth_alf = np.sqrt(2.0 * (temp_alf/8.61732814974056e-8) * bltz_k / mass_alf) # np.array([speed_light, speed_light])
    lmbd_alf = 0.5 * (kperp * vth_alf / omgc_alf) ** 2
    x0_alf = omg / (kprll * vth_alf)
    
    npp = nperp * nprll
    lmbdr_alf = 1.0 / lmbd_alf
    w1_alf = (omgp_alf / omg) ** 2 * np.exp(-lmbd_alf) * x0_alf
    w2_alf = (w1_alf * omg / omgc_alf) * (vth_alf / speed_light) ** 2 * x0_alf
    te_xx_alf = np.zeros(nalf, dtype = complex)
    te_xy_alf = np.zeros(nalf, dtype = complex)
    te_xz_alf = np.zeros(nalf, dtype = complex)
    te_yy_alf = np.zeros(nalf, dtype = complex)
    te_yz_alf = np.zeros(nalf, dtype = complex)
    te_zz_alf = np.zeros(nalf, dtype = complex)
    
    for n in range(-nmax, nmax + 1):
        n2 = n * n
        xn_alf = (omg - n * omgc_alf) / (kprll * vth_alf)
        print('********', omgc_alf, kprll * vth_alf)
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

constants_cgs()
nrho_alf = np.array([2.0e13, 2.0e13])
mass_alf = np.array([mass_elec, mass_amu])
chrg_alf = np.array([-chrg_elem, chrg_elem])
temp_alf = np.array([0.2, 0.2])
b0 = 5.0
kperp = 0.05
kprll = 0.05
omg = 2.11985e9
nmax = 1
print(dteps(nrho_alf, mass_alf, chrg_alf, temp_alf, b0, kperp, kprll, omg, nmax))
