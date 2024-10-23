# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import sph_harm, jv, yv  # jv, yv = Bessel j, y functions
from sympy.physics.quantum.cg import CG     # clebsch-gordan


def vsh(j, l, m, phi, theta):
    """
    Calculates the value of the vector eigen functions of J2 operator evaluated
    for given j, l, m at the angular coordinates theta phi (theta = azimuth?.)
    (not radial part yet!).
    This is eq. (8) at the Nora Tischler et. al. "The role of angular momentum
    in the constructions of electromagnetic multipolar fields" paper.
    """
    # Normalization values (clebsch-gordan coefficients in CG(j1, m1, j2, m2,
    # j3, m3))
    # We construct this values for the eigen functions of S2 and Sz.
    cp = float(CG(1, 1, l, m - 1, j, m).doit())     # c+1
    c0 = float(CG(1, 0, l, m, j, m).doit())         # c0
    cm = float(CG(1, -1, l, m + 1, j, m).doit())    
    0# c-1

    # We prepare an array with the M + mu values (that is M + 1, M, M - 1).
    # It takes value 0 for the cases that the correspononding Clebsch Gordan
    # coefficient is null for mysterious reasons (in that case the returned
    # A_{p, 0, m} value is zero).
    M_mu = np.array([int(cp != 0) * (m - 1), int(c0 != 0) * m,
     int(cm != 0) * (m + 1)])

    # We evaluate the spherical harmonic function note that
    # sph_harm is in mathematic notation??!
    # The (-1)^M fix the spherical harmonic Y function criteria. Al final el (-1)^M estaba dentro del polinomio de Legendre de la función de sph_harm y no hacía falta ponerlo.
    # Y = (-1) ** np.abs(M) * sph_harm(M, l, theta, phi)
    # Phi is the azimuthal angle and theta is the polar angle.
    Y = sph_harm(M_mu, l, phi, theta)

    # We multiplicate each spherical harmonic for the corresponding Clebsch
    # Gordan coefficient
    Ap = cp * Y[..., 0]
    A0 = c0 * Y[..., 1]
    Am = cm * Y[..., 2]

    return Ap, A0, Am

def eim(l, m, p, r, phi, theta):
    """
    Returns the combination of vector spherical harmonic that describes a
    field with well defined helicity p. Here we consider the radial dependence.
    Computes A(m) + i p A(e) at equation (4.5) of Zambrana thesis, where
    A(m) is defined at (7.26) Rose's or (9) N. Tischler's paper.
    A(e) is define at (7.32b) Rose's.
    """
    # Radial spherical bessel functions values. + 1/2 comes from spherical,
    # and + 1 and - 1 for considering all the cases
    b = np.sqrt(np.pi / 2 / (r * np.ones((1, 3)))) *\
     jv(np.array([l + 1/2, l + 1 + 1/2, l - 1 + 1/2]), r)
    b = b.T     # transpose

    # Load vector spherical harmonics with respective bessel component,
    # here we calculate TLLM, TLL+1M, TLL-1M at N. Tischler's / Rose's notation
    Ap1, A01, Am1 = vsh(l, l, m, phi, theta)
    b1 = b[0]
    Ap2, A02, Am2 = vsh(l, l + 1, m, phi, theta)
    b2 = b[1]
    Ap3, A03, Am3 = vsh(l, l - 1, m, phi, theta)
    b3 = b[2]
 
    # Combination (b1 * A1 = A(m), same as M of Jackson) the rest is ip*A(e) (A(e)=-N of Jackson)
    # In this case the relation between A(m) and A(e): A(e)=-i/k curl A(m) (As in Rose)
    Ap = (-b1 * Ap1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Ap2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Ap3) / np.sqrt(2)
    A0 = (-b1 * A01 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * A02 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * A03) / np.sqrt(2)
    Am = (-b1 * Am1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Am2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Am3) / np.sqrt(2)

    return Ap, A0, Am


def eimSC(l, m, p, r, phi, theta):
    """
    Same as eim but it uses hankel functions (z3) instead of bessel (z1)
    as it is no longer required to converge at the origin
    """
    # Radial spherical bessel functions values. + 1/2 comes from spherical,
    # and + 1 and - 1 for considering all the cases
    b = np.sqrt(np.pi / 2 / (r * np.ones((1, 3)))) *\
     (jv(np.array([l + 1/2, l + 1 + 1/2, l - 1 + 1/2]), r) +
     1j * yv(np.array([l + 1/2, l + 1 + 1/2, l - 1 + 1/2]), r))
    b = b.T     # transpose

    # Load vector spherical harmonics with respective bessel component,
    # here we calculate TLLM, TLL+1M, TLL-1M at N. Tischler's / Rose's notation
    Ap1, A01, Am1 = vsh(l, l, m, phi, theta)
    b1 = b[0]
    Ap2, A02, Am2 = vsh(l, l + 1, m, phi, theta)
    b2 = b[1]
    Ap3, A03, Am3 = vsh(l, l - 1, m, phi, theta)
    b3 = b[2]

    # Combination (b1 * A1 = A(m)) the rest is A(e)
    Ap = (-b1 * Ap1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Ap2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Ap3) / np.sqrt(2)
    A0 = (-b1 * A01 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * A02 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * A03) / np.sqrt(2)
    Am = (-b1 * Am1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Am2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Am3) / np.sqrt(2)

    return Ap, A0, Am


def XeimFF(l, m, p, kr, phi, theta):
    """
    Same as eim but we are using far field expresions for the bessel functions.
    """
    # Radial spherical bessel functions values for asimptotic limit of kr.
    b = (-1j) ** (np.array([l + 1, l + 2, l])) * np.exp(1j * kr) / (2 * kr) +\
     (1j) ** (np.array([l + 1, l + 2, l])) * np.exp(-1j * kr) / (2 * kr)
    b = b.T     # transpose

    # Load vector spherical harmonics with respective bessel component,
    # here we calculate TLLM, TLL+1M, TLL-1M at N. Tischler's / Rose's notation
    Ap1, A01, Am1 = vsh(l, l, m, phi, theta)
    b1 = b[0]
    Ap2, A02, Am2 = vsh(l, l + 1, m, phi, theta)
    b2 = b[1]
    Ap3, A03, Am3 = vsh(l, l - 1, m, phi, theta)
    b3 = b[2]

    # Combination (-b1 * A1 = A(m), same as M of Jackson) the rest is ip*A(e) (A(e)=-N of Jackson)
    # In this case the relation between A(m) and A(e): A(e)=-i/k curl A(m) (As in Rose)
    Ap = (-b1 * Ap1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Ap2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Ap3) / np.sqrt(2)
    A0 = (-b1 * A01 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * A02 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * A03) / np.sqrt(2)
    Am = (-b1 * Am1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Am2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Am3) / np.sqrt(2)

    return Ap, A0, Am


def XeimSCFF(l, m, p, kr, phi, theta):
    """
    Same as XeimFF but it uses hankel functions (z3) instead of bessel (z1)
    as it is no longer required to converge at the origin
    """
    # Radial spherical bessel functions values for asimptotic limit of kr.
    b = (-1j) ** (np.array([l + 1, l + 2, l])) * np.exp(1j * kr) / (kr)
    b = b.T     # transpose 1 and - 1 for considering all the cases

    # Load vector spherical harmonics with respective bessel component,
    # here we calculate TLLM, TLL+1M, TLL-1M at N. Tischler's / Rose's notation
    Ap1, A01, Am1 = vsh(l, l, m, phi, theta)
    b1 = b[0]
    Ap2, A02, Am2 = vsh(l, l + 1, m, phi, theta)
    b2 = b[1]
    Ap3, A03, Am3 = vsh(l, l - 1, m, phi, theta)
    b3 = b[2]

    # Combination (-b1 * A1 = A(m)) the rest is A(e)
    Ap = (-b1 * Ap1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Ap2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Ap3) / np.sqrt(2)
    A0 = (-b1 * A01 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * A02 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * A03) / np.sqrt(2)
    Am = (-b1 * Am1 - 1j * p * np.sqrt(l / (2 * l + 1)) * b2 * Am2 +\
     1j * p * np.sqrt((l + 1)/(2 * l + 1)) * b3 * Am3) / np.sqrt(2)

    return Ap, A0, Am
