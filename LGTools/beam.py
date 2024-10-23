# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import genlaguerre
from math import factorial

def calc_w0(z0, wavel):
    """
    Calculates w0 beamwaist from the z0 depth of focus
    """
    return np.sqrt(wavel * z0 / np.pi)


def calc_z0(w0, wavel):
    """
    Calculates the depth of focus from the w0 beamwaist
    """
    return w0 ** 2 * np.pi / wavel


def calc_w(z, w0, z0):
    """
    Calculates the beamwaste evolution with the distance
    """
    return w0 * np.sqrt(1 + (z / z0) ** 2)


def R(z, z0):
    """
    Calculates the radius of curvature
    """
    return z * (1 + (z0 / z) ** 2)


def psi(z, z0, l, q):
    """
    Calculates the Couy phase of the beam
    """
    N = np.abs(l) + 2 * q

    return np.arctan(z / z0) * (N + 1)

def LG(r, z, l, q, w0, wavel, phi):
    """
    Calculates the Field distribution for a Laguerre-Gaussian beam.
    """
    k = 2 * np.pi / wavel

    z0 = calc_z0(w0, wavel)
    wz = calc_w(z, w0, z0)
    if z == 0:
        Rz = 1
    else:
        Rz = R(z, z0)
    psiz = psi(z, z0, l, q)

    Nlq = np.sqrt(2 * factorial(q) /
     (np.pi * factorial(q + np.abs(l))))

    LGql = genlaguerre(q, np.abs(l))

    Exp_Rz = 1 if z == 0 else np.exp(-1j * k * r ** 2 / 2 / Rz)

    u = Nlq / wz * (r * np.sqrt(2) / wz) ** np.abs(l) *\
     np.exp(- r ** 2 / wz ** 2) * LGql(2 * r ** 2 / wz ** 2) *\
     Exp_Rz *\
     np.exp(+1j * l * phi) * np.exp(1j * k * z) *\
     np.exp(1j * psiz)

    return u

def LG_zambrana(q, l, r, w):
    l = np.abs(l)
    LGql = genlaguerre(q, np.abs(l))
    u = LGql(2 * r ** 2 / w ** 2)
    b = factorial(l + q) / factorial(q) / factorial(l)
    LG = r ** l * u * np.exp(- r ** 2 / w ** 2) * np.sqrt(2 ** (l + 1) /
     (np.pi * b * factorial(l))) / (w ** (l + 1))
    return LG

def w_0_size(l, q, f, n2, NA, cutoff):
    """
    Calculates the beam size at the entrance of the focusin lens
    for a specific filling factor (cutoff).
    """
    
    resw = 2500
    th_max = np.arcsin(NA / n2)
    th = np.linspace(0, th_max, 4000)
    D = f * NA * 2 / n2
    w = np.linspace(10e-6, D, resw)
    Integral = np.zeros(resw, dtype=complex)

    for i in range(resw):
        gn = LG_zambrana(q, l, f * np.sin(th), w[i])
        Integral[i] = np.trapz(f ** 2 * 2 * np.pi * np.sin(th) *
        np.cos(th) * (gn ** 2), th)
    index = np.where(np.abs(Integral - cutoff) == np.abs(Integral - cutoff).min())[0]
    w0 = w[index]
    return w0[0]
