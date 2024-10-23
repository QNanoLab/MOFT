# -*- coding: utf-8 -*-
"""
COORDINATES CONVERSION TOOLS -v2

Alvaro
06/04/2018
"""

# Import common modules:
import numpy as np
from scipy.linalg import expm


#                   ........................................
def CartVecConversion(Vsph, Possph):
    """
    Spherical vector to cartesian converter
    .........
    Requires:
    *) Vsph = [Vr, Vp, Vt]: Vector r, phi (polar), and theta (azimuthal)
     components
    *) Possph = [r, p, t]: r, phi, and theta coordinates for each (Vr, Vp, Vt);
     i.e.: (Vr, Vp, Vt) <-> (r, p, t)
    .........
    Returns:
    np.array([Vx, Vy, Vz])
    """

    # Assigning values:
    Vr, Vp, Vt = Vsph
    r, p, t = Possph

    # Setting cartesian vector components:
    Vx = np.sin(t) * np.cos(p) * Vr + np.cos(t) * np.cos(p) * Vt -\
         np.sin(p) * Vp
    Vy = np.sin(t) * np.sin(p) * Vr + np.cos(t) * np.sin(p) * Vt +\
         np.cos(p) * Vp
    Vz = np.cos(t) * Vr - np.sin(t) * Vt

    return np.array([Vx, Vy, Vz])

#                   ........................................



#                   ........................................
def SphVecConversion(Vcart, Possph):
    """
    Cartesian vector to spherical converter
    .........
    Requires:
    *) Vcart = [Vx, Vy, Vz]: Vector in cartesian coordinates
    *) Possph = [r, p, t]: r, phi, and theta coordinates for each (Vx, Vy, Vz);
     i.e.: (Vr, Vp, Vt) <-> (r, p, t)
    .........
    Returns:
    np.array([Vr, Vp, Vt])
    """

    # Assigining values
    Vx, Vy, Vz = Vcart
    r, p, t = Possph

    # Setting spherical vector components:
    Vr = np.sin(t) * np.cos(p) * Vx + np.sin(t) * np.sin(p) * Vy +\
         np.cos(t) * Vz
    Vt = np.cos(t) * np.cos(p) * Vx + np.cos(t) * np.sin(p) * Vy -\
         np.sin(t) * Vz
    Vp = -np.sin(p) * Vx + np.cos(p) * Vy

    return np.array([Vr, Vp, Vt])

#                   ........................................



#                   ........................................
def cart2sph(cart):
    """
    Cartesian coorindates to spherical converter
    .........
    Requires:
    *) cart = [x, y, z]: Cartesian coordinates
    .........
    Returns:
    np.array([r, phi, theta])
    """

    # Assigining values:
    x, y, z = cart

    # Setting spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return np.array([r, phi, theta])

#                   ........................................



#                   ........................................
def sph2cart(sph):
    """
    Spherical coorindates to cartesian converter
    .........
    Requires:
    *) sph = [r, phi, theta]: Spherical coordinates
    .........
    Returns:
    np.array([x, y, z])
    """

    # Assigining values:
    r, p, t = sph

    # Setting cartesian coordinates:
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)

    return np.array([x, y, z])

#                   ........................................



#                   ........................................
def Rodrigues(v, u):
    """
    Implements the Rodrigues rotation matrix between two vectors.
    This matrix is such that if R = Rodrigues(v, u), then u . R = v.
    And more interesting, if vu is a vector placed at u's possition, then
    vv = vu . R will be a vector placed at v possition pointing at the same
    direction as vu with vu's norm.
    .........
    Requieres:
    *) u & v: UNITARY! vectors to study (commonly origin to possition)
    .........
    Returns:
    R, the Rodrigues rotation matrix
    """
    # Normal tangential vector
    n = np.cross(u, v) / np.linalg.norm(np.cross(u, v))
    # To matrix
    n_x = np.cross(np.eye(3), n)

    # Angle between vectors
    a = np.arccos(np.dot(u, v)/ np.linalg.norm(u) / np.linalg.norm(v))

    # Rodrigues formula
    R =  expm(a * n_x)

    return R

#                   ........................................
