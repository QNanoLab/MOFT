# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Pool
import itertools as it
from LGTools.vsh import eimSC, eim

def ON_AXIS_multipole_summation_loop_2_planes(l, C_l_off_1, C_l_off_m1, alpha_array, beta_array, alpha_int_array, beta_int_array, L_n_1, L_n_2, p, nr, n2, kr, phi, theta, kr2, phi2, theta2, resolution):
    """
    Calculates the summation in j and m_z of well-defined helicity multipoles, for two spatial planes in spherical coordinates, for a specific multipolar coefficients in ON-AXIS configuration.
    """

    if p == 0:
        m= L_n_1 + 1
        
        Ap,A0,Am = eim(l, m, 1, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_in_1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_in_1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_in_1_1 = A0

        Ap,A0,Am = eimSC(l, m, 1, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_1 = A0

        Ap,A0,Am = eim(l, m, 1, kr*nr/n2, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_1 = A0

        Ap,A0,Am = eimSC(l, m, -1, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_m1 = A0

        Ap,A0,Am = eim(l, m, -1, kr*nr/n2, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_m1 = A0
        
        Ap,A0,Am = eim(l, m, 1, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_in_1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_in_1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_in_1_12 = A0

        Ap,A0,Am = eimSC(l, m, 1, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_12 = A0

        Ap,A0,Am = eim(l, m, 1, kr2*nr/n2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_12 = A0

        Ap,A0,Am = eimSC(l, m, -1, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_m12 = A0

        Ap,A0,Am = eim(l, m, -1, kr2*nr/n2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_m12 = A0

        m= L_n_2 - 1
        
        Ap,A0,Am = eim(l, m, -1, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_in_m1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_in_m1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_in_m1_m1 = A0

        Ap,A0,Am = eimSC(l, m, -1, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_m1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_m1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_m1_m1 = A0

        Ap,A0,Am = eim(l, m, -1, kr*nr/n2, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_m1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_m1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_m1_m1 = A0

        Ap,A0,Am = eimSC(l, m, 1, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_m1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_m1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_m1_1 = A0

        Ap,A0,Am = eim(l, m, 1, kr*nr/n2, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_m1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_m1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_m1_1 = A0
        
        Ap,A0,Am = eim(l, m, -1, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_in_m1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_in_m1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_in_m1_m12 = A0

        Ap,A0,Am = eimSC(l, m, -1, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_m1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_m1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_m1_m12 = A0

        Ap,A0,Am = eim(l, m, -1, kr2*nr/n2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_m1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_m1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_m1_m12 = A0

        Ap,A0,Am = eimSC(l, m, 1, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_m1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_m1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_m1_12 = A0

        Ap,A0,Am = eim(l, m, 1, kr2*nr/n2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_m1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_m1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_m1_12 = A0
    
    else:
        m= L_n_1 + p
        
        Ap,A0,Am = eim(l, m, p, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_in_1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_in_1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_in_1_1 = A0

        Ap,A0,Am = eimSC(l, m, p, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_1 = A0

        Ap,A0,Am = eim(l, m, p, kr*nr/n2, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_1 = A0

        Ap,A0,Am = eimSC(l, m, -p, kr, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_m1 = A0

        Ap,A0,Am = eim(l, m, -p, kr*nr/n2, phi, theta)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_m1 = A0
        
        Ap,A0,Am = eim(l, m, p, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_in_1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_in_1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_in_1_12 = A0

        Ap,A0,Am = eimSC(l, m, p, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_12 = A0

        Ap,A0,Am = eim(l, m, p, kr2*nr/n2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_12 = A0

        Ap,A0,Am = eimSC(l, m, -p, kr2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_1_m12 = A0

        Ap,A0,Am = eim(l, m, -p, kr2*nr/n2, phi2, theta2)
        Ap = np.reshape(Ap, (resolution,resolution))
        A0 = np.reshape(A0, (resolution,resolution))
        Am = np.reshape(Am, (resolution,resolution))
        Ax_int_1_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
        Ay_int_1_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
        Az_int_1_m12 = A0

    if p == 0: 
        #LINEAR POLARIZATION
        E_in_x =  np.sqrt(1/2)*(C_l_off_1[l]*Ax_in_1_1+C_l_off_m1[l]*Ax_in_m1_m1)
        E_in_y =  np.sqrt(1/2)*(C_l_off_1[l]*Ay_in_1_1+C_l_off_m1[l]*Ay_in_m1_m1)
        E_in_z =  np.sqrt(1/2)*(C_l_off_1[l]*Az_in_1_1+C_l_off_m1[l]*Az_in_m1_m1)

        E_sc_x =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_array[l]*Ax_1_1+beta_array[l]*Ax_1_m1)+C_l_off_m1[l]*(alpha_array[l]*Ax_m1_m1+beta_array[l]*Ax_m1_1))
        E_sc_y =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_array[l]*Ay_1_1+beta_array[l]*Ay_1_m1)+C_l_off_m1[l]*(alpha_array[l]*Ay_m1_m1+beta_array[l]*Ay_m1_1))
        E_sc_z =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_array[l]*Az_1_1+beta_array[l]*Az_1_m1)+C_l_off_m1[l]*(alpha_array[l]*Az_m1_m1+beta_array[l]*Az_m1_1))

        E_int_x =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_int_array[l]*Ax_int_1_1+beta_int_array[l]*Ax_int_1_m1)+C_l_off_m1[l]*(alpha_int_array[l]*Ax_int_m1_m1+beta_int_array[l]*Ax_int_m1_1))
        E_int_y =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_int_array[l]*Ay_int_1_1+beta_int_array[l]*Ay_int_1_m1)+C_l_off_m1[l]*(alpha_int_array[l]*Ay_int_m1_m1+beta_int_array[l]*Ay_int_m1_1))
        E_int_z =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_int_array[l]*Az_int_1_1+beta_int_array[l]*Az_int_1_m1)+C_l_off_m1[l]*(alpha_int_array[l]*Az_int_m1_m1+beta_int_array[l]*Az_int_m1_1))

        E_in_x2 =  np.sqrt(1/2)*(C_l_off_1[l]*Ax_in_1_12+C_l_off_m1[l]*Ax_in_m1_m12)
        E_in_y2 =  np.sqrt(1/2)*(C_l_off_1[l]*Ay_in_1_12+C_l_off_m1[l]*Ay_in_m1_m12)
        E_in_z2 =  np.sqrt(1/2)*(C_l_off_1[l]*Az_in_1_12+C_l_off_m1[l]*Az_in_m1_m12)

        E_sc_x2 =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_array[l]*Ax_1_12+beta_array[l]*Ax_1_m12)+C_l_off_m1[l]*(alpha_array[l]*Ax_m1_m12+beta_array[l]*Ax_m1_12))
        E_sc_y2 =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_array[l]*Ay_1_12+beta_array[l]*Ay_1_m12)+C_l_off_m1[l]*(alpha_array[l]*Ay_m1_m12+beta_array[l]*Ay_m1_12))
        E_sc_z2 =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_array[l]*Az_1_12+beta_array[l]*Az_1_m12)+C_l_off_m1[l]*(alpha_array[l]*Az_m1_m12+beta_array[l]*Az_m1_12))

        E_int_x2 =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_int_array[l]*Ax_int_1_12+beta_int_array[l]*Ax_int_1_m12)+C_l_off_m1[l]*(alpha_int_array[l]*Ax_int_m1_m12+beta_int_array[l]*Ax_int_m1_12))
        E_int_y2 =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_int_array[l]*Ay_int_1_12+beta_int_array[l]*Ay_int_1_m12)+C_l_off_m1[l]*(alpha_int_array[l]*Ay_int_m1_m12+beta_int_array[l]*Ay_int_m1_12))
        E_int_z2 =  np.sqrt(1/2)*(C_l_off_1[l]*(alpha_int_array[l]*Az_int_1_12+beta_int_array[l]*Az_int_1_m12)+C_l_off_m1[l]*(alpha_int_array[l]*Az_int_m1_m12+beta_int_array[l]*Az_int_m1_12))

    else:
        #CIRCULAR POLARIZATION
        E_in_x =  C_l_off_1[l]*Ax_in_1_1
        E_in_y =  C_l_off_1[l]*Ay_in_1_1
        E_in_z =  C_l_off_1[l]*Az_in_1_1
        
        E_sc_x =  C_l_off_1[l]*(alpha_array[l]*Ax_1_1+beta_array[l]*Ax_1_m1)
        E_sc_y =  C_l_off_1[l]*(alpha_array[l]*Ay_1_1+beta_array[l]*Ay_1_m1)
        E_sc_z =  C_l_off_1[l]*(alpha_array[l]*Az_1_1+beta_array[l]*Az_1_m1)
            
        E_int_x =  C_l_off_1[l]*(alpha_int_array[l]*Ax_int_1_1+beta_int_array[l]*Ax_int_1_m1)
        E_int_y =  C_l_off_1[l]*(alpha_int_array[l]*Ay_int_1_1+beta_int_array[l]*Ay_int_1_m1)
        E_int_z =  C_l_off_1[l]*(alpha_int_array[l]*Az_int_1_1+beta_int_array[l]*Az_int_1_m1)

        E_in_x2 =  C_l_off_1[l]*Ax_in_1_12
        E_in_y2 =  C_l_off_1[l]*Ay_in_1_12
        E_in_z2 =  C_l_off_1[l]*Az_in_1_12
        
        E_sc_x2 =  C_l_off_1[l]*(alpha_array[l]*Ax_1_12+beta_array[l]*Ax_1_m12)
        E_sc_y2 =  C_l_off_1[l]*(alpha_array[l]*Ay_1_12+beta_array[l]*Ay_1_m12)
        E_sc_z2 =  C_l_off_1[l]*(alpha_array[l]*Az_1_12+beta_array[l]*Az_1_m12)
            
        E_int_x2 =  C_l_off_1[l]*(alpha_int_array[l]*Ax_int_1_12+beta_int_array[l]*Ax_int_1_m12)
        E_int_y2 =  C_l_off_1[l]*(alpha_int_array[l]*Ay_int_1_12+beta_int_array[l]*Ay_int_1_m12)
        E_int_z2 =  C_l_off_1[l]*(alpha_int_array[l]*Az_int_1_12+beta_int_array[l]*Az_int_1_m12)
        
    return E_in_x, E_in_y, E_in_z, E_sc_x, E_sc_y, E_sc_z, E_int_x, E_int_y, E_int_z, E_in_x2, E_in_y2, E_in_z2, E_sc_x2, E_sc_y2, E_sc_z2, E_int_x2, E_int_y2, E_int_z2
   
def ON_AXIS_multipole_summation_loop_2_planes_wrapper(args):
    return ON_AXIS_multipole_summation_loop_2_planes(*args)

def ON_AXIS_multipole_summation_2_planes(C_l_off_1, C_l_off_m1, alpha_array, beta_array, alpha_int_array, beta_int_array, L_n_1, L_n_2, p, nr, n2, kr, phi, theta, kr2, phi2, theta2, resolution, l_max, ncores):

    E_in_x = np.zeros((resolution,resolution), dtype=complex)
    E_in_y = np.zeros((resolution,resolution), dtype=complex)
    E_in_z = np.zeros((resolution,resolution), dtype=complex)

    E_sc_x = np.zeros((resolution,resolution), dtype=complex)
    E_sc_y = np.zeros((resolution,resolution), dtype=complex)
    E_sc_z = np.zeros((resolution,resolution), dtype=complex)

    E_int_x = np.zeros((resolution,resolution), dtype=complex)
    E_int_y = np.zeros((resolution,resolution), dtype=complex)
    E_int_z = np.zeros((resolution,resolution), dtype=complex)

    E_in_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_in_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_in_z2 = np.zeros((resolution,resolution), dtype=complex)

    E_sc_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_sc_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_sc_z2 = np.zeros((resolution,resolution), dtype=complex)

    E_int_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_int_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_int_z2 = np.zeros((resolution,resolution), dtype=complex)

    pool = Pool(processes=ncores)

    j = np.arange(1, l_max+1, 1)

    arg_list = list(zip(
        j, it.repeat(C_l_off_1), it.repeat(C_l_off_m1), it.repeat(alpha_array), it.repeat(beta_array), it.repeat(alpha_int_array), it.repeat(beta_int_array),
        it.repeat(L_n_1), it.repeat(L_n_2), it.repeat(p), it.repeat(nr), it.repeat(n2), it.repeat(kr), it.repeat(phi), it.repeat(theta), it.repeat(kr2), it.repeat(phi2), it.repeat(theta2), it.repeat(resolution)
    ))
    for o, values in enumerate(pool.imap(ON_AXIS_multipole_summation_loop_2_planes_wrapper, arg_list), 1):
            E_in_xl,E_in_yl,E_in_zl,E_sc_xl,E_sc_yl,E_sc_zl,E_int_xl,E_int_yl,E_int_zl,E_in_xl2,E_in_yl2,E_in_zl2,E_sc_xl2,E_sc_yl2,E_sc_zl2,E_int_xl2,E_int_yl2,E_int_zl2 = values
            
            E_in_x =  E_in_x + E_in_xl
            E_in_y =  E_in_y + E_in_yl
            E_in_z =  E_in_z + E_in_zl
            
            E_sc_x =  E_sc_x + E_sc_xl
            E_sc_y =  E_sc_y + E_sc_yl
            E_sc_z =  E_sc_z + E_sc_zl

            E_int_x =  E_int_x + E_int_xl
            E_int_y =  E_int_y + E_int_yl
            E_int_z =  E_int_z + E_int_zl

            E_in_x2 =  E_in_x2 + E_in_xl2
            E_in_y2 =  E_in_y2 + E_in_yl2
            E_in_z2 =  E_in_z2 + E_in_zl2
            
            E_sc_x2 =  E_sc_x2 + E_sc_xl2
            E_sc_y2 =  E_sc_y2 + E_sc_yl2
            E_sc_z2 =  E_sc_z2 + E_sc_zl2

            E_int_x2 =  E_int_x2 + E_int_xl2
            E_int_y2 =  E_int_y2 + E_int_yl2
            E_int_z2 =  E_int_z2 + E_int_zl2
    pool.close()
    
    ###  INTESISTY OF ELECTRIC FIELDS   ###
    E_in_XZ=np.abs(E_in_x)**2+np.abs(E_in_y)**2+np.abs(E_in_z)**2

    E_sc_XZ=np.abs(E_sc_x)**2+np.abs(E_sc_y)**2+np.abs(E_sc_z)**2

    E_tot_XZ=np.abs(E_sc_x+E_in_x)**2+np.abs(E_sc_y+E_in_y)**2+np.abs(E_sc_z+E_in_z)**2

    E_int_XZ=np.abs(E_int_x)**2+np.abs(E_int_y)**2+np.abs(E_int_z)**2

    E_in_XY=np.abs(E_in_x2)**2+np.abs(E_in_y2)**2+np.abs(E_in_z2)**2

    E_sc_XY=np.abs(E_sc_x2)**2+np.abs(E_sc_y2)**2+np.abs(E_sc_z2)**2

    E_tot_XY=np.abs(E_sc_x2+E_in_x2)**2+np.abs(E_sc_y2+E_in_y2)**2+np.abs(E_sc_z2+E_in_z2)**2

    E_int_XY=np.abs(E_int_x2)**2+np.abs(E_int_y2)**2+np.abs(E_int_z2)**2
    
    return E_in_XZ, E_sc_XZ, E_tot_XZ, E_int_XZ, E_in_XY, E_sc_XY, E_tot_XY, E_int_XY

def OFF_AXIS_multipole_summation_loop_2_planes_Mz(l, C_l_off_1, C_l_off_m1, alpha_array, beta_array, alpha_int_array, beta_int_array, p, nr, n2, kr, phi, theta, kr2, phi2, theta2, resolution, l_max):
    """
    Calculates the summation in j and m_z of well-defined helicity multipoles, for two spatial planes in spherical coordinates, for a specific multipolar coefficients in OFF-AXIS configuration.
    """

    E_in_x = np.zeros((resolution,resolution), dtype=complex)
    E_in_y = np.zeros((resolution,resolution), dtype=complex)
    E_in_z = np.zeros((resolution,resolution), dtype=complex)

    E_sc_x = np.zeros((resolution,resolution), dtype=complex)
    E_sc_y = np.zeros((resolution,resolution), dtype=complex)
    E_sc_z = np.zeros((resolution,resolution), dtype=complex)

    E_int_x = np.zeros((resolution,resolution), dtype=complex)
    E_int_y = np.zeros((resolution,resolution), dtype=complex)
    E_int_z = np.zeros((resolution,resolution), dtype=complex)

    E_in_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_in_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_in_z2 = np.zeros((resolution,resolution), dtype=complex)

    E_sc_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_sc_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_sc_z2 = np.zeros((resolution,resolution), dtype=complex)

    E_int_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_int_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_int_z2 = np.zeros((resolution,resolution), dtype=complex)

    for m in range (l_max-l, l_max+l+1):
        m_0 = -l_max + m
            
        if p == 0:            
            Ap,A0,Am = eim(l, m_0, 1, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_in_1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_in_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_in_1 = A0

            Ap,A0,Am = eimSC(l, m_0, 1, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_1 = A0

            Ap,A0,Am = eim(l, m_0, 1, kr*nr/n2, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_1 = A0

            Ap,A0,Am = eim(l, m_0, -1, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_in_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_in_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_in_m1 = A0

            Ap,A0,Am = eimSC(l, m_0, -1, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_m1 = A0

            Ap,A0,Am = eim(l, m_0, -1, kr*nr/n2, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_m1 = A0
            
            Ap,A0,Am = eim(l, m_0, 1, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_in_12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_in_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_in_12 = A0

            Ap,A0,Am = eimSC(l, m_0, 1, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_12 = A0

            Ap,A0,Am = eim(l, m_0, 1, kr2*nr/n2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_12 = A0

            Ap,A0,Am = eim(l, m_0, -1, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_in_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_in_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_in_m12 = A0

            Ap,A0,Am = eimSC(l, m_0, -1, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_m12 = A0

            Ap,A0,Am = eim(l, m_0, -1, kr2*nr/n2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_m12 = A0
        
        else:
            Ap,A0,Am = eim(l, m_0, p, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_in_1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_in_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_in_1 = A0

            Ap,A0,Am = eimSC(l, m_0, p, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_1 = A0

            Ap,A0,Am = eim(l, m_0, p, kr*nr/n2, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_1 = A0

            Ap,A0,Am = eimSC(l, m_0, -p, kr, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_m1 = A0

            Ap,A0,Am = eim(l, m_0, -p, kr*nr/n2, phi, theta)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_m1 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_m1 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_m1 = A0
            
            Ap,A0,Am = eim(l, m_0, p, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_in_12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_in_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_in_12 = A0

            Ap,A0,Am = eimSC(l, m_0, p, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_12 = A0

            Ap,A0,Am = eim(l, m_0, p, kr2*nr/n2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_12 = A0

            Ap,A0,Am = eimSC(l, m_0, -p, kr2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_m12 = A0

            Ap,A0,Am = eim(l, m_0, -p, kr2*nr/n2, phi2, theta2)
            Ap = np.reshape(Ap, (resolution,resolution))
            A0 = np.reshape(A0, (resolution,resolution))
            Am = np.reshape(Am, (resolution,resolution))
            Ax_int_m12 = -(1/(np.sqrt(2)))*(Ap-Am)
            Ay_int_m12 = -(1j/(np.sqrt(2)))*(Ap+Am)
            Az_int_m12 = A0

        if p == 0: 
            #LINEAR POLARIZATION
            E_in_x = E_in_x + np.sqrt(1/2)*(C_l_off_1[l,m]*Ax_in_1+C_l_off_m1[l,m]*Ax_in_m1)
            E_in_y = E_in_y + np.sqrt(1/2)*(C_l_off_1[l,m]*Ay_in_1+C_l_off_m1[l,m]*Ay_in_m1)
            E_in_z = E_in_z + np.sqrt(1/2)*(C_l_off_1[l,m]*Az_in_1+C_l_off_m1[l,m]*Az_in_m1)

            E_sc_x = E_sc_x + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_array[l]*Ax_1+beta_array[l]*Ax_m1)+C_l_off_m1[l,m]*(alpha_array[l]*Ax_m1+beta_array[l]*Ax_1))
            E_sc_y = E_sc_y + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_array[l]*Ay_1+beta_array[l]*Ay_m1)+C_l_off_m1[l,m]*(alpha_array[l]*Ay_m1+beta_array[l]*Ay_1))
            E_sc_z = E_sc_z + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_array[l]*Az_1+beta_array[l]*Az_m1)+C_l_off_m1[l,m]*(alpha_array[l]*Az_m1+beta_array[l]*Az_1))

            E_int_x = E_int_x + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_int_array[l]*Ax_int_1+beta_int_array[l]*Ax_int_m1)+C_l_off_m1[l,m]*(alpha_int_array[l]*Ax_int_m1+beta_int_array[l]*Ax_int_1))
            E_int_y = E_int_y + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_int_array[l]*Ay_int_1+beta_int_array[l]*Ay_int_m1)+C_l_off_m1[l,m]*(alpha_int_array[l]*Ay_int_m1+beta_int_array[l]*Ay_int_1))
            E_int_z = E_int_z + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_int_array[l]*Az_int_1+beta_int_array[l]*Az_int_m1)+C_l_off_m1[l,m]*(alpha_int_array[l]*Az_int_m1+beta_int_array[l]*Az_int_1))

            E_in_x2 = E_in_x2 + np.sqrt(1/2)*(C_l_off_1[l,m]*Ax_in_12+C_l_off_m1[l,m]*Ax_in_m12)
            E_in_y2 = E_in_y2 + np.sqrt(1/2)*(C_l_off_1[l,m]*Ay_in_12+C_l_off_m1[l,m]*Ay_in_m12)
            E_in_z2 = E_in_z2 + np.sqrt(1/2)*(C_l_off_1[l,m]*Az_in_12+C_l_off_m1[l,m]*Az_in_m12)

            E_sc_x2 = E_sc_x2 + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_array[l]*Ax_12+beta_array[l]*Ax_m12)+C_l_off_m1[l,m]*(alpha_array[l]*Ax_m12+beta_array[l]*Ax_12))
            E_sc_y2 = E_sc_y2 + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_array[l]*Ay_12+beta_array[l]*Ay_m12)+C_l_off_m1[l,m]*(alpha_array[l]*Ay_m12+beta_array[l]*Ay_12))
            E_sc_z2 = E_sc_z2 + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_array[l]*Az_12+beta_array[l]*Az_m12)+C_l_off_m1[l,m]*(alpha_array[l]*Az_m12+beta_array[l]*Az_12))

            E_int_x2 = E_int_x2 + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_int_array[l]*Ax_int_12+beta_int_array[l]*Ax_int_m12)+C_l_off_m1[l,m]*(alpha_int_array[l]*Ax_int_m12+beta_int_array[l]*Ax_int_12))
            E_int_y2 = E_int_y2 + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_int_array[l]*Ay_int_12+beta_int_array[l]*Ay_int_m12)+C_l_off_m1[l,m]*(alpha_int_array[l]*Ay_int_m12+beta_int_array[l]*Ay_int_12))
            E_int_z2 = E_int_z2 + np.sqrt(1/2)*(C_l_off_1[l,m]*(alpha_int_array[l]*Az_int_12+beta_int_array[l]*Az_int_m12)+C_l_off_m1[l,m]*(alpha_int_array[l]*Az_int_m12+beta_int_array[l]*Az_int_12))

        else:
            #CIRCULAR POLARIZATION
            E_in_x = E_in_x + C_l_off_1[l,m]*Ax_in_1
            E_in_y = E_in_y + C_l_off_1[l,m]*Ay_in_1
            E_in_z = E_in_z + C_l_off_1[l,m]*Az_in_1
            
            E_sc_x = E_sc_x + C_l_off_1[l,m]*(alpha_array[l]*Ax_1+beta_array[l]*Ax_m1)
            E_sc_y = E_sc_y + C_l_off_1[l,m]*(alpha_array[l]*Ay_1+beta_array[l]*Ay_m1)
            E_sc_z = E_sc_z + C_l_off_1[l,m]*(alpha_array[l]*Az_1+beta_array[l]*Az_m1)
                
            E_int_x = E_int_x + C_l_off_1[l,m]*(alpha_int_array[l]*Ax_int_1+beta_int_array[l]*Ax_int_m1)
            E_int_y = E_int_y + C_l_off_1[l,m]*(alpha_int_array[l]*Ay_int_1+beta_int_array[l]*Ay_int_m1)
            E_int_z = E_int_z + C_l_off_1[l,m]*(alpha_int_array[l]*Az_int_1+beta_int_array[l]*Az_int_m1)

            E_in_x2 = E_in_x2 + C_l_off_1[l,m]*Ax_in_12
            E_in_y2 = E_in_y2 + C_l_off_1[l,m]*Ay_in_12
            E_in_z2 = E_in_z2 + C_l_off_1[l,m]*Az_in_12
            
            E_sc_x2 = E_sc_x2 + C_l_off_1[l,m]*(alpha_array[l]*Ax_12+beta_array[l]*Ax_m12)
            E_sc_y2 = E_sc_y2 + C_l_off_1[l,m]*(alpha_array[l]*Ay_12+beta_array[l]*Ay_m12)
            E_sc_z2 = E_sc_z2 + C_l_off_1[l,m]*(alpha_array[l]*Az_12+beta_array[l]*Az_m12)
                
            E_int_x2 = E_int_x2 + C_l_off_1[l,m]*(alpha_int_array[l]*Ax_int_12+beta_int_array[l]*Ax_int_m12)
            E_int_y2 = E_int_y2 + C_l_off_1[l,m]*(alpha_int_array[l]*Ay_int_12+beta_int_array[l]*Ay_int_m12)
            E_int_z2 = E_int_z2 + C_l_off_1[l,m]*(alpha_int_array[l]*Az_int_12+beta_int_array[l]*Az_int_m12)
            
    return E_in_x, E_in_y, E_in_z, E_sc_x, E_sc_y, E_sc_z, E_int_x, E_int_y, E_int_z, E_in_x2, E_in_y2, E_in_z2, E_sc_x2, E_sc_y2, E_sc_z2, E_int_x2, E_int_y2, E_int_z2
   
def OFF_AXIS_multipole_summation_loop_2_planes_Mz_wrapper(args):
    return OFF_AXIS_multipole_summation_loop_2_planes_Mz(*args)

def OFF_AXIS_multipole_summation_loop_2_planes(C_l_off_1, C_l_off_m1, alpha_array, beta_array, alpha_int_array, beta_int_array, p, nr, n2, kr, phi, theta, kr2, phi2, theta2, resolution, l_max, ncores):

    E_in_x = np.zeros((resolution,resolution), dtype=complex)
    E_in_y = np.zeros((resolution,resolution), dtype=complex)
    E_in_z = np.zeros((resolution,resolution), dtype=complex)

    E_sc_x = np.zeros((resolution,resolution), dtype=complex)
    E_sc_y = np.zeros((resolution,resolution), dtype=complex)
    E_sc_z = np.zeros((resolution,resolution), dtype=complex)

    E_int_x = np.zeros((resolution,resolution), dtype=complex)
    E_int_y = np.zeros((resolution,resolution), dtype=complex)
    E_int_z = np.zeros((resolution,resolution), dtype=complex)

    E_in_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_in_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_in_z2 = np.zeros((resolution,resolution), dtype=complex)

    E_sc_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_sc_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_sc_z2 = np.zeros((resolution,resolution), dtype=complex)

    E_int_x2 = np.zeros((resolution,resolution), dtype=complex)
    E_int_y2 = np.zeros((resolution,resolution), dtype=complex)
    E_int_z2 = np.zeros((resolution,resolution), dtype=complex)

    pool = Pool(processes=ncores)

    j = np.arange(1, l_max+1, 1)

    arg_list = list(zip(
        j, it.repeat(C_l_off_1),it.repeat(C_l_off_m1), it.repeat(alpha_array), it.repeat(beta_array), it.repeat(alpha_int_array), it.repeat(beta_int_array),
        it.repeat(p), it.repeat(nr), it.repeat(n2), it.repeat(kr), it.repeat(phi), it.repeat(theta), it.repeat(kr2), it.repeat(phi2), it.repeat(theta2), it.repeat(resolution), it.repeat(l_max)
    ))
    for o, values in enumerate(pool.imap(OFF_AXIS_multipole_summation_loop_2_planes_Mz_wrapper, arg_list), 1):
            E_in_xl,E_in_yl,E_in_zl,E_sc_xl,E_sc_yl,E_sc_zl,E_int_xl,E_int_yl,E_int_zl,E_in_xl2,E_in_yl2,E_in_zl2,E_sc_xl2,E_sc_yl2,E_sc_zl2,E_int_xl2,E_int_yl2,E_int_zl2 = values
            
            E_in_x =  E_in_x + E_in_xl
            E_in_y =  E_in_y + E_in_yl
            E_in_z =  E_in_z + E_in_zl
            
            E_sc_x =  E_sc_x + E_sc_xl
            E_sc_y =  E_sc_y + E_sc_yl
            E_sc_z =  E_sc_z + E_sc_zl

            E_int_x =  E_int_x + E_int_xl
            E_int_y =  E_int_y + E_int_yl
            E_int_z =  E_int_z + E_int_zl

            E_in_x2 =  E_in_x2 + E_in_xl2
            E_in_y2 =  E_in_y2 + E_in_yl2
            E_in_z2 =  E_in_z2 + E_in_zl2
            
            E_sc_x2 =  E_sc_x2 + E_sc_xl2
            E_sc_y2 =  E_sc_y2 + E_sc_yl2
            E_sc_z2 =  E_sc_z2 + E_sc_zl2

            E_int_x2 =  E_int_x2 + E_int_xl2
            E_int_y2 =  E_int_y2 + E_int_yl2
            E_int_z2 =  E_int_z2 + E_int_zl2
    pool.close()

    ###  INTESISTY OF ELECTRIC FIELDS   ###
    E_in_XZ=np.abs(E_in_x)**2+np.abs(E_in_y)**2+np.abs(E_in_z)**2
    E_sc_XZ=np.abs(E_sc_x)**2+np.abs(E_sc_y)**2+np.abs(E_sc_z)**2
    E_tot_XZ=np.abs(E_sc_x+E_in_x)**2+np.abs(E_sc_y+E_in_y)**2+np.abs(E_sc_z+E_in_z)**2
    E_int_XZ=np.abs(E_int_x)**2+np.abs(E_int_y)**2+np.abs(E_int_z)**2
    
    E_in_XY=np.abs(E_in_x2)**2+np.abs(E_in_y2)**2+np.abs(E_in_z2)**2
    E_sc_XY=np.abs(E_sc_x2)**2+np.abs(E_sc_y2)**2+np.abs(E_sc_z2)**2
    E_tot_XY=np.abs(E_sc_x2+E_in_x2)**2+np.abs(E_sc_y2+E_in_y2)**2+np.abs(E_sc_z2+E_in_z2)**2
    E_int_XY=np.abs(E_int_x2)**2+np.abs(E_int_y2)**2+np.abs(E_int_z2)**2
    
    return E_in_XZ, E_sc_XZ, E_tot_XZ, E_int_XZ, E_in_XY, E_sc_XY, E_tot_XY, E_int_XY