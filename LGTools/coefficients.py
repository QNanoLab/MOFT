# -*- coding: utf-8 -*-
#%%
import numpy as np
from LGTools.beam import calc_z0, LG
from scipy.special import spherical_jn, spherical_yn, jacobi
import scipy.special as sci
import pickle
from pathlib import Path



try:
    DATA_PATH = Path(__file__).parents[1] / 'TRANSLATION_MATRICES'
    
    with open(DATA_PATH / 't_z.pkl', 'rb') as fp:
        Tz_pickle = pickle.load(fp)
    with open(DATA_PATH / 't_xy.pkl', 'rb') as fp:
        Txy_pickle = pickle.load(fp)
except FileNotFoundError:
    print("No precalc matrices!!!!!!!!")



def wigner_small_d_jab(l, m, p, t):
    """
    Computes the Wigner's small d-matrix
    """
    if np.isscalar(l):
        j=l
        k = min(j-m,j+m,j-p,j+p)
        
        if k == j+p:
            a = m-p; lmb= m-p
        elif k == j-p:
            a = p-m; lmb = 0
        elif k == j+m:
            a = p-m; lmb = 0
        elif k == j-m:
            a = m-p; lmb = m-p
        b = 2*(j-k) - a
        
        if k<0:
            d=0
        else:
            jab=jacobi(k,a,b)

            bncoeff = (-1)**lmb*np.sqrt(sci.binom(2*j-k,k+a)/sci.binom(k+b,b))
            d = bncoeff*np.sin(t/2)**a*np.cos(t/2)**b*jab(np.cos(t))
    else:
        d=np.zeros((len(l)), dtype=complex)
        for i in range(0, len(l)):
            j=l[i]
            k = min(j-m,j+m,j-p,j+p)
        
            if k == j+p:
                a = m-p; lmb= m-p
            elif k == j-p:
                a = p-m; lmb = 0
            elif k == j+m:
                a = p-m; lmb = 0
            elif k == j-m:
                a = m-p; lmb = m-p
            b = 2*(j-k) - a
            
            if k<0:
                d[i]=0
            else:
                jab=jacobi(k,a,b)

                bncoeff = (-1)**lmb*np.sqrt(sci.binom(2*j-k,k+a)/sci.binom(k+b,b))
                d[i] = bncoeff*np.sin(t/2)**a*np.cos(t/2)**b*jab(np.cos(t))
    return d

def coefficients_focus(n_max, l, q, p, f, k, n2, w0, z, NA, z0=None, exp_phase=True):
    """
    Computes C_l for a laguerre gauss field decomposition in vector shperical
    harmonics.
    D is the standard weight.
    f = focal
    n2 = medium refractive index
    w evaluated at z
    NA: numerical aperture
    """
    # Setting m contribuition:
    m = l + p  # + 1 para soluctionar indices?

    # calculate beam parameters:
    wavel = 2 * np.pi / (k)
    if z0 is None:
        z0 = calc_z0(w0, wavel)

    # Setting dimension of the array
    dim = (n_max+1)

    # Computing common amplitude D
    D = np.zeros(dim, dtype=object)

    for n in range(0, n_max+1):
        D[n] = (1j)**n * np.sqrt(2*n + 1)

    # Computing "also" common amplitude Cnm
    C = np.zeros(dim, dtype=object)
    
    if NA==0:
        C[1:] = np.ones(C[1:].size) * np.sqrt(2*np.pi)
    else:
        # integration linspace
        tkM = np.arcsin(NA/n2)     # tkM is the maximal half angle
        tk = np.linspace(0, tkM, 1000)

        if exp_phase:
            exTerm = np.exp(-1j * k * n2 * f)
        else:
            exTerm = 1
        Gn = LG(f * np.sin(tk), z, l, q, w0, wavel, 0) *\
        np.sqrt(np.cos(tk)) * f * exTerm

        for n in range(max(0, m), n_max + 1):

            dtk = wigner_small_d_jab(n, m, p, tk)
            
            coef = np.trapz(Gn * dtk * np.sin(tk), tk) * np.sqrt(np.pi)

            C[n] = complex(coef)

    return D, C

def Z_disp_TM_chain(kd, p):
    """
    Computes the dot product between the precalculated translation matrices for reaching a specific ON-AXIS kd displacement.
    """
    
    kd_list_z= np.array([0.1,0.01,0.001,0.0001,0.5,0.05,0.005,0.0005,0.25,0.025,0.0025,0.00025,0.75,0.075,0.0075,0.00075,1.0,2.5,5.0,7.5,10.0,-0.1,-0.01,-0.001,-0.0001,-0.5,-0.05,-0.005,-0.0005,-0.25,-0.025,-0.0025,-0.00025,-0.75,-0.075,-0.0075,-0.00075,-1.0,-2.5,-5.0,-7.5,-10.0])

    if np.abs(kd)<0.0001:
        kd=0
    precis=1
    first_it=0
    kd_sum_1=0
    while precis>0.0001:
        kd_select1=(kd_list_z+kd_sum_1)-kd
        kd_select2=np.zeros((len(kd_list_z)))
        if kd>0:
            for i in range(0,len(kd_list_z)):
                kd_select2[i]=1000
                if kd_select1[i]<0:
                    kd_select2[i]=kd_select1[i]
        else:
            for i in range(0,len(kd_list_z)):
                kd_select2[i]=1000
                if kd_select1[i]>0:
                    kd_select2[i]=kd_select1[i]
        kd_sum_2=kd_list_z[np.argmin(np.abs(kd_select2))]
        kd_sum_1=kd_sum_1+kd_sum_2
        precis=np.abs(kd-kd_sum_1)
        T_M_sum=Tz_pickle[p][str(kd_sum_2)]
        if first_it==0:
            T_M_tot=T_M_sum
            first_it=1
        else:
            T_M_tot=np.dot(T_M_tot,T_M_sum)
    
    return T_M_tot

def XY_disp_TM_chain(kd, p, kd_sum_1=0, first_it=0, T_M_tot_X=1):
    """
    Computes the dot product between the precalculated translation matrices for reaching a specific OFF-AXIS kd displacement.
    """

    kd_list= np.array([0.1,0.01,0.001,0.0001,0.5,0.05,0.005,0.0005,0.25,0.025,0.0025,0.00025,0.75,0.075,0.0075,0.00075,1.0,2.5,5.0,7.5,10.0,-0.1,-0.01,-0.001,-0.0001,-0.5,-0.05,-0.005,-0.0005,-0.25,-0.025,-0.0025,-0.00025,-0.75,-0.075,-0.0075,-0.00075,-1.0,-2.5,-5.0,-7.5,-10.0])
    precis=1
    while precis>0.0001:
        kd_select1=(kd_list+kd_sum_1)-kd
        kd_select2=np.zeros((len(kd_list)))
        for i in range(0,len(kd_list)):
            kd_select2[i]=1000
            if kd_select1[i]<0:
                kd_select2[i]=kd_select1[i]
        kd_sum_2=kd_list[np.argmin(np.abs(kd_select2))]
        kd_sum_1=kd_sum_1+kd_sum_2
        precis=kd-kd_sum_1
        T_M_sum_X=Txy_pickle[p][str(kd_sum_2)]
        if first_it==0:
            T_M_tot_X=T_M_sum_X
            first_it=1
        else:
            T_M_tot_X=np.dot(T_M_tot_X,T_M_sum_X)
    
    return T_M_tot_X, kd_sum_1, first_it

def coefficients_Z_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max):
    """
    Computes the multipolar coefficients from a translation matrix representing an ON-AXIS displacement and ON-FOCUS coefficients.
    """

    C_dist_array= np.append(np.append(np.arange(1,l_max+1,1),l_max),np.arange(l_max,0,-1))
    min_lim_C=np.sum(C_dist_array[:l_max+m_z_p])
    max_lim_C=np.sum(C_dist_array[:l_max+m_z_p+1])

    C_l_off_1_mz = np.dot(T_M_tot[:,min_lim_C:max_lim_C], C_l_1[max(np.abs(m_z_p),1):])
    C_l_off_1_2 = C_l_off_1_mz[min_lim_C:max_lim_C].T*D[max(np.abs(m_z_p),1):]
    C_l_off_1 = np.zeros((l_max+1),dtype=complex)
    C_l_off_1[max(np.abs(m_z_p),1):] = C_l_off_1_2
    
    return C_l_off_1

def coefficients_XY_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max):
    """
    Computes the multipolar coefficients from a translation matrix representing an OFF-AXIS displacement and ON-FOCUS coefficients.
    """

    C_dist_array= np.append(np.append(np.arange(1,l_max+1,1),l_max),np.arange(l_max,0,-1))
    min_lim_C=np.sum(C_dist_array[:l_max+m_z_p])
    max_lim_C=np.sum(C_dist_array[:l_max+m_z_p+1])

    C_l_off_1_mz = np.dot(T_M_tot[:,min_lim_C:max_lim_C], C_l_1[max(np.abs(m_z_p),1):])
    C_l_off_1 = np.zeros((l_max+1, 2*(l_max)+1), dtype=complex)

    for mz in range (0, 2*l_max+1):
        C_l_off_1[-C_dist_array[mz]:,mz] = C_l_off_1_mz[(np.sum(C_dist_array[:mz])):(np.sum(C_dist_array[:(mz+1)]))]
    C_l_off_1=C_l_off_1.T*D
    C_l_off_1=C_l_off_1.T
    
    return C_l_off_1

def BHCoefficients(x, l, nr, mu1, mu):
    """
    Calculates Bohren and Huffman Mie coefficients
    """
    j_x = spherical_jn(l, x)
    j_mx = spherical_jn(l, nr * x)
    h_x = spherical_jn(l, x) + 1j * spherical_yn(l, x)
    dj_x = j_x + x * spherical_jn(l, x, derivative=True)
    dj_mx = j_mx + nr * x * spherical_jn(l, nr * x, derivative=True)
    dh_x = h_x + x * (spherical_jn(l, x, derivative=True) +
     1j * spherical_yn(l, x, derivative=True))

    a = (mu * nr ** 2 * j_mx * dj_x - mu1 * j_x * dj_mx) /\
     (mu * nr ** 2 * j_mx * dh_x - mu1 * h_x * dj_mx)

    b = (mu1 * j_mx * dj_x - mu * j_x * dj_mx) /\
     (mu1 * j_mx * dh_x - mu * h_x * dj_mx)

    c = (mu1 * j_x * dh_x - mu * h_x * dj_x) /\
     (mu1 * j_mx * dh_x - mu * h_x * dj_mx)

    d = (mu1 * nr * j_x * dh_x - mu1 * nr * h_x * dj_x) /\
     (mu * nr ** 2 * j_mx * dh_x - mu1 * h_x * dj_mx)

    return a, b, c, d
