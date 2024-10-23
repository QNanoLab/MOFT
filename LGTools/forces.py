# -*- coding: utf-8 -*-
# %%
"""
ANALITIC CALCULATIONS OF OPTICAL FORCES FOR DISPLACED FOCUSED BEAMS. Created on Wed Sep 18 2024

@author: Iker Gómez Viloria

The formulas shown in this document are adressed and explained in the paper ... 
"""

import numpy as np
from scipy import constants as spc
import cmath

def Z_Forces_Analitic_Calculation(l1,l2,p,k,l_array,C,Cn,alpha_l,beta_l):
    """
    Optical forces in z direction generated by ON-AXIS cylindrically symmetric monochromatic beams on spherical particles. 
    """

    T_z = (alpha_l[:-1] + np.conj(alpha_l[1:])) + 2 * alpha_l[:-1]*np.conj(alpha_l[1:]) + \
        2*beta_l[:-1]*np.conj(beta_l[1:])
    
    if p!=0:
        A_j = C
        m_z = l1 + p
        l_min=max(np.abs(m_z),1)
        Beta=(l_array)*(l_array+2)*(l_array+m_z+1)*(l_array-m_z+1)/((2*(l_array)+3)*(2*(l_array)+1))

        FZ1 = np.dot(np.sqrt(Beta[l_min:-1])/(l_array[l_min:-1]+1),\
            (A_j[l_min:-1] * np.conj(A_j[l_min+1:]) * T_z[l_min:]).astype(complex).imag) -\
            np.dot(p*m_z/(l_array[l_min:]*(l_array[l_min:]+1)) ,\
            A_j[l_min:]*np.conj(A_j[l_min:]) * (alpha_l[l_min:]*np.conj(alpha_l[l_min:]) - beta_l[l_min:]*np.conj(beta_l[l_min:]) + np.real(alpha_l[l_min:])))

        FZ = spc.epsilon_0/(2*(k)**2) * np.real(FZ1)

    else:
        p=1
        m_z = l1 + p
        A_j = C/np.sqrt(2)
        l_min=max(np.abs(m_z),1)
        Beta=(l_array)*(l_array+2)*(l_array+m_z+1)*(l_array-m_z+1)/((2*(l_array)+3)*(2*(l_array)+1))

        FZ1 = np.dot(np.sqrt(Beta[l_min:-1])/(l_array[l_min:-1]+1),\
            (A_j[l_min:-1] * np.conj(A_j[l_min+1:]) * T_z[l_min:]).astype(complex).imag) -\
            np.dot(p*m_z/(l_array[l_min:]*(l_array[l_min:]+1)) ,\
            A_j[l_min:]*np.conj(A_j[l_min:]) * (alpha_l[l_min:]*np.conj(alpha_l[l_min:]) - beta_l[l_min:]*np.conj(beta_l[l_min:]) + np.real(alpha_l[l_min:])))

        p=-1
        m_z = l2 + p
        A_j = Cn/np.sqrt(2)
        l_min=max(np.abs(m_z),1)
        Beta=(l_array)*(l_array+2)*(l_array+m_z+1)*(l_array-m_z+1)/((2*(l_array)+3)*(2*(l_array)+1))

        FZm1 = np.dot(np.sqrt(Beta[l_min:-1])/(l_array[l_min:-1]+1),\
            (A_j[l_min:-1] * np.conj(A_j[l_min+1:]) * T_z[l_min:]).astype(complex).imag) -\
            np.dot(p*m_z/(l_array[l_min:]*(l_array[l_min:]+1)) ,\
            A_j[l_min:]*np.conj(A_j[l_min:]) * (alpha_l[l_min:]*np.conj(alpha_l[l_min:]) - beta_l[l_min:]*np.conj(beta_l[l_min:]) + np.real(alpha_l[l_min:])))

        FZ = spc.epsilon_0/(2*(k)**2) * np.real(FZ1+FZm1)

    return FZ


def X_Forces_Analitic_Calculation(l,p,k,l_max,m_array,C,Cn,alpha_l,beta_l):
    """
    Optical forces in x direction generated by OFF-AXIS cylindrically symmetric monochromatic beams on spherical particles. 
    """

    m_lim = l_max-l
    m_mlim = -m_lim-1
    m_2mlim = -m_lim+1

    if m_mlim == 0:
        m_mlim = None
    if m_2mlim == 0:
        m_2mlim = None

    Alpha_2=(l)*(l+2)*(l-m_array+1)*(l-m_array)/((2*(l)+3)*(2*(l)+1))
    Alpha_3=(l)*(l+2)*(l+m_array+2)*(l+m_array+1)/((2*(l)+3)*(2*(l)+1))
    Alpha_4=(l-m_array)*(l+m_array+1)

    if p==0:
        A = C/np.sqrt(2)
        Am = Cn/np.sqrt(2)
        B_1 = (A.T * alpha_l + Am.T * beta_l).T
        B_m1 = (Am.T * alpha_l + A.T * beta_l).T

        if m_lim == 0:
            F1 = 0
            F2 = 0
            F3 = np.dot(1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[:-1]]),(A[l,:-1]*np.conjugate(B_1[l,1:])+B_1[l,:-1]*np.conjugate(A[l,1:])+2*B_1[l,:-1]*np.conjugate(B_1[l,1:])))
            F4 = 0
            F5 = 0
            F6 = np.dot(-1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[:-1]]),(Am[l,:-1]*np.conjugate(B_m1[l,1:])+B_m1[l,:-1]*np.conjugate(Am[l,1:])+2*B_m1[l,:-1]*np.conjugate(B_m1[l,1:]))) 
        else:
            F1 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_2[m_lim-1:m_mlim]]),(A[l+1,m_lim-1:m_mlim]*np.conjugate(B_1[l,m_lim:-m_lim])+B_1[l+1,m_lim-1:m_mlim]*np.conjugate(A[l,m_lim:-m_lim])+2*B_1[l+1,m_lim-1:m_mlim]*np.conjugate(B_1[l,m_lim:-m_lim])))
            F2 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_3[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(B_1[l+1,m_lim+1:m_2mlim])+B_1[l,m_lim:-m_lim]*np.conjugate(A[l+1,m_lim+1:m_2mlim])+2*B_1[l,m_lim:-m_lim]*np.conjugate(B_1[l+1,m_lim+1:m_2mlim])))
            F3 = np.dot(1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(B_1[l,m_lim+1:m_2mlim])+B_1[l,m_lim:-m_lim]*np.conjugate(A[l,m_lim+1:m_2mlim])+2*B_1[l,m_lim:-m_lim]*np.conjugate(B_1[l,m_lim+1:m_2mlim])))
            F4 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_2[m_lim-1:m_mlim]]),(Am[l+1,m_lim-1:m_mlim]*np.conjugate(B_m1[l,m_lim:-m_lim])+B_m1[l+1,m_lim-1:m_mlim]*np.conjugate(Am[l,m_lim:-m_lim])+2*B_m1[l+1,m_lim-1:m_mlim]*np.conjugate(B_m1[l,m_lim:-m_lim])))
            F5 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_3[m_lim:-m_lim]]),(Am[l,m_lim:-m_lim]*np.conjugate(B_m1[l+1,m_lim+1:m_2mlim])+B_m1[l,m_lim:-m_lim]*np.conjugate(Am[l+1,m_lim+1:m_2mlim])+2*B_m1[l,m_lim:-m_lim]*np.conjugate(B_m1[l+1,m_lim+1:m_2mlim])))
            F6 = np.dot(-1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[m_lim:-m_lim]]),(Am[l,m_lim:-m_lim]*np.conjugate(B_m1[l,m_lim+1:m_2mlim])+B_m1[l,m_lim:-m_lim]*np.conjugate(Am[l,m_lim+1:m_2mlim])+2*B_m1[l,m_lim:-m_lim]*np.conjugate(B_m1[l,m_lim+1:m_2mlim])))
        F_xl =  spc.epsilon_0/(4*k**2) * np.real((1j/(l+1)) * (F1+F2+F3+F4+F5+F6))

    else:
        A = C

        if m_lim == 0:
            T_3 = np.real(alpha_l[l]) + (alpha_l[l]*np.conj(alpha_l[l]) - beta_l[l]*np.conj(beta_l[l]))
                
            F1 = 0
            F2 = 0
            F3 = np.dot(p * 2* 1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[:-1]]),(A[l,:-1]*np.conjugate(A[l,1:])*T_3))
        else:
            T_2 = alpha_l[l] + np.conj(alpha_l[l+1]) + 2 * (np.conj(alpha_l[l+1])*alpha_l[l] + np.conj(beta_l[l+1])*beta_l[l])
            T_3 = np.real(alpha_l[l]) + (alpha_l[l]*np.conj(alpha_l[l]) - beta_l[l]*np.conj(beta_l[l]))
                
            F1 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_2[m_lim-1:m_mlim]]),(A[l+1,m_lim-1:m_mlim]*np.conjugate(A[l,m_lim:-m_lim])*np.conj(T_2)))
            F2 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_3[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(A[l+1,m_lim+1:m_2mlim])*T_2))
            F3 = np.dot(p * 2* 1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(A[l,m_lim+1:m_2mlim])*T_3))
            
        F_xl =  np.real(spc.epsilon_0/(4*k**2) * (1j/(l+1)) * (F1+F2+F3))

    return F_xl

def XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C,Cn,alpha_l,beta_l):
    """
    Optical forces in x and y directions generated by OFF-AXIS cylindrically symmetric monochromatic beams on spherical particles. 
    """

    m_lim = l_max-l
    m_mlim = -m_lim-1
    m_2mlim = -m_lim+1

    if m_mlim == 0:
        m_mlim = None
    if m_2mlim == 0:
        m_2mlim = None

    Alpha_2=(l)*(l+2)*(l-m_array+1)*(l-m_array)/((2*(l)+3)*(2*(l)+1))
    Alpha_3=(l)*(l+2)*(l+m_array+2)*(l+m_array+1)/((2*(l)+3)*(2*(l)+1))
    Alpha_4=(l-m_array)*(l+m_array+1)

    if p==0:
        A = C/np.sqrt(2)
        Am = Cn/np.sqrt(2)
        B_1 = (A.T * alpha_l + Am.T * beta_l).T
        B_m1 = (Am.T * alpha_l + A.T * beta_l).T

        if m_lim == 0:
            F1 = 0
            F2 = 0
            F3 = np.dot(1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[:-1]]),(A[l,:-1]*np.conjugate(B_1[l,1:])+B_1[l,:-1]*np.conjugate(A[l,1:])+2*B_1[l,:-1]*np.conjugate(B_1[l,1:])))
            F4 = 0
            F5 = 0
            F6 = np.dot(-1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[:-1]]),(Am[l,:-1]*np.conjugate(B_m1[l,1:])+B_m1[l,:-1]*np.conjugate(Am[l,1:])+2*B_m1[l,:-1]*np.conjugate(B_m1[l,1:]))) 
        else:
            F1 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_2[m_lim-1:m_mlim]]),(A[l+1,m_lim-1:m_mlim]*np.conjugate(B_1[l,m_lim:-m_lim])+B_1[l+1,m_lim-1:m_mlim]*np.conjugate(A[l,m_lim:-m_lim])+2*B_1[l+1,m_lim-1:m_mlim]*np.conjugate(B_1[l,m_lim:-m_lim])))
            F2 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_3[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(B_1[l+1,m_lim+1:m_2mlim])+B_1[l,m_lim:-m_lim]*np.conjugate(A[l+1,m_lim+1:m_2mlim])+2*B_1[l,m_lim:-m_lim]*np.conjugate(B_1[l+1,m_lim+1:m_2mlim])))
            F3 = np.dot(1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(B_1[l,m_lim+1:m_2mlim])+B_1[l,m_lim:-m_lim]*np.conjugate(A[l,m_lim+1:m_2mlim])+2*B_1[l,m_lim:-m_lim]*np.conjugate(B_1[l,m_lim+1:m_2mlim])))
            F4 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_2[m_lim-1:m_mlim]]),(Am[l+1,m_lim-1:m_mlim]*np.conjugate(B_m1[l,m_lim:-m_lim])+B_m1[l+1,m_lim-1:m_mlim]*np.conjugate(Am[l,m_lim:-m_lim])+2*B_m1[l+1,m_lim-1:m_mlim]*np.conjugate(B_m1[l,m_lim:-m_lim])))
            F5 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_3[m_lim:-m_lim]]),(Am[l,m_lim:-m_lim]*np.conjugate(B_m1[l+1,m_lim+1:m_2mlim])+B_m1[l,m_lim:-m_lim]*np.conjugate(Am[l+1,m_lim+1:m_2mlim])+2*B_m1[l,m_lim:-m_lim]*np.conjugate(B_m1[l+1,m_lim+1:m_2mlim])))
            F6 = np.dot(-1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[m_lim:-m_lim]]),(Am[l,m_lim:-m_lim]*np.conjugate(B_m1[l,m_lim+1:m_2mlim])+B_m1[l,m_lim:-m_lim]*np.conjugate(Am[l,m_lim+1:m_2mlim])+2*B_m1[l,m_lim:-m_lim]*np.conjugate(B_m1[l,m_lim+1:m_2mlim])))
        
        F_l = spc.epsilon_0/(4*k**2) * (1j/(l+1)) * (F1+F2+F3+F4+F5+F6)
 
    else:
        A = C

        if m_lim == 0:
            T_3 = np.real(alpha_l[l]) + (alpha_l[l]*np.conj(alpha_l[l]) - beta_l[l]*np.conj(beta_l[l]))

            F1 = 0
            F2 = 0
            F3 = np.dot(p * 2* 1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[:-1]]),(A[l,:-1]*np.conjugate(A[l,1:])*T_3))
        else:
            T_2 = alpha_l[l] + np.conj(alpha_l[l+1]) + 2 * (np.conj(alpha_l[l+1])*alpha_l[l] + np.conj(beta_l[l+1])*beta_l[l])
            T_3 = np.real(alpha_l[l]) + (alpha_l[l]*np.conj(alpha_l[l]) - beta_l[l]*np.conj(beta_l[l]))

            F1 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_2[m_lim-1:m_mlim]]),(A[l+1,m_lim-1:m_mlim]*np.conjugate(A[l,m_lim:-m_lim])*np.conj(T_2)))
            F2 = np.dot(np.asarray([cmath.sqrt(x) for x in Alpha_3[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(A[l+1,m_lim+1:m_2mlim])*T_2))
            F3 = np.dot(p * 2* 1j/l * np.asarray([cmath.sqrt(x) for x in Alpha_4[m_lim:-m_lim]]),(A[l,m_lim:-m_lim]*np.conjugate(A[l,m_lim+1:m_2mlim])*T_3))
            
        F_l = spc.epsilon_0/(4*k**2) * (1j/(l+1)) * (F1+F2+F3)

    F_xl = np.real(F_l) 
    F_yl = np.imag(F_l)

    return F_xl, F_yl

def Z_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C,Cn,alpha_l,beta_l):
    """
    Optical forces in z direction generated by OFF-AXIS cylindrically symmetric monochromatic beams on spherical particles. 
    """

    m_lim = l_max-l

    T_z = (alpha_l[l] + np.conj(alpha_l[l+1])) + 2 * alpha_l[l]*np.conj(alpha_l[l+1]) +\
        2*beta_l[l]*np.conj(beta_l[l+1])
    
    Beta=(l)*(l+2)*(l+m_array+1)*(l-m_array+1)/((2*(l)+3)*(2*(l)+1))

    if p!=0:
        A_j = C

        FZ1 = (np.dot(np.sqrt(Beta[m_lim:-m_lim])/((l+1)),\
            (A_j[l,m_lim:-m_lim] * np.conj(A_j[l+1,m_lim:-m_lim]) * T_z).astype(complex).imag)) -\
            np.dot(p*m_array[m_lim:-m_lim]/(l*(l+1)) ,\
            A_j[l,m_lim:-m_lim] * np.conj(A_j[l,m_lim:-m_lim]) * ((alpha_l[l]*np.conj(alpha_l[l]) - beta_l[l]*np.conj(beta_l[l])) + np.real(alpha_l[l])))
        
        FZ = spc.epsilon_0/(2*(k)**2) * np.real(FZ1)
                
    else:
        A_1 = C/np.sqrt(2)
        A_m1 = Cn/np.sqrt(2)
        B_1 = (A_1.T * alpha_l + A_m1.T * beta_l).T
        B_m1 = (A_m1.T * alpha_l + A_1.T * beta_l).T

        FZ1 = (np.dot(np.sqrt(Beta[m_lim:-m_lim])/((l+1)),\
            (A_1[l+1,m_lim:-m_lim]*np.conj(B_1[l,m_lim:-m_lim]) - A_1[l,m_lim:-m_lim]*np.conj(B_1[l+1,m_lim:-m_lim]) + 2*B_1[l+1,m_lim:-m_lim]*np.conj(B_1[l,m_lim:-m_lim])).astype(complex).imag)) +\
            np.dot(m_array[m_lim:-m_lim]/(l*(l+1)) ,\
            np.real(A_1[l,m_lim:-m_lim] * np.conj(B_1[l,m_lim:-m_lim])) + B_1[l,m_lim:-m_lim] * np.conj(B_1[l,m_lim:-m_lim]))

        FZm1 = (np.dot(np.sqrt(Beta[m_lim:-m_lim])/((l+1)),\
            (A_m1[l+1,m_lim:-m_lim]*np.conj(B_m1[l,m_lim:-m_lim]) - A_m1[l,m_lim:-m_lim]*np.conj(B_m1[l+1,m_lim:-m_lim]) + 2*B_m1[l+1,m_lim:-m_lim]*np.conj(B_m1[l,m_lim:-m_lim])).astype(complex).imag)) +\
            np.dot(-m_array[m_lim:-m_lim]/(l*(l+1)) ,\
            np.real(A_m1[l,m_lim:-m_lim] * np.conj(B_m1[l,m_lim:-m_lim])) + B_m1[l,m_lim:-m_lim] * np.conj(B_m1[l,m_lim:-m_lim]))

        FZ = -spc.epsilon_0/(2*(k)**2) * np.real(FZ1+FZm1)

    return FZ

def Energy_from_forces(forces_disp, step_size):
    """
    Calculates the potential well for a given force_vs_beam_displacement array.
    """
    
    F_matrix,_ = np.meshgrid(forces_disp,forces_disp)
    F_matrix_tri = F_matrix*np.tri(*F_matrix.shape)
    Pot = np.sum(F_matrix_tri,axis=1)*step_size

    return Pot