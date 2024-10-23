# -*- coding: utf-8 -*-
# %%
"""
ANALITIC CALCULATIONS OF FORCES FOR DISPLACED FOCUSED BEAMS. Created on Wed Apr  7 15:33:22 2021

@author: Iker
"""

import numpy as np
from LGTools.coefficients import Z_disp_TM_chain, XY_disp_TM_chain, coefficients_XY_disp_TM
from LGTools.forces import XY_Forces_Analitic_Calculation_l, Z_Forces_Analitic_Calculation_l, Energy_from_forces
from multiprocessing import Pool
import itertools as it
from tqdm import trange
from scipy.signal import find_peaks

def Z_J_equilibrium_point_calculator(forces_disp, disp_array, step_size):
    """
    Finds the main equilibrium point from the force_vs_beam_displacement array of z axis.
    """
    Fz_0_m=forces_disp[len(disp_array)//2]
    Fz_0_p=forces_disp[len(disp_array)//2 +1]
    Pot_z = Energy_from_forces(forces_disp, step_size)
    d=0
    if np.any(forces_disp < 0) and np.any(forces_disp > 0) and len(find_peaks(-Pot_z,height=-max(abs(Pot_z))))>0 :
        if Fz_0_m<0 and Fz_0_p>0:
            kz=(Fz_0_p-Fz_0_m)/(step_size)
            eq_true=1
        else:
            if Fz_0_m>0 and Fz_0_p>0:
                while Fz_0_m>0 and (len(disp_array)//2 + d)>0:
                    d=d-1
                    Fz_0_m=forces_disp[len(disp_array)//2 + d]
                    Fz_0_p=forces_disp[len(disp_array)//2 + d + 1]
                    kz=(Fz_0_p-Fz_0_m)/(step_size)
                    eq_true=1
            else:
                while Fz_0_p<0 and (len(disp_array)//2 + d + 1)<(len(forces_disp)-1):
                    d=d+1
                    Fz_0_m=forces_disp[len(disp_array)//2 + d]
                    Fz_0_p=forces_disp[len(disp_array)//2 + d + 1]
                    kz=(Fz_0_p-Fz_0_m)/(step_size)
                    eq_true=1
        if (len(disp_array)//2 + d - 1)<0 or (len(disp_array)//2 + d)>len(forces_disp):
            print('NO TRAPPING IN Z IN THE GIVEN DISPLACEMENT RANGE')
            eq_true=0
        else:
            z_interpolation=np.abs(forces_disp[len(disp_array)//2 + d])*step_size/(np.abs(forces_disp[len(disp_array)//2 + d])+np.abs(forces_disp[len(disp_array)//2 + d + 1]))
            eq_disp_z=disp_array[len(disp_array)//2 + d]+z_interpolation
    else:
        print("NO TRAPPING IN Z")
        eq_disp_z=0
        eq_true=0
        kz=0

    return eq_true, eq_disp_z, kz

def X_J_equilibrium_point_calculator(forces_disp_x, disp_array_x, step_size_x):
    """
    Finds the main equilibrium point from the force_vs_beam_displacement array of x axis.
    """

    Pot_x = Energy_from_forces(forces_disp_x, step_size_x)

    d_eq_X_1 = -np.abs(disp_array_x[np.argmin(Pot_x)])
    arg_eq_disp = np.argmin(np.abs(disp_array_x - d_eq_X_1))
    #Find equilibrium point and calculate kx
    Fx_0_m=forces_disp_x[arg_eq_disp]
    Fx_0_p=forces_disp_x[arg_eq_disp+1]

    d=0
    eq_true=0
    if Fx_0_m<0 and Fx_0_p>0:
        kx=(Fx_0_p-Fx_0_m)/(step_size_x)
    else:
        if Fx_0_m>0 and Fx_0_p>0:
            while Fx_0_m>0 and (arg_eq_disp + d)>0:
                d=d-1
                Fx_0_m=forces_disp_x[arg_eq_disp + d]
                Fx_0_p=forces_disp_x[arg_eq_disp + d + 1]
                kx=(Fx_0_p-Fx_0_m)/(step_size_x)
                eq_true=1
        else:
            while Fx_0_p<0 and (arg_eq_disp + d + 1)<len(forces_disp_x)-1:
                d=d+1
                Fx_0_m=forces_disp_x[arg_eq_disp + d]
                Fx_0_p=forces_disp_x[arg_eq_disp + d + 1]
                kx=(Fx_0_p-Fx_0_m)/(step_size_x)
                eq_true=1

    x_interpolation=np.abs(forces_disp_x[arg_eq_disp + d])*step_size_x/(np.abs(forces_disp_x[arg_eq_disp + d])+np.abs(forces_disp_x[arg_eq_disp + d + 1]))
    eq_disp_x = disp_array_x[arg_eq_disp + d]+x_interpolation

    return eq_true, eq_disp_x, kx

def Forces_z_summation_loop(l,p,k,l_max,m_array,C_l_off_1_EQXZ1,C_l_off_1_EQXZ2,C_l_off_1_EQXZ3,C_l_off_m1_EQXZ1,C_l_off_m1_EQXZ2,C_l_off_m1_EQXZ3,alpha_array,beta_array):

    F_l_EQXZ1 = Z_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_EQXZ1,C_l_off_m1_EQXZ1,alpha_array,beta_array)
    F_l_EQXZ2 = Z_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_EQXZ2,C_l_off_m1_EQXZ2,alpha_array,beta_array)
    F_l_EQXZ3 = Z_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_EQXZ3,C_l_off_m1_EQXZ3,alpha_array,beta_array)

    return F_l_EQXZ1,F_l_EQXZ2,F_l_EQXZ3

def Forces_z_summation_loop_wrapper(args):
    return Forces_z_summation_loop(*args)

def Forces_x_summation_loop(l,p,k,l_max,m_array,C_l_off_1_EQXZ4,C_l_off_1_EQXZ5,C_l_off_1_EQXZ6,C_l_off_m1_EQXZ4,C_l_off_m1_EQXZ5,C_l_off_m1_EQXZ6,alpha_array,beta_array):

    F_l4, F_y1 = XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_EQXZ4,C_l_off_m1_EQXZ4,alpha_array,beta_array)
    F_l5, F_y1 = XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_EQXZ5,C_l_off_m1_EQXZ5,alpha_array,beta_array)
    F_l6, F_y1 = XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_EQXZ6,C_l_off_m1_EQXZ6,alpha_array,beta_array)

    return F_l4, F_l5, F_l6

def Forces_x_summation_loop_wrapper(args):
    return Forces_x_summation_loop(*args)
            
def Off_axis_eq_p_finder(C_l_off_1_EQXZ1, C_l_off_m1_EQXZ1, T_M_tot_X2, T_M_tot_X2_m1, disp_z, eq_point_X, it_max, d_iter_z, d_iter_x, D, C_l_1, C_l_m1, L_n_1, L_n_2, S1, l_max, k, p, m_array, alpha_array, beta_array, P_0, ncores):
    """
    Traces the equilibrium point OFF-AXIS using the linear scaling of the optical forces when small displacements of the beam are performed.
    The coefficients and translation matrices of an initial approached equilibrium point must be provided.
    It is possible to optimize the number of iteations realized for tracing the equilibrium point in x and z adjusting d_iter_z and d_iter_x.
    """
    
    ###### OFF-AXIS EQUILIBRIUM POINT FINDER ######
    m_z_p=L_n_1+p

    for i in trange(0,it_max):
        #print(disp_z)
        d_iter_alarm=1
        d_iter_mult=1
        
        ###### ITERATION OF X EQUILIBRIUM POINT FINDER ######
        #### 2 points around equilibrium point in Z direction and chech if the Fz=0 is in between, interpolate to find that z point ##
        while d_iter_alarm == 1:
            d_iter=d_iter_z/(2**i) * d_iter_mult
            disp_z_kz = disp_z + d_iter
            kd_z2= disp_z_kz*k

            if p==0:
                T_M_tot_Z2 = Z_disp_TM_chain(kd_z2, 1)
                T_M_tot_eq3 = np.dot(T_M_tot_X2,T_M_tot_Z2)
                C_l_off_1_EQXZ2 = S1*coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq3, L_n_1+1, l_max)
                
                T_M_tot_Z2 = Z_disp_TM_chain(kd_z2, -1)
                T_M_tot_eq3 = np.dot(T_M_tot_X2_m1,T_M_tot_Z2)
                C_l_off_m1_EQXZ2 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot_eq3, L_n_2-1, l_max)

            else:
                T_M_tot_Z2 = Z_disp_TM_chain(kd_z2, p)
                T_M_tot_eq3 = np.dot(T_M_tot_X2,T_M_tot_Z2)
                C_l_off_1_EQXZ2 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq3, m_z_p, l_max)
                C_l_off_m1_EQXZ2 = 0

            disp_z_kz = disp_z - d_iter
            kd_z2= disp_z_kz*k

            if p==0:
                T_M_tot_Z2 = Z_disp_TM_chain(kd_z2, 1)
                T_M_tot_eq3 = np.dot(T_M_tot_X2,T_M_tot_Z2)
                C_l_off_1_EQXZ3 = S1*coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq3, L_n_1+1, l_max)
                
                T_M_tot_Z2 = Z_disp_TM_chain(kd_z2, -1)
                T_M_tot_eq3 = np.dot(T_M_tot_X2_m1,T_M_tot_Z2)
                C_l_off_m1_EQXZ3 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot_eq3, L_n_2-1, l_max)

            else:
                T_M_tot_Z2 = Z_disp_TM_chain(kd_z2, p)
                T_M_tot_eq4=np.dot(T_M_tot_X2,T_M_tot_Z2)
                C_l_off_1_EQXZ3 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq4, m_z_p, l_max)
                C_l_off_m1_EQXZ3 = 0
            
            Force_EQXZ1 = 0
            Force_EQXZ2 = 0
            Force_EQXZ3 = 0

            l = np.arange(1, l_max, 1)

            pool = Pool(processes=ncores)

            arg_list = list(zip(
                l, it.repeat(p), it.repeat(k), it.repeat(l_max), it.repeat(m_array),
                it.repeat(C_l_off_1_EQXZ1), it.repeat(C_l_off_1_EQXZ2), it.repeat(C_l_off_1_EQXZ3), it.repeat(C_l_off_m1_EQXZ1), it.repeat(C_l_off_m1_EQXZ2), it.repeat(C_l_off_m1_EQXZ3),
                it.repeat(alpha_array), it.repeat(beta_array)
            ))

            for o, values in enumerate(pool.imap(Forces_z_summation_loop_wrapper, arg_list), 1):
                F_l_EQXZ1,F_l_EQXZ2,F_l_EQXZ3=values
                Force_EQXZ1 =  Force_EQXZ1 + F_l_EQXZ1
                Force_EQXZ2 =  Force_EQXZ2 + F_l_EQXZ2
                Force_EQXZ3 =  Force_EQXZ3 + F_l_EQXZ3
            pool.close()

            kz_EQXZ1 = (P_0) * (Force_EQXZ2-Force_EQXZ1)/d_iter
            # print(kz_EQXZ1)

            if kz_EQXZ1<0:
                if Force_EQXZ2<0 and Force_EQXZ1>0:
                    d_eq_z=(disp_z*Force_EQXZ2 - (disp_z+d_iter)*Force_EQXZ1) / (Force_EQXZ2-Force_EQXZ1)
                    d_iter_alarm=0
                else:
                    if Force_EQXZ1<0 and Force_EQXZ3>0:
                        d_eq_z=((disp_z-d_iter)*Force_EQXZ1 - disp_z*Force_EQXZ3) / (Force_EQXZ1-Force_EQXZ3)
                        d_iter_alarm=0
                    else:
                        d_iter_mult=d_iter_mult+1
                        # print('Insuficient value of d_iter parameter')
            else:
                if Force_EQXZ2>0 and Force_EQXZ1<0:
                    d_eq_z=(disp_z*Force_EQXZ2 - (disp_z+d_iter)*Force_EQXZ1) / (Force_EQXZ2-Force_EQXZ1)
                    d_iter_alarm=0
                else:
                    if Force_EQXZ1>0 and Force_EQXZ3<0:
                        d_eq_z=((disp_z-d_iter)*Force_EQXZ1 - disp_z*Force_EQXZ3) / (Force_EQXZ1-Force_EQXZ3)
                        d_iter_alarm=0
                    else:
                        d_iter_mult=d_iter_mult+1
                        # print('Insuficient value of d_iter parameter')

        disp_z=d_eq_z
        kd_z_eq2=d_eq_z*k

        ###### ITERATION OF X EQUILIBRIUM POINT FINDER ######
        if p==0:
            T_M_tot_Z_EQ2 = Z_disp_TM_chain(kd_z_eq2, 1)
            T_M_tot_eq5=np.dot(T_M_tot_X2,T_M_tot_Z_EQ2)
            C_l_off_1_EQXZ4 = S1*coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq5, L_n_1+1, l_max)

            T_M_tot_Z_EQ2_m1 = Z_disp_TM_chain(kd_z_eq2, -1)
            T_M_tot_eq5=np.dot(T_M_tot_X2_m1,T_M_tot_Z_EQ2_m1)
            C_l_off_m1_EQXZ4 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot_eq5, L_n_2-1, l_max)

        else:
            T_M_tot_Z_EQ2 = Z_disp_TM_chain(kd_z_eq2, p)
            T_M_tot_eq5=np.dot(T_M_tot_X2,T_M_tot_Z_EQ2)
            C_l_off_1_EQXZ4 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq5, m_z_p, l_max)
            C_l_off_m1_EQXZ4=0
        
        d_iter_alarm=1
        d_iter_mult=1
        
        #### 2 points around equilibrium point in X direction and chech if the Fx=0 is in between, interpolate to find that x point ##
        while d_iter_alarm == 1:
            d_iter=d_iter_x/(2**i) * d_iter_mult

            disp = eq_point_X + d_iter
            kd=disp*k
            if p==0:
                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, 1)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2)
                C_l_off_1_EQXZ5 = S1*coefficients_XY_disp_TM(D, C_l_1, T_M_tot, L_n_1+1, l_max)

                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, -1)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2_m1)
                C_l_off_m1_EQXZ5 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot, L_n_2-1, l_max)

            else:
                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, p)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2)
                C_l_off_1_EQXZ5 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max)
                C_l_off_m1_EQXZ5 = 0

            disp = eq_point_X - d_iter
            kd=disp*k
            if p==0:
                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, 1)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2)
                C_l_off_1_EQXZ6 = S1*coefficients_XY_disp_TM(D, C_l_1, T_M_tot, L_n_1+1, l_max)

                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, -1)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2_m1)
                C_l_off_m1_EQXZ6 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot, L_n_2-1, l_max)
            else:
                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, p)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2)
                C_l_off_1_EQXZ6 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max)
                C_l_off_m1_EQXZ6 = 0

            Force_EQXZ4 = 0
            Force_EQXZ5 = 0
            Force_EQXZ6 = 0

            pool = Pool(processes=ncores)

            arg_list = list(zip(
                l, it.repeat(p), it.repeat(k), it.repeat(l_max), it.repeat(m_array), it.repeat(C_l_off_1_EQXZ4),
                it.repeat(C_l_off_1_EQXZ5), it.repeat(C_l_off_1_EQXZ6), it.repeat(C_l_off_m1_EQXZ4),
                it.repeat(C_l_off_m1_EQXZ5), it.repeat(C_l_off_m1_EQXZ6), it.repeat(alpha_array), it.repeat(beta_array)
            ))

            for o, values in enumerate(pool.imap(Forces_x_summation_loop_wrapper, arg_list), 1):
                F_l4, F_l5, F_l6 = values
                Force_EQXZ4 =  Force_EQXZ4 + F_l4
                Force_EQXZ5 =  Force_EQXZ5 + F_l5
                Force_EQXZ6 =  Force_EQXZ6 + F_l6

            pool.close()

            kx_EQXZ1 = (P_0) * (Force_EQXZ5-Force_EQXZ4)/d_iter
            # print(kz_EQXZ1)

            ## Checking if the Fx=0 is in between, interpolate to find that x point ##
            if kx_EQXZ1<0:
                if Force_EQXZ5<0 and Force_EQXZ4>0:
                    d_eq_x=(eq_point_X*Force_EQXZ5 - (eq_point_X+d_iter)*Force_EQXZ4) / (Force_EQXZ5-Force_EQXZ4)
                    d_iter_alarm=0
                else:
                    if Force_EQXZ4<0 and Force_EQXZ6>0:
                        d_eq_x=((eq_point_X-d_iter)*Force_EQXZ4 - eq_point_X*Force_EQXZ6) / (Force_EQXZ4-Force_EQXZ6)
                        d_iter_alarm=0
                    else:
                        d_iter_mult=d_iter_mult+1
                        # print('Insuficient value of d_iter parameter')
            else:
                if Force_EQXZ5>0 and Force_EQXZ4<0:
                    d_eq_x=(eq_point_X*Force_EQXZ5 - (eq_point_X+d_iter)*Force_EQXZ4) / (Force_EQXZ5-Force_EQXZ4)
                    d_iter_alarm=0
                else:
                    if Force_EQXZ4>0 and Force_EQXZ6<0:
                        d_eq_x=((eq_point_X-d_iter)*Force_EQXZ4 - eq_point_X*Force_EQXZ6) / (Force_EQXZ4-Force_EQXZ6)
                        d_iter_alarm=0
                    else:
                        d_iter_mult=d_iter_mult+1
                        # print('Insuficient value of d_iter parameter')

        ## New approach to off-axis equilibrium point ##    
        eq_point_X=d_eq_x
        kd_x2=eq_point_X*k
        
        if p==0:
            T_M_tot_X_EQ2, kd_sum_1, first_it = XY_disp_TM_chain(kd_x2,  1)
            T_M_tot_eq2=np.dot(T_M_tot_X_EQ2,T_M_tot_Z_EQ2)
            T_M_tot_X2=T_M_tot_X_EQ2
            C_l_off_1 = S1*coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq2, L_n_1+1, l_max)

            T_M_tot_X_EQ2, kd_sum_1, first_it = XY_disp_TM_chain(kd_x2,  -1)
            T_M_tot_eq2=np.dot(T_M_tot_X_EQ2,T_M_tot_Z_EQ2_m1)
            T_M_tot_X2_m1=T_M_tot_X_EQ2
            C_l_off_m1 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot_eq2, L_n_2-1, l_max)
        else:
            T_M_tot_X_EQ2, kd_sum_1, first_it = XY_disp_TM_chain(kd_x2,  p)
            T_M_tot_eq2=np.dot(T_M_tot_X_EQ2,T_M_tot_Z_EQ2)
            T_M_tot_X2=T_M_tot_X_EQ2
            C_l_off_1 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq2, m_z_p, l_max)
            C_l_off_m1=0

        C_l_off_1_EQXZ1=C_l_off_1
        C_l_off_m1_EQXZ1=C_l_off_m1
        
    return d_eq_x, d_eq_z, kx_EQXZ1, kz_EQXZ1, C_l_off_1, C_l_off_m1
    