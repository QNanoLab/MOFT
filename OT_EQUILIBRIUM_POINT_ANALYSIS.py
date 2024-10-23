# -*- coding: utf-8 -*-
# %%
"""
GENERAL FOCUSED LAGUERRE-GAUSSIAN BEAM TRAPPING ANALYSIS. Created on Wed Sep 18 2024

@author: Iker Gómez Viloria

DESCRIPTION: This program calculates the optical forces and torques at the equilibrium point of the optical trap considering:
    -Monochromatic Laguerre-Gaussian focused trapping beams, linearly or circularly polarized.
    -Spherical particles.

    For this purpose, the following steps are taken:

    1. Define the base parameters of the optical system. The main parameters are printed.
    2. Calculate the multipolar coefficients of the incident beam for the on-focus configuration and the Mie coefficients of the particle.
    3. Calculate Fz while the beam is displaced along the z-axis and determine the equilibrium point in this axis.
        3.1. If there is no equilibrium point in the z-axis, "NO TRAPPING IN Z" is printed. END OF THE PROGRAM.
        3.2. If there is an equilibrium point in the z-axis, calculate Fx while the beam is displaced along the x-axis from the equilibrium
             point of the z-axis and determine the equilibrium point in the x-axis.
            3.2.1. If the equilibrium point in the x-axis occurs when the displacement of the beam in this axis is equal to zero, data for
                   the on-axis equilibrium point is printed: kx, fx, kz, fz, Tz, (Fx=Fy=Fz=0, Tx=Ty=0). Electromagnetic fields of the optical
                   trapping system at the equilibrium point are plotted. END OF THE PROGRAM.
            3.2.2. If the equilibrium point in the x-axis occurs when the displacement of the beam in this axis is different from zero, the
                   new equilibrium point off-axis is calculated. Then, data for the off-axis equilibrium point is printed: 
                   Fy (Fx and Fz should approach zero), kx, fx, kz, fz, Tx, Ty, Tz.Electromagnetic fields of the optical trapping system at 
                   the equilibrium point are plotted. END OF THE PROGRAM.
    
    Set `show_force_plots` and `show_field_plots` to True or False to control the display of plots.
    Alternatively, if command line is used, these variables can be controlled by `--plot-forces` and `--plot-fields`. 
    
    WARNING!, `show_field_plots = True` uses a lot of memory. Adjust the value of the variable `resolution` to reduce the memory employed.
    WARNING!, variable `ncores` (number of cores employed in the paralelization process) must be adjusted to computer's resources.
"""

run_by_ipython = True
import sys
if len(sys.argv) > 0 and sys.argv[0] == 'OT_EQUILIBRIUM_POINT_ANALYSIS.py':
    import matplotlib
    matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt
    #plt.ion()
    run_by_ipython = False
else:
    sys.argv = ['']

del sys




import numpy as np
import argparse
from LGTools.coefficients import coefficients_focus, BHCoefficients, coefficients_Z_disp_TM, Z_disp_TM_chain, XY_disp_TM_chain, coefficients_XY_disp_TM
from LGTools.forces import XY_Forces_Analitic_Calculation_l, Z_Forces_Analitic_Calculation, Z_Forces_Analitic_Calculation_l, Energy_from_forces
from LGTools.torques import Z_Torques_Analitic_Calculation, XY_Torques_Analitic_Calculation_l, Z_Torques_Analitic_Calculation_l
from LGTools.equilibrium_point_finders import Off_axis_eq_p_finder, Z_J_equilibrium_point_calculator,X_J_equilibrium_point_calculator
from LGTools.multipolar_fields_forces_plots import Z_forces_energy_plots, X_forces_energy_plots, XZ_XY_fields_plots_TOT, XZ_XY_fields_plots_INC, On_axis_multipolar_coefficients_plot
from LGTools.summation_of_multipoles import ON_AXIS_multipole_summation_2_planes, OFF_AXIS_multipole_summation_loop_2_planes
from LGTools.coordinates_converter import cart2sph
from LGTools.beam import w_0_size
from scipy import constants as spc
from multiprocessing import Pool
from tqdm import trange



# %%
#### BASE PARAMETERS #####

#Sphere parameters
r_p = 500e-9 #radius of the spherical particle (nm)
nr = 1.5+0.0025j #refractive index of the particle
dr = 1 #sphere material densisty (g/cm^3)
mu = 1 #magnetic permeability of the particle (N/A^2)
v_sphere = (4/3)*np.pi*r_p**(3) #sphere volume (m^3)
wr = v_sphere * dr*1e3 * 9.81 #sphere weight (N)

#Beam parameters
B_P = 1 #Beam optical power before the objective lens (W)
wl = 1064e-9 #beam wavelength (m)
p = -1 #for left polarization set p=1, for right p=-1 and for linear p = 0
L_n_1 = -3 #Azimuthal order of the LG beam: with p and for linear polarization, for the beam component with p
L_n_2 = 0 #Second azimuthal order of the LG beam: use only for linear polarization (p=0), for the beam component with -p
S = 1 #use only for linear polarization (p=0), +1 for linear polarization in Y and -1 for linear polarization in X
q = 0 #radial order of LG
m_z_p = L_n_1+p #total angular momentum in z for on-axis configuration

#Beam focusing parameters
z = 0 #position of the paraxial (collimated) LG beam before focusing (m)
NA = 1.25 #Numerical Aperture of the lens
D_l = 10e-3 #Size (diameter) of the lens (m)
cutoff = 0.96 #objective lens filling factor
# w0 = 2e-3 #beam size (radius) just before entering the lens (m)

#Medium parameters
dm = 1000 #medium densisty (kg/m^3)
wm = v_sphere * dm * 9.81 #buoyant force (N)
mu1 = 1 #magnetic permeability of the medium surrounding the particle (N/A^2)
n2 = 1.33 #refractive index of the medium surrounding the particle

# SHOW PLOTS:
parser = argparse.ArgumentParser(
            prog='OT_EQUILIBRIUM_POINT_ANALYSIS',
            description='Calculates the optical forces and torques at the equilibrium point of the optical trap',
            epilog='Adjust the value of the variable `resolution` to reduce the memory for --plot-fields'
            )

parser.add_argument('--plot-forces',
            action='store_true', help='Plot the graphs of multipolar content, forces and energies') 

parser.add_argument('--plot-fields',
            action='store_true', help='Plot electromagnetic fields. WARNING, it uses a lot of memory')

args = parser.parse_args()

if run_by_ipython:
## Select the option that you prefer regarding plots!!!!!
# =======================================================
    show_force_plots = True
    show_field_plots = True
else:
    show_force_plots = args.plot_forces
    show_field_plots = args.plot_fields

#Medium/Beam/Particle dependen parameters
try:
    f = n2*D_l/(2*NA) #focal distance of focused beam
except ZeroDivisionError:
    raise ValueError('NA should be greater than 0')

w0 = w_0_size(max(np.abs(L_n_1), np.abs(L_n_2)), q, f, n2, NA, cutoff) #beam size (radius) just before entering the lens

k = (2*np.pi/wl)*n2 #wavenumber (1/m)
x = r_p*k #Optical size
I_0 = B_P*8*k**2*np.sqrt(spc.mu_0/spc.epsilon_0) #electromagnetic field intensity (W/m^2)

#Beam displacement parameters
max_disp_limit_z = 2000e-9# - L_n_1*100e-9
min_disp_limit_z = -max_disp_limit_z
steps_z = 50
step_size_z = ((max_disp_limit_z-min_disp_limit_z)/steps_z)
disp_array_z = np.arange(min_disp_limit_z,max_disp_limit_z+step_size_z,step_size_z)

max_disp_limit_x = (1500e-9 + max(np.abs(L_n_1), np.abs(L_n_2))*250e-9)
min_disp_limit_x = -max_disp_limit_x
steps_x = 2*steps_z
step_size_x =  ((max_disp_limit_x-min_disp_limit_x)/steps_x)
disp_array_x = np.arange(min_disp_limit_x,max_disp_limit_x+step_size_x,step_size_x)

forces_disp_z = np.zeros((len(disp_array_z)))
forces_disp_x = np.zeros((len(disp_array_x)))

#Computational parameters
ncores = 10 #cores employed in the paralelization
l_max = 30 #top limmit of the summation over j (must be equal to the l_max of translation matrices)

d_iter_z = 10e-9 #short displacement in Z to find the equilibrium point OFF AXIS
d_iter_x = 10e-9 #short displacement in X to find the equilibrium point OFF AXIS
it_max = 5 #number of iterations to find the equilibrium point OFF AXIS

m_array = np.arange(0,2*(l_max)+1,1)-l_max
l_array = np.arange(0,l_max+1,1)

#Generate the grid where the EM fields are evaluated (spherical coordinates)
resolution = 200 #resolution of the grid

max_disp_limit_y = 1500e-9
min_disp_limit_y = -max_disp_limit_y

vx = np.linspace(min_disp_limit_x,max_disp_limit_x,resolution)
vy = np.linspace(min_disp_limit_y,max_disp_limit_y,resolution)
vz = np.linspace(min_disp_limit_z,max_disp_limit_z,resolution)

VX,VZ = np.meshgrid(vx,vz,indexing= 'ij')
VX,VY = np.meshgrid(vx,vy,indexing= 'ij')
#VY,VZ = np.meshgrid(vy,vz,indexing= 'ij')

vxx = np.reshape(VX, (resolution*resolution,1))
vyy = np.reshape(VY, (resolution*resolution,1))
vzz = np.reshape(VZ, (resolution*resolution,1))

R,phi,theta = cart2sph([vxx,0,vzz])
R2,phi2,theta2 = cart2sph([vxx,vyy,0])
#R3,phi3,theta3 = cart2sph([0,vyy,vzz])

kr= R*k
kr2= R2*k
R = np.reshape(R, (resolution, resolution))
R2 = np.reshape(R2, (resolution, resolution))

#Print general parameters of the optical system
print(' ')
print('Sphere radius = '+str(r_p*1e9)+'nm')
print('Sphere refractive index = '+str(nr))
print('Beam wavelength = '+str(round(wl*1e9))+'nm')
if NA !=0:
    print('Beam width entering the objective = '+str(round(w0*1e3,2))+'mm')
print('NA (Numerical Aperture) = '+str(NA))
print('Helicity = '+str(p))
if p == 0:
    print('Topological charge of left polarized component = '+str(L_n_1))
    print('Topological charge of right polarized component = '+str(L_n_2))
    if S==1:
        print('Linearly polarized in X axis')
    if S==-1:
        print('Linearly polarized in Y axis')
else:
    print('Topological charge = '+str(L_n_1))
print('Top multipolar order summation limit = '+str(l_max))
print(' ')
print(f'Plotting optical forces figures: {show_force_plots}')
print(f'Plotting electromagnetic fields figures: {show_field_plots}')
print(' ')


#Mie coefficients and alpha and beta (also for internal EM fields)
a_array,b_array,c_array,d_array = BHCoefficients(x, l_array, nr/n2, mu1, mu)
alpha_array = -(a_array+b_array)/(2)
beta_array = (a_array-b_array)/(2)
alpha_int_array = (d_array+c_array)/(2)
beta_int_array = -(d_array-c_array)/(2)

#On-focus multipolar content
#The linear polarization (p=0) is constructed as the superpositiom of both helicities, so the corresponding multipolar contents are considered.
if p == 0:
    D,C_l_1 = coefficients_focus(l_max, L_n_1, q, 1, f, k,  n2, w0, z, NA)
    D,C_l_m1 = coefficients_focus(l_max, L_n_2, q, -1, f, k, n2, w0, z, NA)
    D_C_l_1 = S*D*C_l_1
    D_C_l_m1 = D*C_l_m1
    C_l_1_s = np.abs(D_C_l_1/np.sqrt(2))**2+np.abs(D_C_l_m1/np.sqrt(2))**2 
else:
    D,C_l_1 = coefficients_focus(l_max, L_n_1, q, p, f, k, n2, w0, z, NA)
    C_l_m1 = 0
    D_C_l_1 = D*C_l_1
    D_C_l_m1 = 0
    C_l_1_s = np.abs(D_C_l_1)**2

if show_force_plots:
    On_axis_multipolar_coefficients_plot(p, D_C_l_1, D_C_l_m1, alpha_array, beta_array)

print('Sum of Incident BSC = '+str(round(np.sum(C_l_1_s),3)))
# print('Sum of Scattering Coeff. = '+str(np.sum(sca_C)))
# print('Sum of Coupling Coeff. = '+str(np.sum(C_tot)))

#%%
####### FORCES IN Z IN ORDER TO DETERMINE THE EQUILIBRIUM POINT #######

for d in trange(0, len(disp_array_z)):
    disp = disp_array_z[d]
    kd=disp*k

    if p == 0:
        T_M_tot_1 = Z_disp_TM_chain(kd, 1)
        C_l_off_1 = S*coefficients_Z_disp_TM(D, C_l_1, T_M_tot_1, L_n_1+1, l_max)

        T_M_tot_m1 = Z_disp_TM_chain(kd, -1)
        C_l_off_m1 = coefficients_Z_disp_TM(D, C_l_m1, T_M_tot_m1, L_n_1-1, l_max)

    else:
        T_M_tot = Z_disp_TM_chain(kd, p)
        C_l_off_1 = coefficients_Z_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max)
        C_l_off_m1 = 0

    F_z =  I_0 * Z_Forces_Analitic_Calculation(L_n_1,L_n_2,p,k,l_array,C_l_off_1,C_l_off_m1,alpha_array,beta_array) - wr + wm
    forces_disp_z[d]=F_z

Pot_z = Energy_from_forces(forces_disp_z, step_size_z)
if show_force_plots:
    Z_forces_energy_plots(forces_disp_z, disp_array_z, Pot_z)

####### EQUILIBRIUM POINT IN Z AND ITS TRANSLATION MATRIX CALCULATIONS ##########

eq_true, eq_disp_z, kz = Z_J_equilibrium_point_calculator(forces_disp_z, disp_array_z, step_size_z)

#%%
####### FORCES IN X IN THE EQUILIBRIUM POINT OF Z #######

if eq_true==1:
    kd=eq_disp_z*k
    
    if p == 0:
        T_M_tot_Z_1=Z_disp_TM_chain(kd, 1)
        T_M_tot_Z_m1=Z_disp_TM_chain(kd, -1)
    else:
        T_M_tot_Z=Z_disp_TM_chain(kd, p)

    first_it=0
    kd_sum_1=0
    T_M_tot_X=0
    first_it_1=0
    kd_sum_1=0
    T_M_tot_X_1=0
    first_it_m1=0
    kd_sum_m1=0
    T_M_tot_X_m1=0

    for d in trange(1, (len(disp_array_x)//2+1)):
        disp = disp_array_x[steps_x//2+d]
        kd=disp*k

        if p == 0:
            T_M_tot_X_1, kd_sum_1, first_it_1 = XY_disp_TM_chain(kd, 1, kd_sum_1, first_it_1, T_M_tot_X_1)
            T_M_tot_1 = np.dot(T_M_tot_X_1,T_M_tot_Z_1)
            C_l_off_1 = S*coefficients_XY_disp_TM(D, C_l_1, T_M_tot_1, 1+L_n_1, l_max)

            T_M_tot_X_m1, kd_sum_m1, first_it_m1 = XY_disp_TM_chain(kd, -1, kd_sum_m1, first_it_m1, T_M_tot_X_m1)
            T_M_tot_m1 = np.dot(T_M_tot_X_m1,T_M_tot_Z_m1)
            C_l_off_m1 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot_m1, -1+L_n_2, l_max)
        else:
            T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, p, kd_sum_1, first_it, T_M_tot_X)
            T_M_tot = np.dot(T_M_tot_X,T_M_tot_Z)
            C_l_off_1 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max)

        F_x = 0
        def multipoles_summation_loop(l):               
            F_l, F_y =  XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1,C_l_off_m1,alpha_array,beta_array)
            return F_l
    
        pool = Pool(processes=ncores)

        for o, value in enumerate(pool.imap(multipoles_summation_loop, range(1,l_max,1)), 1):
            F_x =  F_x + value
        pool.close()

        forces_disp_x[steps_x//2+d]=(I_0) * F_x
        forces_disp_x[steps_x//2-d]=-(I_0) * F_x

    Pot_x = Energy_from_forces(forces_disp_x, step_size_x)
    k_x = forces_disp_x[steps_x//2]-forces_disp_x[steps_x//2-1]/(step_size_x)

    if k_x>0:
        if show_force_plots:
            X_forces_energy_plots(forces_disp_x, disp_array_x, Pot_x)
        print('ON-AXIS OPTICAL TRAPPING')
        print(' ')
        print('PARAMETERS RELATED WITH OPTICAL FORCES AND TORQUES AT THE EQUILIBRIUM POINT:')
        print(' ')
        print('Equilibrium point in Z at beam focus position on ' + str(round((eq_disp_z)*1e9,2)) + ' nm')
        print('k_z = ' + str(kz) + ' N/m')
        print(f'f_z = {np.sqrt(kz/(v_sphere * dr*1e3))/(2*np.pi)*1e-3:.1f} kHz')
        print(' ')
        print('Equilibrium point in X at beam focus position on 0 nm')
        print('k_x = ' + str(k_x) + ' N/m')
        print(f'f_x = {np.sqrt(k_x/(v_sphere * dr*1e3))/(2*np.pi)*1e-3:.1f} kHz')

        ####### Z TORQUE IN THE EQUILIBRIUM POINT OF Z #######
        if p == 0:
            C_l_off_1 = S*coefficients_Z_disp_TM(D, C_l_1, T_M_tot_Z_1, L_n_1+1, l_max)
            C_l_off_m1 = coefficients_Z_disp_TM(D, C_l_m1, T_M_tot_Z_m1, L_n_1-1, l_max)
        else:
            C_l_off_1 = coefficients_Z_disp_TM(D, C_l_1, T_M_tot_Z, m_z_p, l_max)

        T_z =  I_0 * Z_Torques_Analitic_Calculation(L_n_1,L_n_2,p,k,C_l_off_1,C_l_off_m1,alpha_array,beta_array)
        
        print(' ')
        print('Torque in Z direction = ' + str(T_z) + ' N*m')
        print(' ')
        if show_field_plots:
            ###   FIELDS CALCULATION   ###
            E_in, E_sc, E_tot, E_int, E_in2, E_sc2, E_tot2, E_int2=ON_AXIS_multipole_summation_2_planes(C_l_off_1, C_l_off_m1, alpha_array, beta_array, alpha_int_array, beta_int_array, L_n_1, L_n_2, p, nr, n2, kr, phi, theta, kr2, phi2, theta2, resolution, l_max, ncores)
            ##### PLOTS #####
            print('VISUALIZATION OF THE TOTAL ELECTROMAGNETIC FIELD INTENSITY AT THE EQUILIBRIUM POINT:')
            XZ_XY_fields_plots_TOT(VX,VY,VZ,R,R2,r_p, max_disp_limit_x,max_disp_limit_y,max_disp_limit_z, E_tot, E_int, E_tot2, E_int2)
            print('VISUALIZATION OF THE INCIDENT ELECTROMAGNETIC FIELD INTENSITY:')
            XZ_XY_fields_plots_INC(VX,VY,VZ, max_disp_limit_x,max_disp_limit_y,max_disp_limit_z, E_in, E_in2)

    else:
        print('OFF-AXIS OPTICAL TRAPPING')
        print(' ')

        #TRANSLATION MATRIX OF EQUILIBRIUM POINT IN X
        eq_true, eq_point_X, k_x=X_J_equilibrium_point_calculator(forces_disp_x, disp_array_x, step_size_x)
        
        kd_x2=eq_point_X*k
        if p == 0:
            T_M_tot_X2_1, kd_sum_1, first_it_1 = XY_disp_TM_chain(kd_x2, 1)
            T_M_tot_eq2_1=np.dot(T_M_tot_X2_1,T_M_tot_Z_1)
            C_l_off_1_EQXZ1 = S*coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq2_1, L_n_1+1, l_max)
            
            T_M_tot_X2_m1, kd_sum_m1, first_it_m1 = XY_disp_TM_chain(kd_x2, -1)
            T_M_tot_eq2_m1=np.dot(T_M_tot_X2_m1,T_M_tot_Z_m1)
            C_l_off_m1_EQXZ1 = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot_eq2_m1, L_n_2-1, l_max)

        else:
            T_M_tot_X2_1, kd_sum_1, first_it = XY_disp_TM_chain(kd_x2, p)
            T_M_tot_eq2=np.dot(T_M_tot_X2_1,T_M_tot_Z)
            C_l_off_1_EQXZ1 = coefficients_XY_disp_TM(D, C_l_1, T_M_tot_eq2, m_z_p, l_max)
            C_l_off_m1_EQXZ1=0
            T_M_tot_X2_m1=0

        d_eq_x, d_eq_z, kx_EQXZ1, kz_EQXZ1, C_l_off_1, C_l_off_m1 = Off_axis_eq_p_finder(C_l_off_1_EQXZ1, C_l_off_m1_EQXZ1, T_M_tot_X2_1, T_M_tot_X2_m1, eq_disp_z, eq_point_X, it_max, d_iter_z, d_iter_x, D, C_l_1, C_l_m1, L_n_1, L_n_2, S, l_max, k, p, m_array, alpha_array, beta_array, I_0, ncores)

        if p == 0:
            T_M_tot_Z_EQ2 = Z_disp_TM_chain(k*d_eq_z, 1)
            T_M_tot_Z_EQ2_m1 = Z_disp_TM_chain(k*d_eq_z, -1)
        else:
            T_M_tot_Z_EQ2 = Z_disp_TM_chain(k*d_eq_z, p)

        forces_disp_x2 = np.zeros((len(disp_array_x)))
         
        for d in range(1, (len(disp_array_x)//2+1)):
            disp = disp_array_x[steps_x//2+d]
            kd=disp*k
            if p == 0:
                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, 1)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2)
                C_l_off_1_X = S*coefficients_XY_disp_TM(D, C_l_1, T_M_tot, L_n_1+1, l_max)

                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, -1)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2_m1)
                C_l_off_m1_X = coefficients_XY_disp_TM(D, C_l_m1, T_M_tot, L_n_2-1, l_max)
            else:
                T_M_tot_X, kd_sum_1, first_it = XY_disp_TM_chain(kd, p)
                T_M_tot=np.dot(T_M_tot_X,T_M_tot_Z_EQ2)
                C_l_off_1_X = coefficients_XY_disp_TM(D, C_l_1, T_M_tot, m_z_p, l_max)
                C_l_off_m1_X = 0

            F_x2 = 0
            def multipoles_summation_loop(l):               
                F_l, F_y =  XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1_X,C_l_off_m1_X,alpha_array,beta_array)
                return F_l
            
            pool = Pool(processes=ncores)

            for o, value in enumerate(pool.imap(multipoles_summation_loop, range(1,l_max,1)), 1):
                F_x2 =  F_x2 + value
            pool.close()

            forces_disp_x2[steps_x//2+d]=(I_0) * F_x2
            forces_disp_x2[steps_x//2-d]=-(I_0) * F_x2

        F_matrix_x2,_ = np.meshgrid(forces_disp_x2,forces_disp_x2)
        F_matrix_tri_x2 = F_matrix_x2*np.tri(*F_matrix_x2.shape)
        Pot_x2 = np.sum(F_matrix_tri_x2,axis=1)*step_size_x

        if show_force_plots:
            X_forces_energy_plots(forces_disp_x2, disp_array_x, Pot_x2)

        print('PARAMETERS RELATED WITH OPTICAL FORCES AND TORQUES AT THE EQUILIBRIUM POINT:')
        print(' ')
        print('Equilibrium point in Z at beam focus position on ' + str(round((d_eq_z)*1e9,2)) + ' nm')
        print('k_z = ' + str(kz_EQXZ1) + ' N/m')
        print(f'f_z = {np.sqrt(kz_EQXZ1/(v_sphere * dr*1e3))/(2*np.pi)*1e-3:.1f} kHz')
        print(' ')
        print('Equilibrium point in X at beam focus position on +/- ' + str(round(np.abs(d_eq_x*1e9),2)) + ' nm')
        print('k_x = ' + str(kx_EQXZ1) + ' N/m')
        print(f'f_x = {np.sqrt(kx_EQXZ1/(v_sphere * dr*1e3))/(2*np.pi)*1e-3:.1f} kHz')
        
        Force_x = 0
        Force_y = 0
        Force_z = 0
        Torque_z = 0
        Torque_x = 0
        Torque_y = 0

        def multipoles_summation_loop(l):
            F_xl, F_yl = XY_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1,C_l_off_m1,alpha_array,beta_array)
            F_zl = Z_Forces_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1,C_l_off_m1,alpha_array,beta_array)
            Tz_l = Z_Torques_Analitic_Calculation_l(l,p,k,m_array,C_l_off_1,C_l_off_m1,alpha_array,beta_array)
            Tx_l, Ty_l = XY_Torques_Analitic_Calculation_l(l,p,k,l_max,m_array,C_l_off_1,C_l_off_m1,alpha_array,beta_array)
            return Tz_l, Tx_l, Ty_l, F_xl, F_yl, F_zl
        
        pool = Pool(processes=ncores)

        for o, values in enumerate(pool.imap(multipoles_summation_loop, range(1,l_max,1)), 1):
            Tz_l, Tx_l, Ty_l, F_xl, F_yl, F_zl = values

            Force_x =  Force_x + F_xl
            Force_y =  Force_y + F_yl
            Force_z =  Force_z + F_zl

            Torque_z =  Torque_z + Tz_l
            Torque_x =  Torque_x + Tx_l
            Torque_y =  Torque_y + Ty_l
        pool.close()

        print(' ')
        print('Force in X direction = ' + str(I_0 * Force_x) + ' N')
        print('Force in Y direction = ' + str(I_0 * Force_y) + ' N')
        print('Force in Z direction = ' + str(I_0 * Force_z) + ' N')
        print(' ')
        print('Torque in Z direction = ' + str(I_0 * Torque_z) + ' N*m')
        print('Torque in X direction = ' + str(I_0 * Torque_x) + ' N*m')
        print('Torque in Y direction = ' + str(I_0 * Torque_y) + ' N*m')
        print(' ')

        if show_field_plots:
            ###   FIELDS CALCULATION   ###
            E_in, E_sc, E_tot, E_int, E_in2, E_sc2, E_tot2, E_int2= OFF_AXIS_multipole_summation_loop_2_planes(C_l_off_1, C_l_off_m1, alpha_array, beta_array, alpha_int_array, beta_int_array, p, nr, n2, kr, phi, theta, kr2, phi2, theta2, resolution, l_max, ncores)
            ##### PLOTS #####
            print('VISUALIZATION OF THE TOTAL ELECTROMAGNETIC FIELD INTENSITY AT THE EQUILIBRIUM POINT:')
            XZ_XY_fields_plots_TOT(VX,VY,VZ,R,R2,r_p, max_disp_limit_x,max_disp_limit_y,max_disp_limit_z, E_tot, E_int, E_tot2, E_int2)
            print('VISUALIZATION OF THE INCIDENT ELECTROMAGNETIC FIELD INTENSITY:')
            XZ_XY_fields_plots_INC(VX,VY,VZ,max_disp_limit_x,max_disp_limit_y,max_disp_limit_z, E_in, E_in2)
else:
    print("NO TRAPPING IN Z")


# %%
