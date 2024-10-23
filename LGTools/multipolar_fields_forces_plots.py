# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def Z_forces_energy_plots(forces_disp_z, disp_array_z, Pot_z):
    """
    Plots the force_vs_beam_displacement overlapped with the energy_vs_beam_displacement graphs for z axis.
    """
    
    plt.figure(figsize=(4,2.67), dpi=500, facecolor="w")
    color_f='forestgreen'
    ax1 = plt.axes([0.15,0.15,0.75,0.75])
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Beam focus position in Z($\mu$m)', fontsize=14)
    ax1.set_ylabel('Force in Z(pN)', fontsize=14, color=color_f)
    ax1.plot(disp_array_z*1e6, forces_disp_z*1e12, color=color_f)
    ax1.axhline(0, linestyle = ':', color=color_f, linewidth = 1)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=color_f)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.margins(x=0)

    ax2 = ax1.twinx() 
    color_e = 'tab:orange'
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.set_ylabel('Energy in Z(aJ)', color=color_e, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(disp_array_z*1e6, (Pot_z)*1e18, color=color_e, linestyle='dashed')
    ax2.tick_params(axis='y', labelsize=12, labelcolor=color_e)
    ax2.margins(x=0)
    # plt.savefig('Force_Energy_IMGs/Fz_Ez_W'+str(wl)+'_R'+str(r_p)+'_nr'+str(nr)+'_P='+str(p)+'_L1='+str(L_n_1)+'.svg', transparent=True)
    plt.show()

def X_forces_energy_plots(forces_disp_x, disp_array_x, Pot_x):
    """
    Plots the force_vs_beam_displacement overlapped with the energy_vs_beam_displacement graphs for x axis.
    """

    plt.figure(figsize=(4,2.67), dpi=500)
    color_f='b'
    ax1 = plt.axes([0.15,0.15,0.75,0.75])
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Beam focus position in X($\mu$m)', fontsize=14)
    ax1.set_ylabel('Force in X(pN)', fontsize=14, color=color_f)
    ax1.plot(disp_array_x*1e6, forces_disp_x*1e12, color=color_f)
    ax1.axhline(0, linestyle = ':', color=color_f, linewidth = 1)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=color_f)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.margins(x=0)

    ax2 = ax1.twinx()
    color_e = 'm'
    ax2.set_ylabel('Energy in X(aJ)', color=color_e, fontsize=14)  
    ax2.plot(disp_array_x*1e6, (Pot_x)*1e18, color=color_e, linestyle='dashed')
    ax2.tick_params(axis='y', labelsize=12, labelcolor=color_e)
    ax2.margins(x=0)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    # plt.savefig('Force_Energy_IMGs/Fx_Ex_W'+str(wl)+'_R'+str(r_p)+'_nr'+str(nr)+'_P='+str(p)+'_L1='+str(L_n_1)+'.svg', transparent=True)
    plt.show()

def XZ_XY_fields_plots_INC(VX,VY,VZ,size_x, size_y, size_z, E_in_xz, E_in_xy):
    """
    Plots the incident electromagnetic fields of the system for XZ and XY planes into a colormap.
    """
    
    sx = 6
    sy = sx*(size_y+size_z)/size_x*0.85
    
    fig = plt.figure(figsize=(sx,sy), dpi=300)
    fig.subplots_adjust(hspace=0)
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_ylabel("Z axis ("+u"\u03bcm)", fontsize=14)
    pc = ax1.pcolormesh(VX*1e6, VZ*1e6, (E_in_xz),cmap='jet',shading='auto')
    ax1.get_xaxis().set_visible(False)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="7%", pad="5%")
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label(label='(V$^{2}$/m$^{2}$)',size=14, labelpad=-10)
    cb.ax.xaxis.set_label_position("top")
    cb.set_ticks([np.nanmax(E_in_xz),np.nanmin(E_in_xz)])
    cb.set_ticklabels(["Max","Min"])
    cb.ax.tick_params(labelsize=14)
    cb.ax.xaxis.set_ticks_position("top")
    ax1.tick_params(axis='y', labelsize=12)
    ax1.text((0.55*(size_x)*1e6), -(0.8*(size_z)*1e6), 'Y=0', fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
    ax1.set_yticks([-1,0,1])
    ax1.margins(x=0)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(1,2,1, sharex = ax1)
    ax2.set_xlabel("X axis ("+u"\u03bcm)", fontsize=14)
    ax2.set_ylabel("Y axis ("+u"\u03bcm)", fontsize=14)
    pc = ax2.pcolormesh(VX*1e6, VY*1e6, (E_in_xy),cmap='jet',shading='auto')
    ax2.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.margins(x=0)
    ax2.set_aspect('equal')
    ax2.text((0.55*(size_x)*1e6), -(0.75*(size_y)*1e6), 'Z=0', fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
    fig.tight_layout()
    plt.subplots_adjust(hspace=-0.1)
    #plt.savefig(dirname+'/Tot_field_W'+str(wl)+'_R'+str(r_p)+'_nr'+str(nr)+'_P='+str(p)+'_L1='+str(L_n_1)+'.png', transparent=True)
    plt.show()

def XZ_XY_fields_plots_TOT(VX,VY,VZ,R,R2,r_p, size_x, size_y, size_z, E_tot_xz, E_int_xz, E_tot_xy, E_int_xy):
    """
    Plots the total electromagnetic fields of the optical for XZ and XY planes into a colormap.
    """

    E_tot_xz[np.where(R <= r_p)] = 0
    E_int_xz[np.where(R > r_p*0.95)] = np.nan
    E_int_xz[np.where(R > r_p)] = 0
    E_tot_xz=E_tot_xz+E_int_xz

    E_tot_xy[np.where(R2 <= r_p)] = 0
    E_int_xy[np.where(R2 > r_p*0.95)] = np.nan
    E_int_xy[np.where(R2 > r_p)] = 0
    E_tot_xy=E_tot_xy+E_int_xy

    sx = 6
    sy = sx*(size_y+size_z)/size_x*0.85
    
    fig = plt.figure(figsize=(sx,sy), dpi=300)
    fig.subplots_adjust(hspace=0)
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_ylabel("Z axis ("+u"\u03bcm)", fontsize=14)
    pc = ax1.pcolormesh(VX*1e6, VZ*1e6, (E_tot_xz),cmap='jet',shading='auto')
    ax1.get_xaxis().set_visible(False)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="7%", pad="5%")
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label(label='(V$^{2}$/m$^{2}$)',size=14, labelpad=-10)
    cb.ax.xaxis.set_label_position("top")
    cb.set_ticks([np.nanmax(E_tot_xz),np.nanmin(E_tot_xz)])
    cb.set_ticklabels(["Max","Min"])
    cb.ax.tick_params(labelsize=14)
    cb.ax.xaxis.set_ticks_position("top")
    ax1.tick_params(axis='y', labelsize=12)
    ax1.text((0.55*(size_x)*1e6), -(0.8*(size_z)*1e6), 'Y=0', fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
    ax1.set_yticks([-1,0,1])
    ax1.margins(x=0)
    ax1.set_aspect('equal')

    E_tot_xy2=E_tot_xy#np.concatenate((E_tot_xy[:,:len(VX)//2], np.fliplr(E_tot_xy[:,:len(VX)//2])), axis=1)
    ax2 = fig.add_subplot(1,2,1, sharex = ax1)
    ax2.set_xlabel("X axis ("+u"\u03bcm)", fontsize=14)
    ax2.set_ylabel("Y axis ("+u"\u03bcm)", fontsize=14)
    pc = ax2.pcolormesh(VX*1e6, VY*1e6, (E_tot_xy2),cmap='jet',shading='auto')
    ax2.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.margins(x=0)
    ax2.set_aspect('equal')
    ax2.text((0.55*(size_x)*1e6), -(0.75*(size_y)*1e6), 'Z=0', fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
    fig.tight_layout()
    plt.subplots_adjust(hspace=-0.1)
    #plt.savefig(dirname+'/Tot_field_W'+str(wl)+'_R'+str(r_p)+'_nr'+str(nr)+'_P='+str(p)+'_L1='+str(L_n_1)+'.png', transparent=True)
    plt.show()

def On_axis_multipolar_coefficients_plot(p, D_C_l_1, D_C_l_m1, alpha_array, beta_array, l_vis_lim=15):
    """
    Plots of the multipolar content of the incident beam, the scattering coefficients and the coupling between them into a bar plot. 
    """

    l_array=np.arange(1,l_vis_lim+1,1)
    sca_C = np.abs((alpha_array + beta_array)**2)
    if p==0:
        C_l_1_s = np.abs(D_C_l_1/np.sqrt(2))**2+np.abs(D_C_l_m1/np.sqrt(2))**2
        C_tot = np.abs((D_C_l_1*alpha_array/np.sqrt(2) + D_C_l_1*beta_array/np.sqrt(2))**2)+np.abs((D_C_l_m1*alpha_array/np.sqrt(2) + D_C_l_m1*beta_array/np.sqrt(2))**2)
    else:
        C_l_1_s = np.abs((D_C_l_1)**2)
        C_tot = np.abs((D_C_l_1*alpha_array + D_C_l_1*beta_array)**2)

    width = 0.3  # the width of the bars

    # fig, ax = plt.subplots(figsize=(5,1.7), dpi=300)
    # rects2 = ax.bar(l_array, C_l_1_s[1:l_vis_lim+1], width*2, label='Incident BSC', color='lightskyblue', edgecolor='blue', hatch="//")
    # ax.set_ylabel('Coefficient weight')
    # ax.set_xlabel('Multipolar order j')
    # # ax.set_yticks(np.array([0.1,0.3,0.5]))
    # ax.set_xticks(l_array)
    # plt.margins(x=0.015)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), labelspacing=0.2, fancybox=True, shadow=False, ncol=3).legendPatch.set_edgecolor("dimgrey")
    # # plt.savefig(dirname+'/Coeff_'+'_L1='+str(L_n_1)+'.svg')
    # plt.plot

    fig, ax = plt.subplots(figsize=(5,1.7), dpi=300)
    rects2 = ax.bar(l_array - width+0.05, C_l_1_s[1:l_vis_lim+1], width, label='Incident BSC', color='lightskyblue', edgecolor='blue', hatch="//")
    rects1 = ax.bar(l_array , sca_C[1:l_vis_lim+1], width, label='Scattering Coeff.', color='palegreen', edgecolor='darkgreen', hatch="..")
    rects3 = ax.bar(l_array + width-0.05, C_tot[1:l_vis_lim+1], width, label='Coupling', color='orangered', edgecolor='darkred')
    ax.set_ylabel('Coefficient weight')
    ax.set_xlabel('Multipolar order j')
    ax.set_xticks(l_array)
    plt.margins(x=0.015)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), labelspacing=0.2, fancybox=True, shadow=False, ncol=3).legendPatch.set_edgecolor("dimgrey")
    # plt.savefig(dirname+'/Coeff_'+'_L1='+str(L_n_1)+'.svg')
    plt.title('ON-FOCUS CONFIGURATION', y=1.25)
    plt.show()