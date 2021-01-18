import numpy as np
from scipy.interpolate import splev, splrep
import cosmic_ray_nuclei_index
from cosmic_ray_nuclei_index import rigidity_calc, undo_log_energy, log_energy
#find the biggest minimum and smallest maximum for splining region for all isotopes
def find_interpolation_range(*argv):
    min_list=[]
    max_list=[]
    for arg in argv:
        min_list.append(arg[0])
        max_list.append(arg[-1])
    return max(min_list),min(max_list)

#energy needs to be an array of energy values in GeV. Not log energy, just energy.
def gimme_Be_rigidity_arrays(energy):
    rig_be_10=rigidity_calc(energy,10,4) # need to make sure energy is not log-energy
    rig_be_9=rigidity_calc(energy,9,4)
    rig_be_7=rigidity_calc(energy,7,4)
    return rig_be_10,rig_be_9, rig_be_7

# now try boron(10 and 11) and carbon(12 and 13)
#Seems like the spline addition is working. Now what we can do is make some of these into useful functions and compact the code
def gimme_B_rigidity_arrays(energy):
    rig_B_10=rigidity_calc(energy,10,5) # need to make sure energy is not log-energy
    rig_B_11=rigidity_calc(energy,11,5)
    return rig_B_10, rig_B_11

def gimme_C_rigidity_arrays(energy):
    rig_C_12=rigidity_calc(energy,12,6) # need to make sure energy is not log-energy
    rig_C_13=rigidity_calc(energy,13,6)
    return rig_C_12, rig_C_13

#it is assumed that x and y are understood by the user (no extra prep is done to x or y)
def spline(x,y,num_steps,minimum,maximum):
    spl = splrep(x,y)
    x_cont = np.linspace(minimum, maximum, num_steps)
    y_spline = splev(x_cont, spl)
    return x_cont, y_spline

#The process to get nuclei flux vs rigidity for simulated isotopes


#The process to get nuclei flux ratios vs rigidity for simulated isotopes
def B_C_ratio(energy,logB10_flux,logB11_flux,logC12_flux,logC13_flux,num_steps):
    #Boron
    rigB10,rigB11=gimme_B_rigidity_arrays(energy)
    rigB10_logged=log_energy(rigB10)
    rigB11_logged=log_energy(rigB11)
    #Carbon
    rigC12,rigC13=gimme_C_rigidity_arrays(energy)
    rigC12_logged=log_energy(rigC12)
    rigC13_logged=log_energy(rigC13)
    
    BCmin_R,BCmax_R=find_interpolation_range(rigB10_logged,rigB11_logged,rigC12_logged,rigC13_logged)
    rigB10_spline_logged, B10_flux_spline_logged=spline(rigB10_logged,logB10_flux, num_steps,BCmin_R,BCmax_R)
    rigB11_spline_logged, B11_flux_spline_logged=spline(rigB11_logged,logB11_flux, num_steps,BCmin_R,BCmax_R)
    rigC12_spline_logged, C12_flux_spline_logged=spline(rigC12_logged,logC12_flux, num_steps,BCmin_R,BCmax_R)
    rigC13_spline_logged, C13_flux_spline_logged=spline(rigC13_logged,logC13_flux, num_steps,BCmin_R,BCmax_R)
    # undo the logarithm
    rigB10_spline=undo_log_energy(rigB10_spline_logged)
    rigB11_spline=undo_log_energy(rigB11_spline_logged)
    B10_flux_spline=undo_log_energy(B10_flux_spline_logged)
    B11_flux_spline=undo_log_energy(B11_flux_spline_logged)
    #add the fluxes together
    B_total_flux_spline=np.add(B10_flux_spline,B11_flux_spline)
    # undo the logarithm
    rigC12_spline=undo_log_energy(rigC12_spline_logged)
    rigC13_spline=undo_log_energy(rigC13_spline_logged)
    C12_flux_spline=undo_log_energy(C12_flux_spline_logged)
    C13_flux_spline=undo_log_energy(C13_flux_spline_logged)
    #add the fluxes together
    C_total_flux_spline=np.add(C12_flux_spline,C13_flux_spline)
    B_C_ratio_spline=np.divide(B_total_flux_spline,C_total_flux_spline)
    return rigB10_spline,B_C_ratio_spline