# THIS COULD BE DONE SO MUCH BETTER
# IE edit the ratios to take lists of numpy arrays that it then parses to determine how many in each list (numerator and denominator of flux)
# then matches their rigidities and splines them on that rigidity
# next calcs the ratio at those rigidities only

import numpy as np
from scipy.interpolate import splev, splrep
import cosmic_ray_nuclei_index
from cosmic_ray_nuclei_index import rigidity_calc, undo_log_energy, log_energy


#find the biggest minimum and smallest maximum for splining region for all isotopes
# which will be a comparison of the first values between all isotopes for minimum and the last values for maximum
def find_interpolation_range(*argv):
    min_list=[]
    max_list=[]
    for arg in argv:
        min_list.append(arg[0])
        max_list.append(arg[-1])
    return max(min_list),min(max_list)

def gimme_rigidity(energy,mass,charge):
    return rigidity_calc(energy,mass,charge)

#energy needs to be an array of energy values in GeV/n. Not log energy, just energy.
def gimme_Li_rigidity_arrays(energy):
    rig_li_6=rigidity_calc(energy,6,3) # need to make sure energy is not log-energy
    rig_li_7=rigidity_calc(energy,7,3)
    return rig_li_6, rig_li_7

#energy needs to be an array of energy values in GeV/n. Not log energy, just energy.
def gimme_Be_rigidity_arrays(energy):
    rig_be_10=rigidity_calc(energy,10,4) # need to make sure energy is not log-energy
    rig_be_9=rigidity_calc(energy,9,4)
    rig_be_7=rigidity_calc(energy,7,4)
    return rig_be_7,rig_be_9, rig_be_10

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

def gimme_O_rigidity_arrays(energy):
    rig_O_16=rigidity_calc(energy,16,8) # need to make sure energy is not log-energy
    rig_O_17=rigidity_calc(energy,17,8)
    rig_O_18=rigidity_calc(energy,18,8)
    return rig_O_16, rig_O_17, rig_O_18

#it is assumed that x and y are understood by the user (no extra prep is done to x or y)
def spline(x,y,num_steps,minimum,maximum):
    spl = splrep(x,y)
    x_cont = np.linspace(minimum, maximum, num_steps)
    y_spline = splev(x_cont, spl)
    return x_cont, y_spline

# This function is passed lists of arrays for the x and y values of both the denominator and numerator isotopes that the ratio will be found for
def calc_ratio_spline(numerator_energy,numerator_flux,denominator_energy,denominator_flux,spline_min_R,spline_max_R,num_steps):
    num_flux=[]
    denom_flux=[]
    numerator_isotopes=len(numerator_flux)
    denominator_isotopes=len(denominator_flux)
    i=0
    while i<numerator_isotopes:
        x_vals,y_vals=spline(numerator_energy[i],numerator_flux[i], num_steps,spline_min_R,spline_max_R)
        y_vals=undo_log_energy(y_vals)
        y_vals=np.array(y_vals)
        num_flux.append(y_vals)
        i+=1
    j=0
    while j<denominator_isotopes:
        x_vals,y_vals=spline(denominator_energy[j],denominator_flux[j], num_steps,spline_min_R,spline_max_R)
        y_vals=undo_log_energy(y_vals)        
        y_vals=np.array(y_vals)
        denom_flux.append(y_vals)
        j+=1
        #for k in num_flux:
        #num_total_flux_spline=np.add()
    #num_flux=np.array(num_flux)
    common_x=np.array(undo_log_energy(x_vals))
    num_total_flux=num_flux[0].copy()
    i=1
    while i<numerator_isotopes:
        num_total_flux=np.add(num_total_flux,num_flux[i])
        i+=1
    denom_total_flux=denom_flux[0].copy()
    j=1
    while j<denominator_isotopes:
        denom_total_flux=np.add(denom_total_flux,denom_flux[j])
        j+=1
    # now divide
    ratio=np.divide(num_total_flux,denom_total_flux)
    return ratio, common_x



#The process to get nuclei flux ratios vs rigidity for simulated isotopes, make sure energy is array that is not logged and is in units of GeV
def B_C_ratio(energy,logB10_flux,logB11_flux,logC12_flux,logC13_flux,num_steps):
    # give it energy and it knows (one per ratio)
    #Boron
    rigB10,rigB11=gimme_B_rigidity_arrays(energy)
    rigB10_logged=log_energy(rigB10)
    rigB11_logged=log_energy(rigB11)
    #Carbon
    rigC12,rigC13=gimme_C_rigidity_arrays(energy)
    rigC12_logged=log_energy(rigC12)
    rigC13_logged=log_energy(rigC13)
    numerator_x_list=[rigB10_logged,rigB11_logged]
    numerator_y_list=[logB10_flux,logB11_flux]
    denominator_x_list=[rigC12_logged,rigC13_logged]
    denominator_y_list=[logC12_flux,logC13_flux]
    spline_min_R,spline_max_R=find_interpolation_range(rigB10_logged,rigB11_logged,rigC12_logged,rigC13_logged)
    ratio, rig_values=calc_ratio_spline(numerator_x_list,numerator_y_list,denominator_x_list,denominator_y_list,spline_min_R,spline_max_R,num_steps)
    return rig_values, ratio

#The process to get nuclei flux ratios vs rigidity for simulated isotopes, make sure energy is array that is not logged and is in units of GeV
def B_O_ratio(energy,logB10_flux,logB11_flux,logC12_flux,logC13_flux,num_steps):
    # give it energy and it knows (one per ratio)
    #Boron
    rigB10,rigB11=gimme_B_rigidity_arrays(energy)
    rigB10_logged=log_energy(rigB10)
    rigB11_logged=log_energy(rigB11)
    #Carbon
    rigC12,rigC13=gimme_C_rigidity_arrays(energy)
    rigC12_logged=log_energy(rigC12)
    rigC13_logged=log_energy(rigC13)
    numerator_x_list=[rigB10_logged,rigB11_logged]
    numerator_y_list=[logB10_flux,logB11_flux]
    denominator_x_list=[rigC12_logged,rigC13_logged]
    denominator_y_list=[logC12_flux,logC13_flux]
    spline_min_R,spline_max_R=find_interpolation_range(rigB10_logged,rigB11_logged,rigC12_logged,rigC13_logged)
    ratio, rig_values=calc_ratio_spline(numerator_x_list,numerator_y_list,denominator_x_list,denominator_y_list,spline_min_R,spline_max_R,num_steps)
    return rig_values, ratio



# need a method like this for each kind of ratio? geezzz

#The process to get nuclei flux ratios vs rigidity for simulated isotopes, make sure energy is array that is not logged and is in units of GeV
def B_O_ratio_original(energy,logB10_flux,logB11_flux,logO16_flux,logO17_flux,logO18_flux,num_steps):
    #Boron
    rigB10,rigB11=gimme_B_rigidity_arrays(energy)
    rigB10_logged=log_energy(rigB10)
    rigB11_logged=log_energy(rigB11)
    #Oxygen
    rigO16,rigO17,rigO18=gimme_O_rigidity_arrays(energy)
    rigO16_logged=log_energy(rigO16)
    rigO17_logged=log_energy(rigO17)
    rigO18_logged=log_energy(rigO18)
    
    BOmin_R,BOmax_R=find_interpolation_range(rigB10_logged,rigB11_logged,rigO16_logged,rigO17_logged,rigO18_logged)
    rigB10_spline_logged, B10_flux_spline_logged=spline(rigB10_logged,logB10_flux, num_steps,BOmin_R,BOmax_R)
    rigB11_spline_logged, B11_flux_spline_logged=spline(rigB11_logged,logB11_flux, num_steps,BOmin_R,BOmax_R)
    rigO16_spline_logged, O16_flux_spline_logged=spline(rigO16_logged,logO16_flux, num_steps,BOmin_R,BOmax_R)
    rigO17_spline_logged, O17_flux_spline_logged=spline(rigO17_logged,logO17_flux, num_steps,BOmin_R,BOmax_R)
    rigO18_spline_logged, O18_flux_spline_logged=spline(rigO18_logged,logO18_flux, num_steps,BOmin_R,BOmax_R)
    # undo the logarithm
    rigB10_spline=undo_log_energy(rigB10_spline_logged)
    rigB11_spline=undo_log_energy(rigB11_spline_logged)
    B10_flux_spline=undo_log_energy(B10_flux_spline_logged)
    B11_flux_spline=undo_log_energy(B11_flux_spline_logged)
    #add the fluxes together
    B_total_flux_spline=np.add(B10_flux_spline,B11_flux_spline)
    # undo the logarithm
    rigO16_spline=undo_log_energy(rigO16_spline_logged)
    rigO17_spline=undo_log_energy(rigO17_spline_logged)
    rigO18_spline=undo_log_energy(rigO18_spline_logged)
    O16_flux_spline=undo_log_energy(O16_flux_spline_logged)
    O17_flux_spline=undo_log_energy(O17_flux_spline_logged)
    O18_flux_spline=undo_log_energy(O18_flux_spline_logged)

    #add the fluxes together
    O_total_flux_spline=np.add(O16_flux_spline,O17_flux_spline)
    O_total_flux_spline=np.add(O_total_flux_spline,O18_flux_spline)
    B_O_ratio_spline=np.divide(B_total_flux_spline,O_total_flux_spline)
    return rigB10_spline,B_O_ratio_spline

# TRY THE FORCE_FIELD
#try the force-field approx
#companion function, EK is a column array
def force_field_factor(Z,A,phi,EK):
    proton_mass= 0.938 #In GeV
    PHI=Z*phi/A #really this is (Z e phi)/A but e=1 in the units I want
    print(PHI)
    EK_shifted=EK.copy()
    EK_shifted=EK_shifted-PHI
    EK_sum=EK_shifted.copy()+2*A*proton_mass
    EK_num=EK_shifted*EK_sum
    EK_orig=EK.copy()
    EK_sum_2=EK_orig+2*A*proton_mass
    EK_denom=EK_orig*EK_sum_2
    return np.true_divide(EK_num,EK_denom), EK_shifted
# this function returns the spectra at Earth of the isotope provided (charge, Z, and mass, A) and its shifted energy values
#given the solar modulation potential phi and the Kinetic energy of the particle
#phi needs to be in units of GV not MV
#EK in GeV (not per nucleon!)
def force_field_approx(LIS,Z,A,phi,EK):
    factor,EK_shifted=force_field_factor(Z,A,phi,EK)
    return LIS*(factor),EK_shifted

#The process to get modulated nuclei flux ratios vs rigidity for simulated isotopes,
# make sure energy is array that is not logged and is in units of GeV/n
# make the energy a list and the flux a list of arrays
#mass is same length as other lists
#charge is same length as other lists
#returns one array of each: ratio and common rigidity
#fluxes should already be logged
def B_C_ratio_modulated(energy,flux,mass,charge,num_steps):
    i=0
    rigidities=[]
    while i <len(energy):
        rigidities.append(np.array(log_energy(gimme_rigidity(energy[i],mass[i],charge[i]))))
        i+=1
    numerator_x_list=[rigidities[0],rigidities[1]]
    numerator_y_list=[flux[0],flux[1]]
    denominator_x_list=[rigidities[2],rigidities[3]]
    denominator_y_list=[flux[2],flux[3]] 
    spline_min_R,spline_max_R=find_interpolation_range(rigidities[0],rigidities[1],rigidities[2],rigidities[3])
    ratio,rig_values=calc_ratio_spline(numerator_x_list,numerator_y_list,
        denominator_x_list,denominator_y_list,spline_min_R,spline_max_R,num_steps)
    return rig_values, ratio
