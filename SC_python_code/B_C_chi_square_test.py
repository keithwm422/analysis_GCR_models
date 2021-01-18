# This code is a combination of a bunch of other tests with jupyter notebooks. 
# Author: Keith McBride, Jan 2021
# Run this to compare the AMS data and ALOT OF GALPROP models 
# using the chi-square test statistic. 
# The resulting output will be the B/C AMS data and the best fit model, as a function of rigidity. 


# IMPORT REGULAR PACKAGES
import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt 

#IMPORT CUSTOM FUNCTIONS AND CONSTANTS
from get_ams_data_functions import *
import cosmic_ray_nuclei_index
from cosmic_ray_nuclei_index import rigidity_calc, undo_log_energy, log_energy
from get_splines import *
from get_residuals_chi_square import *
from open_tarfile import get_fluxes_from_files
# Example function for testing this procedure before expanding to many ratios
def run_chi_square_test(seq):
    # FIRST THE DATA
    numerator='B'  # numerator of the nuclei ratio used in the chi_square calculation (needs to also be AMS_DATA file first character)
    denominator='C'  # same as numerator but denominator for the nuclei ratio
    path='/home/mcbride.342/galprop_sims/AMS_Data/Ratios/' # path for the AMS DATA files
    df=read_in_data(numerator,denominator,path) #read in the AMS data for the given ratio as a pandas dataframe
    rigidity,rigidity_binsize,ratio,ratio_errors=make_energies_and_errors(df,numerator,denominator)  # just have this function do cool stuff and give you a bunch of arrays
    
    #     IF YOU WANT TO PLOT JUST THE DATA (there is a different file for that though)
    #do you want y-axis log-scaled? (1=yes)
    #log_y=1
    #make_plot_of_data(numerator,denominator,rigidity,ratio,rigidity_binsize,ratio_errors,log_y)
    #print("success")
    #     END PLOTTING THE DATA

    # SECOND THE SIMULATION SETS
    # make arrays of the energy axis for all isotope fluxes (really kinetic energy)
    # get energy axis and change to GeV (undo the logarithm to put in actual energy units)
    energy=np.arange(2,9,0.304347391792257)
    energy=undo_log_energy(energy)
    energy=np.true_divide(energy,10**3)
    print("ENERGY ARRAY: ")
    print(energy)
    diffusion_number=3 #(set to 1 to 20)
    #first arg is not used currently for the following function (so it can be anything really)
    fluxes_per_element_full=get_fluxes_from_files(1, diffusion_number)  # to store the loaded fluxes from GALPROP sims
    
    # ALL DATA AND FLUXES HAVE BEEN LOADED
    # ADJUST UNITS OF THE SIMULATION ARRAYS
    # the energy array is the same for all isotopes
    #need to pass the log fluxes found from the models above like so:
    model=0# loaded 0,19 models above so pick one (increasing halo size number here)
    num_steps=200 # number of points in the spline array when interpolating
    # access the actual flux from the loaded simulation sets and then log them
    logB10_flux=log_energy(fluxes_per_element_full[model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron10_loc)])
    logB11_flux=log_energy(fluxes_per_element_full[model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron11_loc)])
    logC12_flux=log_energy(fluxes_per_element_full[model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon12_loc)])
    logC13_flux=log_energy(fluxes_per_element_full[model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon13_loc)])
    # now call function that will spline all those fluxes and the energy axis to a common rigidity range among all the isotopes. 
    rigC13_spline,B_C_ratio_spline=B_C_ratio(energy,logB10_flux,logB11_flux,logC12_flux,logC13_flux,num_steps)
    
    # NOW CAN CALCULATE THE CHI-SQUARE and get residuals
    residuals,chi_square=calculate_chi_square(rigidity,ratio,rigC13_spline, B_C_ratio_spline) 
    #print(chi_square)
    print(f'about to plot model with chi-square {chi_square}, from model {fluxes_per_element_full[model][-1]}')
    
    # plot the model chosen and the AMS data
    plt.figure(figsize=(12,12))
    fnt=14
    x1=0.9*rigC13_spline[0]
    x2=2*rigC13_spline[-1]
 
    plt.plot(rigC13_spline, B_C_ratio_spline,'--',label="Ratio spline")
    plt.errorbar(rigidity,ratio,xerr=rigidity_binsize,yerr=ratio_errors,fmt='o',label="AMS")

    #plt.plot(energy,be_10_be_9_5,'-o',label="L=5")
    plt.xscale("log")
    plt.xlabel("Rigidity (GV)",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    plt.ylabel("Flux ratio",fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    plt.legend(loc='upper right')
    plt.title("Example, Boron and Carbon Ratio", fontsize=fnt)
    plt.savefig("Boron_carbon_fluxratio_rigidity_splines" + str(model) + ".png")
