# This code is a combination of a bunch of other tests with jupyter notebooks. 
# Author: Keith McBride, Jan 2021
# Run this to compare the AMS data and ALOT OF GALPROP models 
# using the chi-square test statistic. 
# The resulting output will be the B/C AMS data and the best fit model, as a function of rigidity. 
# this file builds on the content of B_C_chi_square_test.py

# Set up matplotlib and use a nicer set of plot parameters
# IMPORT REGULAR PACKAGES
import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.colors import LogNorm

#IMPORT CUSTOM FUNCTIONS AND CONSTANTS
from get_ams_data_functions import *
import cosmic_ray_nuclei_index
from cosmic_ray_nuclei_index import rigidity_calc, undo_log_energy, log_energy
from get_splines import *
from get_residuals_chi_square import *
from open_tarfile import get_fluxes_from_files
# Example function for testing this procedure before expanding to many ratios
def run_chi_square_test(seq,num_spline_steps,cutoff):
    # FIRST THE DATA
    MAX_DIFFUSION=20
    numerator='B'  # numerator of the nuclei ratio used in the chi_square calculation (needs to also be AMS_DATA file first character)
    denominator='C'  # same as numerator but denominator for the nuclei ratio
    df=read_in_data(numerator,denominator) #read in the AMS data for the given ratio as a pandas dataframe
    rigidity,rigidity_binsize,ratio,ratio_errors=make_energies_and_errors(df,numerator,denominator)  # just have this function do cool stuff and give you a bunch of arrays   
    print(f'Num data points in AMS data: {len(ratio)}')
    #     IF YOU WANT TO PLOT JUST THE DATA run the code below  #
    #do you want y-axis log-scaled? (1=yes)
    #log_y=1
    #make_plot_of_data(numerator,denominator,rigidity,ratio,rigidity_binsize,ratio_errors,log_y)
    #print("success")
    #     END PLOTTING THE DATA    #
    ################################
    # SECOND THE SIMULATION SETS
    # make arrays of the energy axis for all isotope fluxes (really kinetic energy)
    # get energy axis and change to GeV (undo the logarithm to put in actual energy units)
    energy=np.arange(2,9,0.304347391792257) # construct the log energy array (these are just the powers of base 10) that matches galprop fits files
    energy=undo_log_energy(energy) # undo the logarithm so now these are MeV/nucleon energies 
    energy=np.true_divide(energy,10**3) # change to GeV/nucleon
    #print(f'ENERGY ARRAY: {energy}')
    # BIG LOOP TO GET ALL 400 models in sets of 20.
    chi_square_array=[]
    chi_square_nparray=np.empty([20,20])
    model_name_full=np.empty([20,20],dtype="S10")
    stats_full=np.empty([20,20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
    stats_masked_full=np.empty([20,20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
    diffusion_number=1 #(set to [1,20])
    while diffusion_number<MAX_DIFFUSION+1:
        chi_square_temp=[]  # reset this one at each iteration
        #first arg is not used currently for the following function (so it can be anything really)
        fluxes_per_element_per_diffusion=get_fluxes_from_files(1, diffusion_number)  # to store the loaded fluxes from GALPROP sims
        # ALL DATA AND FLUXES HAVE BEEN LOADED
        # ADJUST UNITS OF THE SIMULATION ARRAYS
        # the energy array is the same for all isotopes
        #need to pass the log fluxes found from the models above like so:
        halo_model=0 # halo size is weird, since it reads them in alphebatical order. model=0 is L=10, model=1 L=11, .. model=10 L=1, model=11 L=20, model=12 L=2 ... model=19 L=9 
        #ratios_splined_per_diffusion=np.empty([20,])
        model_name=np.empty([20,],dtype="S10")
        chi_square_array_per_diffusion=np.empty([20,])
        stats_per_diffusion=np.empty([20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
        stats_masked_per_diffusion=np.empty([20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
        while halo_model<20:
            # access the actual flux from the loaded simulation sets and then log them
            logB10_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron10_loc)])
            logB11_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron11_loc)])
            logC12_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon12_loc)])
            logC13_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon13_loc)])
            # now call function that will spline all those fluxes and the energy axis to a common rigidity range among all the isotopes. 
            rigC13_spline,B_C_ratio_spline=B_C_ratio(energy,logB10_flux,logB11_flux,logC12_flux,logC13_flux,num_spline_steps)
            # NOW CAN CALCULATE THE CHI-SQUARE and get residuals
            # if there is a cutoff in rigidity (use data at this value and above), then use the following
            residuals,chi_square,stats_obj,stats_obj_masked,rigidity_masked,ratio_masked,model_x,model_y,model_x_masked,model_y_masked=calculate_chi_square(rigidity,ratio,rigC13_spline, B_C_ratio_spline,cutoff)
            # these quantities need to be saved into huge arrays, but also apply the formatting inline here
            # formatting is that the loaded files are done in alphabetical order, so the numbers are not in the correct order. 
            # the print statement below shows that the first model read in is L=10, L=11, ... L=19 then L=1, L=20, then L=2, L=3....L=9
            print(f'chi-square {chi_square}, from model {fluxes_per_element_per_diffusion[halo_model][-1]}')
            if halo_model<=9: #this is L=10 so set it 9th position in array, ==1 L=11 set it to 10 pos in array... j==9 L=19 set it to 18 pos in array
                chi_square_array_per_diffusion[halo_model+9]=chi_square
                model_name[halo_model+9]=fluxes_per_element_per_diffusion[halo_model][-1]
                #ratios_splined_per_diffusion[halo_model+9]=B_C_ratio_spline
                stats_per_diffusion[halo_model+9]=stats_obj
                stats_masked_per_diffusion[halo_model+9]=stats_obj_masked
            elif halo_model==10: # this is L=1 so set it to zeroth pos
                chi_square_array_per_diffusion[0]=chi_square
                model_name[0]=fluxes_per_element_per_diffusion[halo_model][-1]
                #ratios_splined_per_diffusion[0]=B_C_ratio_spline
                stats_per_diffusion[0]=stats_obj
                stats_masked_per_diffusion[0]=stats_obj_masked
            elif halo_model==11: #this is L=20 so set it to last pos
                chi_square_array_per_diffusion[19]=chi_square
                model_name[19]=fluxes_per_element_per_diffusion[halo_model][-1]
                #ratios_splined_per_diffusion[19]=B_C_ratio_spline
                stats_per_diffusion[19]=stats_obj
                stats_masked_per_diffusion[19]=stats_obj_masked
            elif halo_model>=12:  # these are j>=12 where j==12 is L=2 so set to 1 pos in array, j==13 set it 2 in array
                chi_square_array_per_diffusion[halo_model-11]=chi_square
                model_name[halo_model-11]=fluxes_per_element_per_diffusion[halo_model][-1]
                #ratios_splined_per_diffusion[halo_model-11]=B_C_ratio_spline
                stats_per_diffusion[halo_model-11]=stats_obj
                stats_masked_per_diffusion[halo_model-11]=stats_obj_masked
            # other variables can be added as needed (for plotting or further investigation)
            halo_model+=1
        chi_square_nparray[diffusion_number-1]=chi_square_array_per_diffusion
        chi_square_array.append(chi_square_array_per_diffusion)
        model_name_full[diffusion_number-1]=model_name
        stats_full[diffusion_number-1]=stats_per_diffusion
        stats_masked_full[diffusion_number-1]=stats_masked_per_diffusion
        diffusion_number+=1
    #something else
    #with open("chisquarefile.txt", 'w') as file:
        #for row in chi_square_array:
            #s = " ".join(map(str, row))
            #file.write(s+'\n')

    #print('{chi_square_array[0][0]})
    print("Saved file")
    return chi_square_array,chi_square_nparray,model_name_full,stats_full,stats_masked_full

def make_colormap(seq):
    # quantities for the analysis
    num_spline_steps=2000
    kinetic_energy_cutoff=20 # GeV/nucleon
    cutoff=rigidity_calc(kinetic_energy_cutoff,13,6) # the other two args are, in order, mass (num nucleons) and charge (Z).
    chi_squares_full, chi_squares_full_np,models,stats_full, stats_masked_full=run_chi_square_test(1,num_spline_steps,cutoff)
    print(f'chi square full num rows {len(chi_squares_full)}')
    print(f'chi square full num columns {len(chi_squares_full[0])}')
    L_values=np.arange(1,20,20)
    D_values=np.arange(1,20,20)
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
    matplotlib.rc('font', **font)
    # We can visualize with a heatmap
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10,10))
    #cax=ax.matshow(sum_squares_full, cmap = 'Greens')
    cax=ax.matshow(chi_squares_full, cmap='plasma', norm=LogNorm(vmin=0.1, vmax=1000))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([i for i in range(20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=14)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    fig.colorbar(cax)
    ax.set_yticks([i for i in range(20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=14)
    #ax.set_xticklabels([str(i) for i in Categories], rotation = 90,fontsize=14)
    ax.set_ylabel("Diffusion Coefficient "r'$10^{28}cm^{2}s^{-1}$', fontsize = 14)
    ax.set_xlabel("Halo Size (kpc)", fontsize = 14)
    #plt.tight_layout()
    plt.title("Chi Square for B/C", y=1.08)
    print("about to save")
    #plt.savefig("heatmap_example_B_C_new_1.png",dpi=400)
    plt.savefig(filepaths.images_path+'heatmap_example_B_C'+str(kinetic_energy_cutoff)+'_.png',dpi=400)
    make_colormap_mask(stats_masked_full,kinetic_energy_cutoff)

def make_colormap_mask(stats_obj_masked,kinetic_energy_cutoff):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
    matplotlib.rc('font', **font)
    # We can visualize with a heatmap
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10,10))
    #cax=ax.matshow(sum_squares_full, cmap = 'Greens')
    cax=ax.matshow(stats_obj_masked[:,:,0], cmap='plasma', norm=LogNorm(vmin=0.1, vmax=1000))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([i for i in range(20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=14)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    fig.colorbar(cax)
    ax.set_yticks([i for i in range(20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=14)
    #ax.set_xticklabels([str(i) for i in Categories], rotation = 90,fontsize=14)
    ax.set_ylabel("Diffusion Coefficient "r'$10^{28}cm^{2}s^{-1}$', fontsize = 14)
    ax.set_xlabel("Halo Size (kpc)", fontsize = 14)
    #plt.tight_layout()
    plt.title("Chi Square Above 20 GeV/n for B/C", y=1.08)
    print("about to save")
    #plt.savefig("heatmap_example_B_C_new_1.png",dpi=400)
    plt.savefig(filepaths.images_path+'heatmap_example_B_C'+str(kinetic_energy_cutoff)+'_masked_.png',dpi=400)
   

## NEW CODE FOR ADJUSTING THE VALUES OF CHISQUARE
