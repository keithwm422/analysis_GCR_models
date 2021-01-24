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
def run_chi_square_test(seq,num_spline_steps):
    # FIRST THE DATA
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
    diffusion_number=1 #(set to [1,20])
    while diffusion_number<21:
        chi_square_temp=[]  # reset this one at each iteration
        #first arg is not used currently for the following function (so it can be anything really)
        fluxes_per_element_per_diffusion=get_fluxes_from_files(1, diffusion_number)  # to store the loaded fluxes from GALPROP sims
        # ALL DATA AND FLUXES HAVE BEEN LOADED
        # ADJUST UNITS OF THE SIMULATION ARRAYS
        # the energy array is the same for all isotopes
        #need to pass the log fluxes found from the models above like so:
        halo_model=0 # halo size is weird, since it reads them in alphebatical order. model=0 is L=10, model=1 L=11, .. model=10 L=1, model=11 L=20, model=12 L=2 ... model=19 L=9 
        ratios_splined_per_diffusion=[]
        chi_square_array_per_diffusion=[]
        while halo_model<20:
            # access the actual flux from the loaded simulation sets and then log them
            logB10_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron10_loc)])
            logB11_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron11_loc)])
            logC12_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon12_loc)])
            logC13_flux=log_energy(fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon13_loc)])
            # now call function that will spline all those fluxes and the energy axis to a common rigidity range among all the isotopes. 
            rigC13_spline,B_C_ratio_spline=B_C_ratio(energy,logB10_flux,logB11_flux,logC12_flux,logC13_flux,num_spline_steps)
            ratios_splined_per_diffusion.append(B_C_ratio_spline)
            # NOW CAN CALCULATE THE CHI-SQUARE and get residuals
            residuals,chi_square=calculate_chi_square(rigidity,ratio,rigC13_spline, B_C_ratio_spline) 
            #print(chi_square)
            print(f'chi-square {chi_square}, from model {fluxes_per_element_per_diffusion[halo_model][-1]}')
            chi_square_array_per_diffusion.append(chi_square)
            chi_square_array_per_diffusion.append(fluxes_per_element_per_diffusion[halo_model][-1])
            halo_model+=1
        chi_square_array.append(chi_square_array_per_diffusion)
        diffusion_number+=1
    # plot the model chosen and the AMS data
    #plt.figure(figsize=(12,12))
    #fnt=14
    #x1=0.9*rigC13_spline[0]
    #x2=2*rigC13_spline[-1]
 
    #plt.plot(rigC13_spline, B_C_ratio_spline,'--',label="Ratio spline")
    #plt.errorbar(rigidity,ratio,xerr=rigidity_binsize,yerr=ratio_errors,fmt='o',label="AMS")

    #plt.plot(energy,be_10_be_9_5,'-o',label="L=5")
    #plt.xscale("log")
    #plt.xlabel("Rigidity (GV)",fontsize=fnt)
    #plt.xticks(fontsize=fnt-4)
    #plt.ylabel("Flux ratio",fontsize=fnt)
    #plt.yticks(fontsize=fnt-4)
    #plt.xlim([x1,x2])
    #plt.legend(loc='upper right')
    #plt.title("Example, Boron and Carbon Ratio", fontsize=fnt)
    #plt.savefig("Boron_carbon_fluxratio_rigidity_splines" + str(halo_model) + ".png")
    
    #this code didnt seem to work for formatting weirdness so try something else
    #np.savetxt("chi_square_values.csv", chi_square_array, delimiter=",")
    
    #something else
    #with open("chisquarefile.txt", 'w') as file:
        #for row in chi_square_array:
            #s = " ".join(map(str, row))
            #file.write(s+'\n')

    #print('{chi_square_array[0][0]})
    print("Saved file")
    return chi_square_array

def format_chi_square(seq, num_spline_steps):
    chi_square_array=run_chi_square_test(1,num_spline_steps)
    print(type(chi_square_array))
    print(f'chi square array num rows {len(chi_square_array)}')
    print(f'chi square array num columns {len(chi_square_array[0])}')
    #print(chi_square_array[0]) # this is still a list since its a list of lists
    diffusion_iter=0
    L=[]
    D=[]
    chi_squares_full=[]
    while diffusion_iter<len(chi_square_array):
        iter=0
        chi_squares=[]
        while iter<len(chi_square_array[diffusion_iter]): #axis for constant diffusion coeff, so halo size axis
            # the first value is L=10, D=1, the second value is L=11, D=1. So the first 20*2 will be D=1 and L in a strange order.
            if (iter % 2) == 0:
                chi_squares.append(float(chi_square_array[diffusion_iter][iter]))
            else:
                L_D_values=chi_square_array[diffusion_iter][iter].split("_")
                #print(L_D_values)
                L.append(int(L_D_values[-3]))
                D.append(int(L_D_values[-1]))
            iter+=1
        # don't append the values since they are out of order. Instead, readjust them into the right order
        j=0
        temp=chi_squares.copy()
        #print(f'j={j} and temp before adjustments {temp}')
        while j<len(chi_squares):
            if j<=9: #this is L=10 so set it 9th position in array, j==1 L=11 set it to 10 pos in array... j==9 L=19 set it to 18 pos in array
                temp[j+9]=chi_squares[j]
                #print(f'j={j} and j should be leq 9')
            elif j==10: # this is L=1 so set it to zeroth pos
                temp[0]=chi_squares[j]
                #print(f'j={j} and j should be 10')
            elif j==11: #this is L=20 so set it to last pos
                temp[19]=chi_squares[j]
                #print(f'j={j} and j should be 11')
            elif j>=12:  # these are j>=12 where j==12 is L=2 so set to 1 pos in array, j==13 set it 2 in array
                temp[j-11]=chi_squares[j]
                #print(f'j={j} and j should be geq 12')
                #print(f'j-11={j-11} and temp should be {chi_squares[j]} but temp is {temp[j-11]}')
            j+=1
        #print(f'j={j} and temp before adjustments {temp}')
        chi_squares_full.append(temp)
        diffusion_iter+=1
    # now for each column in chi_square_full, need to readjust the halo models so they match up accurately
    #print(chi_squares_full[0][5]) # this is still a list since its a list of lists
    #print(L[5])
    #print(D[5])
    #print(chi_square_array[0][10])
    #print(chi_square_array[0][11])
    return chi_squares_full

def make_colormap(seq):
    num_spline_steps=2000
    chi_squares_full=format_chi_square(1,num_spline_steps)
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
    plt.savefig(filepaths.images_path+'heatmap_example_B_C.png',dpi=400)


## NEW CODE FOR ADJUSTING THE VALUES OF CHISQUARE
