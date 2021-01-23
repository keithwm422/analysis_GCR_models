# This code is a combination of a bunch of other tests with jupyter notebooks. 
# Author: Keith McBride, Jan 2021
# Run this to compare the AMS data and ALOT OF GALPROP models 
# using the chi-square test statistic. 
# The resulting output will be the He-3/He-4 AMS data and the best fit model, as a function of rigidity. 
# this file builds on the content of B_C_chi_square_test.py

# Set up matplotlib and use a nicer set of plot parameters
# IMPORT REGULAR PACKAGES
import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')  # declare backend for plotting this OS
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

#spline the galprop ratio so we can easily find the residuals
def spline_the_ratio(energy,ratio,num_steps):
    spl = splrep(energy,ratio)
    energy_cont = np.linspace(2, 7, num_steps)
    He_3_He_4_spline = splev(energy_cont, spl)
    return energy_cont,He_3_He_4_spline 


# Example function for testing this procedure before expanding to many ratios
def run_chi_square_test(seq):
    # FIRST THE DATA
    #read in the ams data on helium
 
    path='/home/mcbride.342/galprop_sims/AMS_Data/Ratios/' # path for the AMS DATA files
    file_name='he_3_4_ams_data.csv'  
    ams=pd.read_csv(path+file_name)
    #print(ams.head())
    #join low and high together as one array to be used as x error bars
    ams_energy=np.array((ams.EK_low.values,ams.Ek_high.values.T))
    ams_energy=ams_energy*1000
    ams_energy_mp=(ams_energy[0,:]+ams_energy[1,:])/2.0
    # now make the error bar sizes (symmetric about these midpoints)
    ams_energy_binsize=(ams_energy[1,:]-ams_energy[0,:])/2.0
    #make the ratio an array
    ams_ratio=np.array(ams._3He_over_4He.values * ams._factor_ratio.values)
    ams_ratio_sys_erros=np.array(ams._sys_ratio.values * ams._factor_ratio)
    ams_ratio_stat_erros=np.array(ams._stat_ratio.values * ams._factor_ratio)
    ams_ratio_errors=np.sqrt(np.square(ams_ratio_stat_erros)+np.square(ams_ratio_sys_erros))
    #ams_ratio_errors
    print(f'ratio {ams_ratio}')
    print(f'energy{ams_energy_mp}')
    ams_energy_mp_1=np.log10(ams_energy_mp) # since the spline will be in log10 energy anyways
    print(f'logged energy {ams_energy_mp_1}')
    #return ams_energy_mp, ams_ratio, ams_energy_binsize, ams_ratio_errors

    #     IF YOU WANT TO PLOT JUST THE DATA (there is a different file for that though)
    #do you want y-axis log-scaled? (1=yes)
    #log_y=1
    #make_plot_of_data(numerator,denominator,rigidity,ratio,rigidity_binsize,ratio_errors,log_y)
    #print("success")
    #     END PLOTTING THE DATA

    # SECOND THE SIMULATION SETS
    # make arrays of the energy axis for all isotope fluxes (really kinetic energy)
    # no need to change to regular energy (log)energy is fine. 
    energy=np.arange(2,9,0.304347391792257)
    #energy=undo_log_energy(energy)
    #energy=np.true_divide(energy,10**3) # do not need GeV/n for He-3 and He-4
    print("ENERGY ARRAY: ")
    print(energy)
    # BIG LOOP TO GET ALL 400 models in sets of 20.
    chi_square_array=[]
    diffusion_number=1 #(set to 1 to 20)
    while diffusion_number<21:
        chi_square_temp=[]  # reset this one at each iteration
        #first arg is not used currently for the following function (so it can be anything really)
        fluxes_per_element_per_diffusion=get_fluxes_from_files(1, diffusion_number)  # to store the loaded fluxes from GALPROP sims
        # ALL DATA AND FLUXES HAVE BEEN LOADED
        # ADJUST UNITS OF THE SIMULATION ARRAYS
        # the energy array is the same for all isotopes
        #need to pass the log fluxes found from the models above like so:
        halo_model=0 # halo size is weird, since it reads them in alphebatical order. model=0 is L=10, model=1 L=11, .. model=10 L=1, model=11 L=20, model=12 L=2 ... model=19 L=9 
        num_steps=200 # number of points in the spline array when interpolating
        ratios_splined_per_diffusion=[]
        chi_square_array_per_diffusion=[]
        while halo_model<20:
            # access the actual flux from the loaded simulation sets and then log them
            # for He-3 and 4 this is should be to calc the ratio and then call splining functions

            He_3_flux=fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.he3_loc)]
            He_4_flux=fluxes_per_element_per_diffusion[halo_model][cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.he4_loc)]
            He_3_4_ratio=np.divide(He_3_flux,He_4_flux)
            #print(f'ratio from sims {He_3_4_ratio}')
            #spline the ratio with the energy
            energy_spline,He_3_4_ratio_spline=spline_the_ratio(energy, He_3_4_ratio,num_steps)            
            ratios_splined_per_diffusion.append(He_3_4_ratio_spline)
            # NOW CAN CALCULATE THE CHI-SQUARE and get residuals
            residuals,chi_square=calculate_chi_square(ams_energy_mp_1,ams_ratio,energy_spline, He_3_4_ratio_spline) 
            #print(chi_square)
            #print(f'chi-square {chi_square}, from model {fluxes_per_element_per_diffusion[halo_model][-1]}')
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
def format_chi_square(seq):
    chi_square_array=run_chi_square_test(1)
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
        print(f'j={j} and temp before adjustments {temp}')
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
        print(f'j={j} and temp before adjustments {temp}')
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
    chi_squares_full=format_chi_square(1)        
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
    plt.title("Chi Square for He-3/He-4", y=1.08)
    print("about to save")
    #plt.savefig("heatmap_example_B_C_new_1.png",dpi=400)
    plt.savefig("heatmap_example_He_3_4_final.png",dpi=400)



