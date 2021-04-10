# This code is a combination of a bunch of other tests with jupyter notebooks. 
# Author: Keith McBride, Jan 2021
# updated April 2021: include confined simulations in z-dir
# use classes for holding all the arrays 
# applies solar modulation in the force-field approximation
# fits spectra in the class, using scipy optimize curvefit, above some input cutoff
# 
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
from Isotope_CR import Nuclei
from Isotope_CR import *
from get_ams_data_functions import *
import cosmic_ray_nuclei_index
from cosmic_ray_nuclei_index import rigidity_calc, undo_log_energy, log_energy
from get_splines import *
from get_residuals_chi_square import *
from open_tarfile import get_fluxes_from_files
# Example function for testing this procedure before expanding to many ratios
def run_analysis_test(seq,num_spline_steps,cutoff,solar_phi):
    ########## FIRST THE DATA ##########
    which_error=1
    MAX_DIFFUSION=5
    numerator='B'  # numerator of the nuclei ratio used in the chi_square calculation (needs to also be AMS_DATA file first character)
    denominator='C'  # same as numerator but denominator for the nuclei ratio
    #num=BORON
    B_C_df=load_ams_ratios('B','C',which_error) # one for each ratio that we want to compare to the galprop models, last arg is for error loaded (0=just stat, 1=sys+stat)
    B_C_data_array_R=np.array(B_C_df.rigidity.values)
    B_C_data_array_ratio=np.array(B_C_df.ratio.values)
    B_C_data_array_ratio_errors=np.array(B_C_df.ratio_errors.values)
    B_O_df=load_ams_ratios('B','O',which_error) # example of which error being 1 to give stats and sys added together
    #num=Be
    #Be_B_df=load_ams_ratios('Be','B')
    #Be_C_df=load_ams_ratios('Be','C')
    #Be_O_df=load_ams_ratios('Be','O')
    #num=Li
    #Li_B_df=load_ams_ratios('Li','B')
    #Li_C_df=load_ams_ratios('Li','C')
    #Li_O_df=load_ams_ratios('Li','O')
    ####OTHER DATA (NOT AMS)
    #pbar_p_df=load_pbarp_ams_ratio()
    #Be10_Be9_df=load_Be10_Be9_ratio()
    #H2_H1_df=load_H2_H1_ratio()
    #He3_He4_ratio=load_He3_He4_ratio()
    #B_C_voyager_df=load_B_C_ratio_voyager()
    #### DATA LOADED
    ### two x axis values occur:
    ### FOR NUCLEI RATIOS (EXCEPT VOYAGER) : Rigidity [GV]
    ### FOR ISOTOPE RATIOS (AND B/C-VOYAGER) : Kinetic Energy [GeV/n]
    ### ALL y axis values are ratios (no intensities or fluxes)
    #print(f'Num data points in AMS data: {len(ratio)}')
    #     IF YOU WANT TO PLOT JUST THE DATA run the code below  #
    #do you want y-axis log-scaled? (1=yes)
    #log_y=1
    #make_plot_of_data(B_C_df,'B','C',log_y,0)
    #make_plot_of_data(B_O_df,'B','C',log_y,1)
    #make_plot_of_B_C_voyager_data(B_C_voyager_df,'B','C',log_show)
    # look at get_ams_data_functions for more plotting functions
    print("success, data loaded")
    ########## END PLOTTING AND LOADING THE DATA ##########
    ########## SECOND THE SIMULATION SETS ##########
    # make arrays of the energy axis for all isotope fluxes (GALPROP states kinetic energy [MeV/n] as FITS file energy axis)
    # get energy axis and change to GeV (undo the logarithm to put in actual energy units) # this is done step by step to make it SUPER transparent. 
    energy=np.arange(2,9,0.304347391792257) # construct the log energy array (these are just the powers of base 10) that matches galprop fits files
    energy=undo_log_energy(energy) # undo the logarithm so now these are MeV/nucleon energies 
    #energy_mev_nuc=energy.copy() # the energy array in MeV/nuc for converting the flux when being read in.
    energy=np.true_divide(energy,10**3) # change to GeV/nucleon because the rigidity conversion is in GeV (mp=0.938 GeV)
    # BIG LOOP TO GET ALL 400 models in sets of 20 (constant diffusion coefficient but varying halo size)
    #chi_square_array=[] # in the end this is a list of the models that is arbitrary dimensional
    chi_square_nparray=np.empty([20,20]) # this is what we expect the chi_square_array to be if all goes well, where each value is the chi_square on a grid of halo size vs diffusion coeff
    pvalue_nparray=np.empty([20,20]) # this is what we expect the chi_square_array to be if all goes well, where each value is the chi_square on a grid of halo size vs diffusion coeff
    model_name_full=np.empty([20,20],dtype="S10")
    stats_full=np.empty([20,20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
    stats_full_reduced=np.empty([20,20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
    #stats_masked_full=np.empty([20,20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
    spectral_index_nparray=np.empty([20,20]) # fitting to B/C ratio
    spectral_amplitude_nparray=np.empty([20,20]) # fitting to B/C ratio
    covariance_nparray=np.empty([20,20,2,2])
    diffusion_number=5 #(set to [1,20])
    while diffusion_number<MAX_DIFFUSION+1:
        chi_square_temp=[]  # reset this one at each iteration
        #first arg is not used currently for the following function (so it can be anything really)
        #fluxes_per_element_per_diffusion=get_fluxes_from_files(1, diffusion_number)  # to store the loaded fluxes from GALPROP sims, this uses tarzipped files
        fluxes_per_element_per_diffusion=get_fluxes_from_files(1, diffusion_number)  # to store the loaded fluxes from GALPROP sims
        # ALL DATA AND FLUXES HAVE BEEN LOADED in the fluxes_per_element_per_diffusion array
        # ADJUST UNITS OF THE SIMULATION ARRAYS
        # the energy array is the same for all isotopes
        #need to pass the log fluxes found from the models above like so:
        halo_model=0 # halo size is weird, since it reads them in alphebatical order. model=0 is L=10, model=1 L=11, .. model=10 L=1, model=11 L=20, model=12 L=2 ... model=19 L=9 
        #ratios_splined_per_diffusion=np.empty([20,])
        model_name=np.empty([20,],dtype="S10")
        chi_square_array_per_diffusion=np.empty([20,])
        pvalue_array_per_diffusion=np.empty([20,])
        stats_per_diffusion=np.empty([20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
        stats_per_diffusion_reduced=np.empty([20,2]) # the stats object is a scipy chiqsuare object which has two components a chisquare and a pvalue
        spectral_index_per_diffusion=np.empty([20,])
        spectral_amplitude_per_diffusion=np.empty([20,])
        covariance_per_diffusion=np.empty([20,2,2])
        while halo_model<20:
            # to get a nuclei ratio, need nuclei fluxes. these are classes too
            b_obj=make_boron_nuclei("boron",5,energy,halo_model,solar_phi,num_spline_steps,fluxes_per_element_per_diffusion)
            c_obj=make_carbon_nuclei("carbon",6,energy,halo_model,solar_phi,num_spline_steps,fluxes_per_element_per_diffusion)
            o_obj=make_oxygen_nuclei("oxygen",8,energy,halo_model,solar_phi,num_spline_steps,fluxes_per_element_per_diffusion)
            be_obj=make_beryllium_nuclei("beryllium",4,energy,halo_model,solar_phi,num_spline_steps,fluxes_per_element_per_diffusion)
            calc_iso_ratio(be_obj,'Be-10','Be-9')
            B_C_ratio_obj=Ratio("Boron-Carbon Ratio")
            B_C_ratio_obj.add_nuclei(b_obj,c_obj,num_spline_steps)
            B_C_ratio_obj.fit_ratio(cutoff)
            B_C_ratio_obj.analyze_ratio(B_C_data_array_R[B_C_data_array_R>cutoff],B_C_data_array_ratio[B_C_data_array_R>cutoff],B_C_data_array_ratio_errors[B_C_data_array_R>cutoff])
            chi_square=B_C_ratio_obj.chi_square_val
            spec_i=B_C_ratio_obj.spectral_index
            spec_A=B_C_ratio_obj.spectral_amplitude
            spec_cov=B_C_ratio_obj.covariance
            # if there is a cutoff in rigidity (use data at this value and above), then use the following
            #residuals,chi_square,stats_obj,stats_obj_masked,rigidity_masked,ratio_masked,model_x,model_y,model_x_masked,model_y_masked=calculate_chi_square(rigidity,ratio,rigC13_spline, B_C_ratio_spline,cutoff)
            # these quantities need to be saved into huge arrays, but also apply the formatting inline here
            # formatting is that the loaded files are done in alphabetical order, so the numbers are not in the correct order. 
            # the print statement below shows that the first model read in is L=10, L=11, ... L=19 then L=1, L=20, then L=2, L=3....L=9
            #print(f'chi-square {chi_square}, from model {fluxes_per_element_per_diffusion[halo_model][-1]}')
            if halo_model<=9: #this is L=10 so set it 9th position in array, ==1 L=11 set it to 10 pos in array... j==9 L=19 set it to 18 pos in array
                chi_square_array_per_diffusion[halo_model+9]=chi_square
                #chi_square_array_per_diffusion[halo_model+9]=chi_reduced
                #pvalue_array_per_diffusion[halo_model+9]=p_value
                model_name[halo_model+9]=fluxes_per_element_per_diffusion[halo_model][-1]
                spectral_index_per_diffusion[halo_model+9]=spec_i
                spectral_amplitude_per_diffusion[halo_model+9]=spec_A
                covariance_per_diffusion[halo_model+9]=spec_cov                
                #ratios_splined_per_diffusion[halo_model+9]=B_C_ratio_spline
                #stats_per_diffusion[halo_model+9]=stats_obj
                #stats_masked_per_diffusion[halo_model+9]=stats_obj_masked
            elif halo_model==10: # this is L=1 so set it to zeroth pos
                #make_plot_ratio_modulated(B_C_ratio_obj.energy_per_nucleon,B_C_ratio_obj.energy_per_nucleon_modulated,
                          #B_C_ratio_obj.ratio_energy_per_nucleon,B_C_ratio_obj.ratio_energy_per_nucleon_modulated)
                chi_square_array_per_diffusion[0]=chi_square
                #chi_square_array_per_diffusion[0]=chi_reduced
                #pvalue_array_per_diffusion[0]=p_value
                model_name[0]=fluxes_per_element_per_diffusion[halo_model][-1]
                spectral_index_per_diffusion[0]=spec_i
                spectral_amplitude_per_diffusion[0]=spec_A
                covariance_per_diffusion[0]=spec_cov                
                #ratios_splined_per_diffusion[0]=B_C_ratio_spline
                #stats_per_diffusion[0]=stats_obj
                #stats_masked_per_diffusion[0]=stats_obj_masked
            elif halo_model==11: #this is L=20 so set it to last pos
                chi_square_array_per_diffusion[19]=chi_square
                #chi_square_array_per_diffusion[19]=chi_reduced
                #pvalue_array_per_diffusion[19]=p_value
                model_name[19]=fluxes_per_element_per_diffusion[halo_model][-1]
                spectral_index_per_diffusion[19]=spec_i
                spectral_amplitude_per_diffusion[19]=spec_A
                covariance_per_diffusion[19]=spec_cov                
                #ratios_splined_per_diffusion[19]=B_C_ratio_spline
                #stats_per_diffusion[19]=stats_obj
                #stats_masked_per_diffusion[19]=stats_obj_masked
            elif halo_model>=12:  # these are j>=12 where j==12 is L=2 so set to 1 pos in array, j==13 set it 2 in array
                chi_square_array_per_diffusion[halo_model-11]=chi_square
                #chi_square_array_per_diffusion[halo_model-11]=chi_reduced
                #pvalue_array_per_diffusion[halo_model-11]=p_value
                model_name[halo_model-11]=fluxes_per_element_per_diffusion[halo_model][-1]
                spectral_index_per_diffusion[halo_model-11]=spec_i
                spectral_amplitude_per_diffusion[halo_model-11]=spec_A
                covariance_per_diffusion[halo_model-11]=spec_cov                
                #ratios_splined_per_diffusion[halo_model-11]=B_C_ratio_spline
                #stats_per_diffusion[halo_model-11]=stats_obj
                #stats_masked_per_diffusion[halo_model-11]=stats_obj_masked
            # other variables can be added as needed (for plotting or further investigation)
            halo_model+=1
        chi_square_nparray[diffusion_number-1]=chi_square_array_per_diffusion
        #pvalue_nparray[diffusion_number-1]=pvalue_array_per_diffusion
        #chi_square_array.append(chi_square_array_per_diffusion)
        model_name_full[diffusion_number-1]=model_name
        spectral_index_nparray[diffusion_number-1]=spectral_index_per_diffusion
        spectral_amplitude_nparray[diffusion_number-1]=spectral_amplitude_per_diffusion
        covariance_nparray[diffusion_number-1]=covariance_per_diffusion
        #stats_full[diffusion_number-1]=stats_per_diffusion
        #stats_masked_full[diffusion_number-1]=stats_masked_per_diffusion
        diffusion_number+=1
    #something else

    #print('{chi_square_array[0][0]})
    print("Saved file")
    #return chi_square_array,chi_square_nparray,model_name_full,stats_full,stats_masked_full
    print(chi_square_nparray)
    #print(chi_square_nparray)
    print(spectral_index_nparray)
    print(spectral_amplitude_nparray)
    print(covariance_nparray)
    np.savetxt('spectralindexfits_-0.33nominal.txt', spectral_index_nparray, fmt='%f')
    np.savetxt('spectralamplitudesfits_-0.33nominal.txt', spectral_amplitude_nparray, fmt='%f')
    np.savetxt('chisquare_cutoff_-0.33nominal_error_'+str(which_error)+'.txt', chi_square_nparray, fmt='%f')
    #print(pvalue_nparray)
    #return chi_square_nparray, pvalue_nparray
    #return spectral_index_nparray,spectral_amplitude_nparray,covariance_nparray
    #with open("spectralindexfile.txt", 'w') as file:
        #for row in spectral_index_nparray:
            #s = " ".join(map(str, row))
            #file.write(s+'\n')



def make_plot_ratio_modulated(energy1,energy2,flux1,flux2):
    fnt=22
    x1=4*10**-3
    x2=2.5*10**5
    y1=0.02
    y2=0.4
    plt.figure(figsize=(14,10))
    plt.plot(energy1,flux1,'r--',label="not modulated")
    plt.plot(energy2,flux2,'b-.',label="modulated 600MV")
    #plt.plot(energy3,10**(-2)*flux3,'k',marker="X",ms=2,label="O-not modulated")
    #plt.plot(energy4,10**(-2)*flux4,'c',marker="o",ms=2,label="O-modulated 600MV")
    #plt.plot(energy3,flux3,'g',marker="*",ms=10,label="modulated 1200MV")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Kinetic Energy [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt)
    plt.ylabel("B/C flux ratio",fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    plt.legend(loc='lower right', fontsize=fnt-4)
    ax = plt.gca()
    ax.tick_params(width=2,length=5)
    #plt.setp(plt.axis.tick_params(axis=both, width=3,length=3))
    plt.title("B/C ratio test", fontsize=fnt)
    #plt.savefig("B_C_ratio_test_modulated_classes.png")
    plt.savefig(filepaths.images_path+'B_C_ratio_test_modulated_classes.png')

def find_chi_minimum(chi_squares_full):
    # quantities for the analysis
    num_spline_steps=2000
    kinetic_energy_cutoff=20 # GeV/nucleon
    cutoff=rigidity_calc(kinetic_energy_cutoff,13,6) # the other two args are, in order, mass (num nucleons) and charge (Z).
    #chi_squares_full, chi_squares_full_np,models,stats_full, stats_masked_full=run_chi_square_test(1,num_spline_steps,cutoff)
    # for the chi_square numpy array with shape [20,20], the first index would be constant diffusion and the second index constant halo size.
    # so varying the second index is sampling different halo sizes
    # when using np.where(array) the return is a tuple.
    # tuple is 2 dimensionful, where tuple[0] is the diffusion index and tuple[1] is the halo index
    # but if many values are found in where, then space is expanded:
    # tuple[0][0] first diffusion index found that logically works
    # tuple[1][0] first halo index found that logically works
    # tuple[0][-1] last diffusion index found that logically works
    #chi_squares_full, pvalues_full=run_chi_square_test(1,num_spline_steps,cutoff)
    #minimum_chi=np.amin(chi_squares_full)
    minimum_tuple=np.where(chi_squares_full==np.amin(chi_squares_full))
    min_chi_square=chi_squares_full[minimum_tuple]
    min_halo_model=minimum_tuple[1]+1
    min_diffusion_model=minimum_tuple[0]+1
    # 95% confidence level will be at a delta chi square of 6.14 above the minimum
    print(f'minimum chi square is {min_chi_square} where L={min_halo_model} and D={min_diffusion_model}')
    delta_chi_square=6.14
    allowed_models=np.where(chi_squares_full<=(min_chi_square+delta_chi_square))
    diffusion_models_allowed=allowed_models[0]+1 # add one since these are indices but correspond to the diffusion coefficient
    halo_models_allowed=allowed_models[1]+1 # add one since these are indices but correspond to the halo size
    #something else
    with open("chisquarefile_cutoff.txt", 'w') as file:
        for row in chi_squares_full:
            s = " ".join(map(str, row))
            file.write(s+'\n')
    print("File saved")
    #file.write(f'{minimum_tuple}')
    #plt.figure(figsize=(12,12))
    #plt.plot(diffusion_models_allowed,halo_models_allowed,'b--',linewidth=3,label='95% interval')
    #plt.plot(min_diffusion_model,min_halo_model,marker='X',ms=8,label='Minimum')
    #plt.xlim([1,20])
    #plt.ylim([1,20])
    #plt.xlabel("Diffusion Coefficient "r'$10^{28}cm^{2}s^{-1}$', fontsize = 14)
    #plt.ylabel("Halo Size (kpc)", fontsize = 14)
    #plt.title("Chi Square 95% Confidence Interval for B/C")
    #plt.savefig(filepaths.images_path+'Confidence_Interval_B_C_'+str(kinetic_energy_cutoff)+'_.png',dpi=400)


def make_colormap(seq):
    # quantities for the analysis
    num_spline_steps=2000
    kinetic_energy_cutoff=20 # GeV/nucleon
    cutoff=rigidity_calc(kinetic_energy_cutoff,13,6) # the other two args are, in order, mass (num nucleons) and charge (Z).
    #chi_squares_full, chi_squares_full_np,models,stats_full, stats_masked_full=run_chi_square_test(1,num_spline_steps,cutoff)
    chi_squares_full, pvalues_full=run_chi_square_test(1,num_spline_steps,cutoff)
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
    # Try tranposing to make halo size on y axis
    chi_squares_transposed=np.transpose(chi_squares_full)
    cax=ax.matshow(chi_squares_transposed, cmap='plasma', norm=LogNorm(vmin=1, vmax=1000),origin='lower')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([i for i in range(20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=14)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    fig.colorbar(cax)
    ax.set_yticks([i for i in range(20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=14)
    #ax.set_xticklabels([str(i) for i in Categories], rotation = 90,fontsize=14)
    ax.set_xlabel("Diffusion Coefficient "r'$10^{28}cm^{2}s^{-1}$', fontsize = 14)
    ax.set_ylabel("Halo Size (kpc)", fontsize = 14)
    #plt.tight_layout()
    plt.title("Chi Square for B/C", y=1.08)
    print("about to save")
    #plt.savefig("heatmap_example_B_C_new_1.png",dpi=400)
    plt.savefig(filepaths.images_path+'heatmap_example_B_C_'+str(kinetic_energy_cutoff)+'_.png',dpi=400)
    #make_pvalue_colormap(pvalues_full,kinetic_energy_cutoff)
    find_chi_minimum(chi_squares_full)
    return True

def make_pvalue_colormap(pvalues_full,kinetic_energy_cutoff):
    # now pvalues
    fig, ax = plt.subplots(figsize=(10,10))
    cax=ax.matshow(pvalues_full, cmap='plasma')
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
    plt.title("P-Value for B/C", y=1.08)
    print("about to save")
    print(pvalues_full)
    plt.savefig(filepaths.images_path+'heatmap_example_B_C_pvalues'+str(kinetic_energy_cutoff)+'_.png')
    #make_colormap_mask(stats_masked_full,kinetic_energy_cutoff)



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
