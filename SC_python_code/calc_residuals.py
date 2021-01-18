#!/usr/bin/env python3
# this code opens a fits file and plots it
#author Keith McBride, 2020-11

import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import splev, splrep

###helper functions###

#log10(E/MeV) = CRVAL3 + k * CDELT3 need to add in 100 GeV to this energy passed here since I am dumb
#flux (MeV/nucleon)2 cm−2sr−1s−1(MeV/nucleon)−1
# flux becomes MeV/nucleon /cm**2 /sr /s 
def undo_log_energy(energy):
    i=0
    while i<len(energy):
        energy[i]=10**(energy[i]+2)
        i+=1

#read in the ams data on this flux ratio
def read_ams_data():
    #read in the ams data on helium
    ams=pd.read_csv('he_3_4_ams_data.csv')
    ams.head()
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
    print(ams_ratio_errors)
    return ams_energy_mp, ams_ratio, ams_energy_binsize, ams_ratio_errors

#spline the galprop ratio so we can easily find the residuals
def spline_the_ratio(energy,ratio):
    spl = splrep(energy,ratio)
    energy_cont = np.linspace(2, 7, 2000)
    He_3_He_4_spline = splev(energy_cont, spl)
    return energy_cont,He_3_He_4_spline

def gimme_residuals(ams_energy_mp_1, ams_ratio, energy_cont, He_3_He_4_spline):
# to calculate residuals 
#find the elements of the spline energy array closest to the ams energy value:
    i=0
    closest_element=[]
    closest_index=[]
    residual=[]
    while i<len(ams_energy_mp_1):
        array=energy_cont
        value=ams_energy_mp_1[i]
        absolute_val_array = np.abs(array - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element.append(array[smallest_difference_index])
        closest_index.append(smallest_difference_index)
        residual.append(ams_ratio[i]-He_3_He_4_spline[smallest_difference_index])
        i+=1
    print(ams_energy_mp_1)
    print(closest_element)
    print(closest_index)
    print(residual)
    #L_val=10
    #D_val=12
    return closest_element,residual

#plot the data and simulation with spline
def plot_data_and_sim(seq):
    ### Declare some variables###
    master_path='/home/mcbride.342/galprop_sims/keith_runs_example/out_FITS/'
    image_file=master_path+'nuclei_full_56_example'
    hdu_list = fits.open(image_file)
    hdr=hdu_list[0].header
    print(len(list(hdr.keys())))
    hdu_list.close()
    image_data = fits.getdata(image_file)
    energy=np.arange(0,7,0.304347391792257)
    undo_log_energy(energy)
    energy_1=np.arange(0,7,0.304347391792257)+2
    #these are at the position of earth we expect.
    x_loc=24
    y_loc=22
    z_loc=20 #40th for the highest halo size of 4, 30th for halo size of 3
    He_3_He_4=np.divide(image_data[10,:,z_loc,y_loc,x_loc],image_data[11,:,z_loc,y_loc,x_loc])
    ams_energy_mp, ams_ratio, ams_energy_binsize, ams_ratio_errors = read_ams_data()
    ams_energy_mp_1=np.log10(ams_energy_mp)
    energy_cont, He_3_He_4_spline = spline_the_ratio(energy_1,He_3_He_4)
    fnt=20
    x1=2
    x2=7
    y1=0
    y2=0.25
    plt.figure(figsize=(12,12))
    plt.plot(energy_1,He_3_He_4,'--o',color='black',label="GALPROP")
    plt.plot(energy_cont,He_3_He_4_spline,'--',color='red',label="spline")
    plt.errorbar(ams_energy_mp_1,ams_ratio,yerr=ams_ratio_errors,fmt='o',label="AMS")
    #plt.plot(energy,he_3_4_3,'-o',label="L=3")
    #plt.plot(energy,he_3_4_4,'-o',label="L=4")
    #plt.plot(energy,he_3_4_5,'-o',label="L=5")
    #plt.xscale("log")
    plt.xlabel("Log10 Kinetic Energy per nucleon "r'$\frac{MeV}{nucleon}$',fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    #plt.yscale("log")
    plt.ylabel("Flux division He-3/He-4",fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    plt.legend(loc='lower right', fontsize=fnt-4)
    plt.title("Example, Helium", fontsize=fnt)
    plt.savefig("He_ratio_ex_ams_spline.png")
    #plt.show()    

#plot the data and simulation and residual with spline
def plot_residuals(seq):
    ### Declare some variables###
    master_path='/home/mcbride.342/galprop_sims/keith_runs_example/out_FITS/'
    image_file=master_path+'nuclei_full_56_example'
    hdu_list = fits.open(image_file)
    hdr=hdu_list[0].header
    print(len(list(hdr.keys())))
    hdu_list.close()
    image_data = fits.getdata(image_file)
    energy=np.arange(0,7,0.304347391792257)
    undo_log_energy(energy)
    energy_1=np.arange(0,7,0.304347391792257)+2
    #these are at the position of earth we expect.
    x_loc=24
    y_loc=22
    z_loc=20 #40th for the highest halo size of 4, 30th for halo size of 3
    He_3_He_4=np.divide(image_data[10,:,z_loc,y_loc,x_loc],image_data[11,:,z_loc,y_loc,x_loc])
    ams_energy_mp, ams_ratio, ams_energy_binsize, ams_ratio_errors = read_ams_data()
    ams_energy_mp_1=np.log10(ams_energy_mp)
    energy_cont, He_3_He_4_spline = spline_the_ratio(energy_1,He_3_He_4)
    closest_energy, residual = gimme_residuals(ams_energy_mp_1, ams_ratio, energy_cont, He_3_He_4_spline)
    fnt=20
    x1=2
    x2=5
    fig,ax=plt.subplots(figsize=(18, 18), dpi=400, nrows=2,sharex=True)
    fig.subplots_adjust(hspace=0)
    #y1=0
    #y2=0.25
    ax[0].plot(energy_1,He_3_He_4,'--o',color='black',label="GALPROP")
    ax[0].plot(energy_cont,He_3_He_4_spline,'--',color='red',label="spline")
    ax[0].errorbar(ams_energy_mp_1,ams_ratio,yerr=ams_ratio_errors,fmt='o',label="AMS")
    #plt.plot(energy,he_3_4_3,'-o',label="L=3")
    #plt.plot(energy,he_3_4_4,'-o',label="L=4")
    #plt.plot(energy,he_3_4_5,'-o',label="L=5")
    #plt.xscale("log")
    ax[0].tick_params(labelsize=fnt-4)
    ax[0].set_ylabel("Flux division He-3/He-4",fontsize=fnt)
    ax[0].set_xlim([x1,x2])
    ax[0].legend(loc='upper right', fontsize=fnt-4)
    ax[1].set_xlabel("Log10 Kinetic Energy per nucleon "r'$\frac{MeV}{nucleon}$',fontsize=fnt)
    ax[1].errorbar(closest_energy,residual,yerr=ams_ratio_errors,fmt='o',color='black')
    ax[1].tick_params(labelsize=fnt-4)
    plt.suptitle("Example, Helium with residuals", fontsize=fnt)
    plt.savefig("He_ratio_ex_ams_spline_residuals.png")
    #plt.show()    
