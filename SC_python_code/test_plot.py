#!/usr/bin/env python3
# this code opens a fits file and plots it
#author Keith McBride, 2020-11

###include shit###
import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
from astropy.io import fits

###helper functions###

#log10(E/MeV) = CRVAL3 + k * CDELT3 need to add in 100 GeV to this energy passed here since I am dumb
#flux (MeV/nucleon)2 cm−2sr−1s−1(MeV/nucleon)−1
# flux becomes MeV/nucleon /cm**2 /sr /s 
def undo_log_energy(energy):
    i=0
    while i<len(energy):
        energy[i]=10**(energy[i]+2)
        i+=1



###MAIN FUNCTION###
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
    #these are at the position of earth we expect.
    x_loc=24
    y_loc=22
    z_loc=20 #40th for the highest halo size of 4, 30th for halo size of 3
    He_3_He_4=np.divide(image_data[10,:,z_loc,y_loc,x_loc],image_data[11,:,z_loc,y_loc,x_loc])
    fnt=20
    x1=10**2
    x2=10**7
    plt.figure(figsize=(12,12))
    plt.plot(energy,He_3_He_4,'--o',color='red')
    #plt.plot(energy,he_3_4_3,'-o',label="L=3")
    #plt.plot(energy,he_3_4_4,'-o',label="L=4")
    #plt.plot(energy,he_3_4_5,'-o',label="L=5")
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon "r'$\frac{MeV}{nucleon}$',fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    #plt.yscale("log")
    plt.ylabel("Flux division He-3/He-4",fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.title("Example, Helium", fontsize=fnt)
    plt.savefig("He_ratio_multiple_halo_sizes.png")
    #plt.show()    
