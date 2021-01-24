#!/usr/bin/env python3
# this code opens a tar gzipped file and fro galprop and makes the relevant CR fluxes into a nice array
#author Keith McBride, 2020-11

import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg') 
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
import os
os.getcwd()
from astropy.io import fits
from scipy.interpolate import splev, splrep
import tarfile
import cosmic_ray_nuclei_index
import filepaths
###helper functions###

### IF YOU WANT TO SEE THE CONTENTS OF THE TAR FILE ###
def print_paths(seq):
    print(os.getcwd())
    #tarfile_path='/home/mcbride.342/galprop_sims/keith_runs_multiple/out_FITS/runs_L_D'
    tarfile_path=filepaths.tarfile_path
    images_path=filepaths.images_path
    diffusion_set=1
    ext1='runs_L_D'
    ext2='.tar.gz'
    tar = tarfile.open(tarfile_path+ext1+str(diffusion_set)+ext2)
    print(tar.getnames())
    tar.close()

### GOES THROUGH THE TARFILE AND FINDS THE FILENAMES OF ALL THE NUCLEI SPECTRA FITS FILES ###
### RETURNS AS A LIST OF FILENAMES
def find_fits_files(seq,diff_number):
    print(f'Inside of directory: {os.getcwd()}')
    #tarfile_path='/home/mcbride.342/galprop_sims/keith_runs_multiple/out_FITS/runs_L_D'
    tarfile_path=filepaths.tarfile_path
    images_path=filepaths.images_path
    diffusion_set=1
    if diff_number>=1 and diff_number<=20: diffusion_set=diff_number
    ext1='runs_L_D'
    ext2='.tar.gz'
    tar = tarfile.open(tarfile_path+ext1+str(diffusion_set)+ext2)
    #print(tar.getnames())
    i=0
    tots=0
    list_found=[]
    print(f'found the following nuclei_full FITS files:')
    while i<len(tar.getmembers()):
        if tar.getmembers()[i].isfile():
            if tar.getmembers()[i].name.find('nuclei_full') !=-1:
                list_found.append(i)
                print(tar.getmembers()[i].name)
                # find the string nuclei_full in the filename
                tots+=1
        i+=1
    print(f'Total files found: {tots}')
    #print(list_found)
    tar.close()
    return list_found

### USES THE find_fits_files FUNCTION ABOVE TO COMPILE AND RETURN ALL THE FLUXES FROM LOADING THE FITS FILES AS AN ARRAY ###
### THE FLUXES ARE ONLY OF THOSE ELEMENTS LOADED IN THE FILE: comsic_ray_nuclei_index.py AND THE ARRAY HAS THEM IN THAT ORDER
def get_fluxes_from_files(seq, diff_number):
    list_found=find_fits_files(seq,diff_number)
    ###FLUXES### in order as the elements we have above
    fluxes_per_element_full=[]
    fluxes_per_element_per_fits_file=[] # clear this after every load of a fits file    
    names=[]
    #tarfile_path='/home/mcbride.342/galprop_sims/keith_runs_multiple/out_FITS/runs_L_D'
    tarfile_path=filepaths.tarfile_path
    images_path=filepaths.images_path
    diffusion_set=1
    if diff_number>=1 and diff_number<=20: diffusion_set=diff_number
    ext1='runs_L_D'
    ext2='.tar.gz'
    tar = tarfile.open(tarfile_path+ext1+str(diffusion_set)+ext2)
    ###JUST EXTRACT BY FILE INTO FITS FILES###
    k=0
    while k<len(list_found):    
        fluxes_per_element_per_fits_file=[]
        j=list_found[k]
        open_file=tar.extractfile(tar.getmembers()[j].name)
        fileFITS_data = fits.getdata(open_file)
        print(type(fileFITS_data))
        print(fileFITS_data.shape)
        for i in cosmic_ray_nuclei_index.element_index: fluxes_per_element_per_fits_file.append(fileFITS_data[i,:,cosmic_ray_nuclei_index.z_loc,cosmic_ray_nuclei_index.y_loc,cosmic_ray_nuclei_index.x_loc])
        fluxes_per_element_per_fits_file.append(tar.getmembers()[j].name)
        #FITS_data_full.append(fileFITS_data)
        # append to the arrays we need instead:
        names.append(tar.getmembers()[j].name)
        fluxes_per_element_full.append(fluxes_per_element_per_fits_file)
        k+=1
        print(f'{k/len(list_found)}% loaded')
    tar.close()
    print(f'Loaded the following FITS file fluxes into an array')
    for k in range(len(list_found)) : print(fluxes_per_element_full[k][-1])
    return fluxes_per_element_full

