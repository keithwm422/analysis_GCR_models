from get_ams_data_functions import *
import pandas as pd
import numpy as np

def read_in_data(numerator,denominator,path):
    extension='ams_data.csv'
    read_file=path+numerator+'_'+denominator+'_'+extension
    ams=pd.read_csv(read_file)
    print(ams.head())
    return ams

def plot_da_flux(seq):
    numerator='B'
    denominator='C'
    path='/home/mcbride.342/galprop_sims/AMS_Data/Ratios/'
    df=read_in_data(numerator,denominator,path)
    rigidity,rigidity_binsize,ratio,ratio_errors=make_energies_and_errors(df,numerator,denominator)
    #do you want y-axis log-scaled? (1=yes)
    log_y=1
    make_plot_of_data(numerator,denominator,rigidity,ratio,rigidity_binsize,ratio_errors,log_y)
    print("success")
