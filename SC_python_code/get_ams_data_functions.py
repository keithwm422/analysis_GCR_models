#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
import filepaths

### Takes the dataframe and returns the different columns separately including the calculated central rigidity bin value.
def make_energies_and_errors(df,num,den):
    rigidity=np.array((df.R_low.values,df.R_high.values.T))
    rigidity_mp=(rigidity[0,:]+rigidity[1,:])/2.0
    rigidity_binsize=(rigidity[1,:]-rigidity[0,:])/2.0
    ratio_name='_'+str(num)+'_'+str(den)+'_'+'ratio'
    ratio=np.array(df[ratio_name].values)
    ratio_sys_errors=np.array(df._sys.values)
    ratio_stat_errors=np.array(df._stat.values)
    ratio_errors=np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    return rigidity_mp,rigidity_binsize,ratio,ratio_errors

### Load the data from a csv file into a Pandas DF ###
def read_in_data(numerator,denominator):
    extension='ams_data.csv'
    read_file=filepaths.data_path+numerator+'_'+denominator+'_'+extension
    ams=pd.read_csv(read_file)
    #print(ams.head())
    return ams

def load_ams_ratios(numerator,denominator):
    # lets try returning the flux ratios as a dataframe
    df=read_in_data(numerator,denominator)
    column_names=['rigidity','rigidity_binsize','ratio','ratio_errors']
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack(make_energies_and_errors(df,numerator,denominator)),
                   columns=column_names)
    return ams_data_formatted_df

def load_pbarp_ams_ratio():
# lets try saving the fluxes per element as a dataframe?
    df=read_in_data('p_bar','p_ratio')
    column_names=['rigidity','rigidity_binsize','ratio','ratio_errors']
    rigidity=np.array((df.R_low.values,df.R_high.values.T))
    rigidity_mp=(rigidity[0,:]+rigidity[1,:])/2.0
    rigidity_binsize=(rigidity[1,:]-rigidity[0,:])/2.0
    ratio_name='p_bar'+'_'+'p_ratio'
    # need to multiply in the _ratio_factor to the errors and the ratio
    ratio=np.array(df[ratio_name].values)
    ratio_sys_errors=np.array(df._sys.values)
    ratio_stat_errors=np.array(df._stat.values)
    ratio=ratio*10.0**(df._ratio_factor.values)
    ratio_errors=10.0**(df._ratio_factor.values)*np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    #print(rigidity_mp.shape)
    #print(ratio.shape)
    #print(ratio_errors.shape)
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack((rigidity_mp,rigidity_binsize,ratio,ratio_errors)),
                   columns=column_names)
    return ams_data_formatted_df


def load_Be10_Be9_ratio():
# lets try saving the fluxes per element as a dataframe?
    extension='.csv'
    read_file=filepaths.data_path+'Be10_Be9_all'+extension
    multi_exp_df=pd.read_csv(read_file)
    #print(multi_exp_df.head())
    column_names=['experiment','kinetic','kinetic_binsize','ratio','ratio_errors']
    names=multi_exp_df._experiment.values
    kinetic=np.array((multi_exp_df.EK_low.values,multi_exp_df.EK_high.values.T))
    kinetic_mp=multi_exp_df.EK_mid.values
    kinetic_binsize=(kinetic[1,:]-kinetic[0,:])/2.0
    ratio_name='_be10'+'_'+'be9_ratio'
    ratio=np.array(multi_exp_df[ratio_name].values)
    ratio_sys_errors=np.array(multi_exp_df._sys_plus.values)
    ratio_stat_errors=np.array(multi_exp_df._stat_plus.values)
    ratio_errors=np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    #print(kinetic_mp.shape)
    #print(ratio.shape)
    #print(ratio_errors.shape)
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack((names,kinetic_mp,kinetic_binsize,ratio,ratio_errors)),
                   columns=column_names)
    return ams_data_formatted_df

def make_plot_of_Beisotope_data(df,numerator,denominator,log_show):
    fnt=20
    x1=df.kinetic.min()-0.01
    x2=10**2
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(16,16))
    exp_names=df.experiment.unique()
    type_points=['X', 'P', 'p', 'o', '^', '*','<','>','v','h','d','s']
    markersize_flt=14
    i=0
    while i<len(exp_names):
        if (i<=1) or ((i>6) and i<10):
            l=exp_names[i]
            label_string=l[:l.find('(')]
            #print(label_string)
            plt.errorbar(df.loc[df.experiment==l,'kinetic'],df.loc[df.experiment==l,'ratio'],
                xerr=df.loc[df.experiment==l,'kinetic_binsize'],yerr=df.loc[df.experiment==l,'ratio_errors'],
                         marker=type_points[i],ms=markersize_flt,linestyle='None',label=label_string)
        elif i>=10:
            l=exp_names[i]
            #print(l)
            label_string=l[:l.find('1')]
            #print(label_string)
            plt.errorbar(df.loc[df.experiment==l,'kinetic'],df.loc[df.experiment==l,'ratio'],
                xerr=df.loc[df.experiment==l,'kinetic_binsize'],yerr=df.loc[df.experiment==l,'ratio_errors'],
                         marker=type_points[i],ms=markersize_flt,linestyle='None',label=label_string)
        i+=1
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    plt.legend(loc='lower right', fontsize=fnt-10)
    plt.title("Example ", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_all_data.png")
    #don't show on supercomputer
    #plt.show()

### Plot the data and save to filepaths declared directory ###
def make_plot_of_data(df,numerator,denominator,log_show):
    fnt=20
    x1=df.rigidity.values[0]-0.1
    x2=1.5*df.rigidity.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(df.rigidity.values,df.ratio.values,xerr=df.rigidity_binsize.values,yerr=df.ratio_errors.values,fmt='o',label="AMS")
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.title("Example", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_ams_data.png")
    #don't show on supercomputer
    #plt.show()

def load_H2_H1_ratio():
# lets try saving the fluxes per element as a dataframe?
    extension='.csv'
    read_file=filepaths.data_path+'H2_H1_pamela_voyager_data'+extension
    multi_exp_df=pd.read_csv(read_file)
    #print(multi_exp_df.head())
    column_names=['experiment','kinetic','kinetic_binsize','ratio','ratio_errors']
    names=multi_exp_df._experiment.values
    kinetic=np.array((multi_exp_df.EK_low.values,multi_exp_df.EK_high.values.T))
    kinetic_mp=multi_exp_df.EK_mid.values
    kinetic_binsize=(kinetic[1,:]-kinetic[0,:])/2.0
    ratio_name='_h2'+'_'+'h1_ratio'
    ratio=np.array(multi_exp_df[ratio_name].values)
    ratio_sys_errors=np.array(multi_exp_df._sys_plus.values)
    ratio_stat_errors=np.array(multi_exp_df._stat_plus.values)
    ratio_errors=np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    #print(kinetic_mp.shape)
    #print(ratio.shape)
    #print(ratio_errors.shape)
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack((names,kinetic_mp,kinetic_binsize,ratio,ratio_errors)),
                   columns=column_names)
    return ams_data_formatted_df


def make_plot_of_Hisotope_data(df,numerator,denominator,log_show):
    fnt=20
    x1=df.kinetic.min()-0.01
    x2=10**2
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(16,16))
    exp_names=df.experiment.unique()
    type_points=['P','p','o']
    markersize_flt=14
    i=0
    while i<len(exp_names):
        l=exp_names[i]
        label_string=l[:l.find('(')]
        print(label_string)
        plt.errorbar(df.loc[df.experiment==l,'kinetic'],df.loc[df.experiment==l,'ratio'],
            xerr=df.loc[df.experiment==l,'kinetic_binsize'],yerr=df.loc[df.experiment==l,'ratio_errors'],
                    marker=type_points[i],ms=markersize_flt,linestyle='None',label=label_string)
        i+=1
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    plt.legend(loc='lower right', fontsize=fnt-10)
    plt.title("Example ", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_all_data.png")
    #don't show on supercomputer
    #plt.show()


def load_He3_He4_ratio():
    file_name=filepaths.data_path+'he_3_4_ams_data.csv'  
    ams=pd.read_csv(file_name)
    #print(ams.head())
    #join low and high together as one array to be used as x error bars
    kinetic=np.array((ams.EK_low.values,ams.Ek_high.values.T))
    #kinetic=kinetic*1000
    kinetic_mp=(kinetic[0,:]+kinetic[1,:])/2.0
    # now make the error bar sizes (symmetric about these midpoints)
    kinetic_binsize=(kinetic[1,:]-kinetic[0,:])/2.0
    #make the ratio an array
    ratio=np.array(ams._3He_over_4He.values * ams._factor_ratio.values)
    ratio_sys_erros=np.array(ams._sys_ratio.values * ams._factor_ratio)
    ratio_stat_erros=np.array(ams._stat_ratio.values * ams._factor_ratio)
    ratio_errors=np.sqrt(np.square(ratio_stat_erros)+np.square(ratio_sys_erros))
    column_names=['kinetic','kinetic_binsize','ratio','ratio_errors']
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack((kinetic_mp,kinetic_binsize,ratio,ratio_errors)),
                   columns=column_names)
    return ams_data_formatted_df


### Plot the data and save to filepaths declared directory ###
def make_plot_of_Heisotope_data(df,numerator,denominator,log_show):
    fnt=20
    xmin=df.kinetic.min()
    x1=0.95*xmin
    xmax=df.kinetic.max()
    x2=1.05*xmax
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(df.kinetic.values,df.ratio.values,xerr=df.kinetic_binsize.values,yerr=df.ratio_errors.values,fmt='o',label="AMS")
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.title("Example", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_exp_data.png")
    #don't show on supercomputer
    #plt.show()

def load_B_C_ratio_voyager():
# lets try saving the fluxes per element as a dataframe?
    extension='.csv'
    read_file=filepaths.data_path+'B_C_test_unmodulated_voyager_ams'+extension
    multi_exp_df=pd.read_csv(read_file)
    #print(multi_exp_df.head())
    column_names=['experiment','kinetic','kinetic_binsize','ratio','ratio_errors']
    # get just the voyager1- B_C ratio data.
    names=multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")]._experiment.values
    kinetic=np.array((multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")].EK_low.values,multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")].EK_high.values.T))
    kinetic_mp=multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")].EK_mid.values
    kinetic_binsize=(kinetic[1,:]-kinetic[0,:])/2.0
    ratio_name='_B'+'_'+'C_ratio'
    ratio=np.array(multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")][ratio_name].values)
    ratio_sys_errors=np.array(multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")]._sys_plus.values)
    ratio_stat_errors=np.array(multi_exp_df[multi_exp_df['_experiment'].str.contains("Voyager1-")]._stat_plus.values)
    ratio_errors=np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    #print(kinetic_mp.shape)
    #print(ratio.shape)
    #print(ratio_errors.shape)
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack((names,kinetic_mp,kinetic_binsize,ratio,ratio_errors)),
                   columns=column_names)
    return ams_data_formatted_df

### Plot the data and save to filepaths declared directory ###
def make_plot_of_B_C_voyager_data(df,numerator,denominator,log_show):
    fnt=20
    xmin=df.kinetic.min()
    x1=0.95*xmin
    xmax=df.kinetic.max()
    x2=1.05*xmax
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(df.kinetic.values,df.ratio.values,xerr=df.kinetic_binsize.values,yerr=df.ratio_errors.values,fmt='o',label="Voyager")
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.title("Example", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_voyager_data.png")
    #don't show on supercomputer
    #plt.show()
