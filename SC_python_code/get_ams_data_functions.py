#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from decimal import Decimal
import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
import filepaths
from scipy.optimize import minimize
from scipy.interpolate import splev, splrep
### Takes the dataframe and returns the different columns separately including the calculated central rigidity bin value. For ratios
def make_energies_and_errors(df,num,den,tots_error):
    rigidity=np.array((df.R_low.values,df.R_high.values.T))
    rigidity_mp=(rigidity[0,:]+rigidity[1,:])/2.0
    rigidity_binsize=(rigidity[1,:]-rigidity[0,:])/2.0
    ratio_name='_'+str(num)+'_'+str(den)+'_'+'ratio'
    ratio=np.array(df[ratio_name].values)
    ratio_sys_errors=np.array(df._sys.values)
    ratio_stat_errors=np.array(df._stat.values)
    ratio_errors=ratio_stat_errors.copy() # just use statistical errors since systematic errors have correlation over energy bins and those can't be used yet.
    if tots_error==1:
        ratio_errors=np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    return rigidity_mp,rigidity_binsize,ratio,ratio_errors

### Takes the dataframe and returns the different columns separately including the calculated central rigidity bin value. For fluxes
def make_flux_energies_and_errors(df,name,tots_error):
    rigidity=np.array((df.R_low.values,df.R_high.values.T))
    rigidity_mp=(rigidity[0,:]+rigidity[1,:])/2.0
    rigidity_binsize=(rigidity[1,:]-rigidity[0,:])/2.0
    flux_name='_'+str(name)+'_flux'
    flux=np.array(df[flux_name].values)
    flux_sys_errors=np.array(df._sys.values)
    flux_stat_errors=np.array(df._stat.values)
    flux_errors=flux_stat_errors.copy() # just use statistical errors since systematic errors have correlation over energy bins and those can't be used yet.
    if tots_error==1:
        flux_errors=np.sqrt(np.square(flux_stat_errors)+np.square(flux_sys_errors))
    return rigidity_mp,rigidity_binsize,flux,flux_errors

### Load the data from a csv file into a Pandas DF ###
def read_in_data(numerator,denominator):
    extension='ams_data.csv'
    read_file=filepaths.data_path+"Ratios/"+numerator+'_'+denominator+'_'+extension
    ams=pd.read_csv(read_file)
    #print(ams.head())
    return ams
### Load the data from a csv file into a Pandas DF ###
def read_in_fluxdata(name):
    extension='_flux_ams.csv'
    read_file=filepaths.data_path+"fluxes/"+name+extension
    ams=pd.read_csv(read_file)
    #print(ams.head())
    return ams

# which error being 1 gives the quadrature of sys and stat, whereas any other value just gives you stat
def load_ams_ratios(numerator,denominator,which_error):
    # lets try returning the flux ratios as a dataframe
    df=read_in_data(numerator,denominator)
    column_names=['rigidity','rigidity_binsize','ratio','ratio_errors']
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack(make_energies_and_errors(df,numerator,denominator,which_error)),
                   columns=column_names)
    return ams_data_formatted_df

# which error being 1 gives the quadrature of sys and stat, whereas any other value just gives you stat
def load_ams_fluxes(name,which_error):
    # lets try returning the flux ratios as a dataframe
    df=read_in_fluxdata(name)
    column_names=['rigidity','rigidity_binsize','flux','flux_errors']
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack(make_flux_energies_and_errors(df,name,which_error)),
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
    read_file=filepaths.data_path+"Ratios/"+'Be10_Be9_all'+extension
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
    fnt=28
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
def make_plot_of_data(df,numerator,denominator,log_show,which_error):
    fnt=28
    x1=df.rigidity.values[0]-0.1
    x2=1.5*df.rigidity.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(df.rigidity.values,df.ratio.values,xerr=df.rigidity_binsize.values,yerr=df.ratio_errors.values,fmt='o',label="AMS")
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.title("Example", fontsize=fnt)
    if which_error==1:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_totserror_ams_data.png")
    else:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_ams_data.png")
    #don't show on supercomputer
    #plt.show()


### Plot the data and an example model and save to filepaths declared directory ###
def make_plot_of_data_and_model(df,numerator,denominator,log_show,which_error,model_x,model_y, L,D):
    fnt=28
    x1=df.rigidity.values[0]-0.1
    x2=1.5*df.rigidity.values[-1]
    y2=0.05+np.max(df.ratio.values)
    y1=np.min(df.ratio.values)-0.009
    plt.figure(figsize=(10,10))
    plt.errorbar(df.rigidity.values,df.ratio.values,xerr=df.rigidity_binsize.values,yerr=df.ratio_errors.values,fmt='o',label="AMS")
    plt.plot(model_x,model_y,'r--',label="Model L="+str(L)+",D="+str(D))
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.legend(loc='upper right', fontsize=fnt)
    plt.title("B/C model and data", fontsize=fnt)
    if which_error==1:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_totserror_ams_data_andmodel"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_ams_data_andmodel"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()

### Plot the ratio data and an example model with residuals and save to filepaths declared directory ###
def make_plot_of_data_and_modelresiduals(df,residuals,numerator,denominator,log_show,which_error,model_x,model_y, L,D,cutoff):
    fnt=28
    x1=df.rigidity.values[0]-0.1
    x2=1.5*df.rigidity.values[-1]
    y1=0.4
    y2=2*10**-2
    gs_kw = dict(width_ratios=[1], height_ratios=[2,1])
    #y2=1.5*df.flux.values[0]
    #y1=0.05*df.flux.values[-1]
    fig,ax=plt.subplots(figsize=(12, 18), dpi=400, nrows=2,sharex=True,gridspec_kw=gs_kw)
    fig.subplots_adjust(hspace=0)
    ax[0].errorbar(df.rigidity.values,df.ratio.values,xerr=df.rigidity_binsize.values,yerr=df.ratio_errors.values,fmt='o',label="AMS")
    ax[0].plot(model_x,model_y,'r--',label="Model L="+str(L)+",D="+str(D))
    ax[1].errorbar(df.rigidity.values[df.rigidity.values>cutoff],np.abs(residuals),yerr=df.ratio_errors.values[df.rigidity.values>cutoff],fmt='o',color='black', label="Residuals")
    ax[0].set_xscale("log")
    ax[1].set_xlabel("Rigidity [GV]",fontsize=fnt)
    ax[0].tick_params(labelsize=fnt-3)
    ax[1].tick_params(labelsize=fnt-3)
    ax[0].tick_params(which='both',width=2, length=7)
    ax[1].tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        ax[0].set_yscale("log")
        #ax[1].set_yscale("log")
    ax[0].set_ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    ax[0].set_xlim([x1,x2])
    ax[1].set_xlim([x1,x2])
    ax[0].set_ylim([y2,y1])
    ax[1].set_ylim([-0.02,0.02])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    ax[0].legend(loc='upper right', fontsize=fnt)
    ax[1].legend(loc='lower left', fontsize=fnt)
    ax[0].set_title("B/C model and data", fontsize=fnt)
    if which_error==1:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_totserror_ams_data_andmodelresiduals_"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_ams_data_andmodelresiduals_"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()

### Plot the data and an example model and save to filepaths declared directory ###
def make_plot_of_fluxdata_and_model(df,name,log_show,which_error,model_x,model_y, L,D):
    fnt=28
    x1=0.5*df.rigidity.values[0]
    x2=1.5*df.rigidity.values[-1]
    y2=1.5*df.flux.values[0]
    y1=0.05*df.flux.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(df.rigidity.values,df.flux.values,xerr=df.rigidity_binsize.values,yerr=df.flux_errors.values,fmt='o',label="AMS")
    plt.plot(model_x,model_y,'r--',label="Model L="+str(L)+",D="+str(D))
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux "r'm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$',fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.legend(loc='lower left', fontsize=fnt)
    plt.title(name+" Flux", fontsize=fnt)
    if which_error==1:
        plt.savefig(filepaths.images_path+"_"+name+"_totserror_ams_data_andmodel"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+"_"+name+"_ams_data_andmodel"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()

### Plot the data and an example model with subplots of the residuals and save to filepaths declared directory ###
def make_plot_of_fluxdata_and_modelresiduals(df,name,log_show,which_error,model_x,model_y, residuals,L,D):
    fnt=28
    x1=0.5*df.rigidity.values[0]
    x2=1.5*df.rigidity.values[-1]
    gs_kw = dict(width_ratios=[1], height_ratios=[2,1])
    y2=1.5*df.flux.values[0]
    y1=0.05*df.flux.values[-1]
    fig,ax=plt.subplots(figsize=(12, 18), dpi=400, nrows=2,sharex=True,gridspec_kw=gs_kw)
    fig.subplots_adjust(hspace=0)
    ax[0].errorbar(df.rigidity.values,df.flux.values,xerr=df.rigidity_binsize.values,yerr=df.flux_errors.values,fmt='o',color='black',label="AMS")
    ax[0].plot(model_x,model_y,'r--',label="Model L="+str(L)+",D="+str(D))
    ax[1].errorbar(df.rigidity.values,np.abs(residuals),yerr=df.flux_errors.values,fmt='o',color='black', label="Residuals")
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Rigidity [GV]",fontsize=fnt)
    ax[0].tick_params(labelsize=fnt-3)
    ax[1].tick_params(labelsize=fnt-3)
    #plt.xticks(fontsize=fnt-4)
    if log_show==1:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
    ax[0].set_ylabel("Flux "r'm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$',fontsize=fnt)
    ax[0].set_xlim([x1,x2])
    ax[1].set_xlim([x1,x2])
    ax[1].set_ylim([10**-10,10**0])
    ax[0].set_ylim([y1,y2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    ax[0].legend(loc='upper right', fontsize=fnt)
    ax[1].legend(loc='upper right', fontsize=fnt)
    #plt.suptitle(name+" Flux", fontsize=fnt,y=0)
    ax[0].set_title(name+" Flux", fontsize=fnt+2)
    if which_error==1:
        plt.savefig(filepaths.images_path+"_"+name+"_totserror_ams_data_andmodelresiduals"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+"_"+name+"_ams_data_andmodelresiduals"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()

def load_H2_H1_ratio():
# lets try saving the fluxes per element as a dataframe?
    extension='.csv'
    read_file=filepaths.data_path+"Ratios/"+'H2_H1_pamela_voyager_data'+extension
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
    fnt=28
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
    file_name=filepaths.data_path+"Ratios/"+'he_3_4_ams_data.csv'  
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
    fnt=28
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
    read_file=filepaths.data_path+"Ratios/"+'B_C_test_unmodulated_voyager_ams'+extension
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
    fnt=28
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


def make_residual_histogram(numerator,denominator,which_error,residuals,chi,L,D):
    fnt=28
    n_bins=20
    range_bins=[-0.02,0.02]
    #x1=df.rigidity.values[0]-0.1
    #x2=1.5*df.rigidity.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(12,12))
    bin_heights, bin_borders,_last_ = plt.hist(residuals,bins=n_bins, range=range_bins,histtype='step',linewidth=3)
    #plt.xscale("log")
    plt.xlabel("Residual",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    #if log_show==1:
    #   plt.yscale("log")
    #plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    #plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    #plt.legend(loc='upper right', fontsize=fnt)
    plt.title(f'{numerator}/{denominator} Model L='+str(L)+",D="+str(D), fontsize=fnt)
    plt.text(-0.015,5,s=r"$\chi ^{2}=$"+str(round(chi,3)),size=20,bbox=dict(boxstyle="round",
                  ec=(1., 0.5, 0.5),
                  fc=(1., 0.8, 0.8),
                  ))
    if which_error==1:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_totserror_residual"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_residual"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()

def make_residual_R_plot(numerator,denominator,which_error,residuals,chi,rigidity,L,D):
    fnt=28
    n_bins=20
    range_bins=[-0.02,0.02]
    #x1=df.rigidity.values[0]-0.1
    #x2=1.5*df.rigidity.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(12,12))
    plt.plot(rigidity[rigidity>65],residuals,linestyle='None',ms=8,marker='o')
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.ylabel("Residual",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    #if log_show==1:
    #   plt.yscale("log")
    #plt.ylabel("Flux division "+numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    #plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    #plt.legend(loc='upper right', fontsize=fnt)
    plt.title(f'{numerator}/{denominator} Model L='+str(L)+",D="+str(D), fontsize=fnt)
    plt.text(70,0.009,s=r"$\chi ^{2}=$"+str(round(chi,3)),size=20,bbox=dict(boxstyle="round",
                  ec=(1., 0.5, 0.5),
                  fc=(1., 0.8, 0.8),
                  ))
    if which_error==1:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_totserror_residual"+str(L)+"_"+str(D)+"R.png")
    else:
        plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_residual"+str(L)+"_"+str(D)+"R.png")
    #don't show on supercomputer
    #plt.show()

def chisq(A,flux_data,flux,errors):
    model=A*flux
    residuals=flux_data-model
    chi_square=np.sum(np.true_divide(np.square(residuals),np.square(errors)))
    return chi_square

# pass these as is and these will be logged and unlogged internally
def apply_scaler(rigidity_data,flux_data,rigidity,flux,errors):
    #now interpolate
    model_x_logged = np.log10(rigidity)
    model_y_logged = np.log10(flux)
    spl = splrep(model_x_logged,model_y_logged) # these should be logged, however, as that helps interpolation
    data_x_logged = np.log10(rigidity_data.copy())
    model_y_spline = splev(data_x_logged, spl)
    model_y_spline=10**(model_y_spline)
    res = minimize(chisq, 10 ,args=(flux_data,model_y_spline,errors), method='Nelder-Mead', tol=1e-6)
    scale_val=res.x
    flux*=scale_val
    print(f'scaled by {scale_val}')
    return flux


### Plot the data and an example model and save to filepaths declared directory ###
### Order matters, B,C,O preferrably
def make_plot_of_multifluxdata_and_modelamplitude(df1,name1,df2,name2,df3,name3,log_show,which_error,n_obj1,n_obj2,n_obj3,L,D):
    fnt=28
    x1=0.5*df1.rigidity.values[0]
    x2=1.5*df3.rigidity.values[-1]
    y2=1.5*df3.flux.values[0]
    y1=0.05*df1.flux.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    # correct the factor difference for the magnitudes of the fluxes for each element
    n1_flux_scaled=apply_scaler(np.array(df1.rigidity.values).copy(),np.array(df1.flux.values).copy(),n_obj1.rigidity_modulated.copy(),
                                                          n_obj1.flux_rigidity_modulated.copy(),np.array(df1.flux_errors.values).copy())
    n2_flux_scaled=apply_scaler(np.array(df2.rigidity.values).copy(),np.array(df2.flux.values).copy(),n_obj2.rigidity_modulated.copy(),
                                                          n_obj2.flux_rigidity_modulated.copy(),np.array(df2.flux_errors.values).copy())
    n3_flux_scaled=apply_scaler(np.array(df3.rigidity.values).copy(),np.array(df3.flux.values).copy(),n_obj3.rigidity_modulated.copy(),
                                                          n_obj3.flux_rigidity_modulated.copy(),np.array(df3.flux_errors.values).copy())
    external_factor=10**(-2)
    n3_flux_scaled*=external_factor
    plt.figure(figsize=(10,10))
    plt.errorbar(df1.rigidity.values,df1.flux.values,xerr=df1.rigidity_binsize.values,yerr=df1.flux_errors.values,fmt='o',label=name1+", AMS")
    plt.errorbar(df2.rigidity.values,df2.flux.values,xerr=df2.rigidity_binsize.values,yerr=df2.flux_errors.values,fmt='o',label=name2+", AMS")
    plt.errorbar(df3.rigidity.values,external_factor*df3.flux.values,xerr=df3.rigidity_binsize.values,yerr=external_factor*df3.flux_errors.values,fmt='o',
                                                                                                               label=f'{Decimal(external_factor):2.0E} x '+name3+", AMS")
    plt.plot(n_obj1.rigidity_modulated,n1_flux_scaled,'b--',label=n_obj1.name+", Model")
    plt.plot(n_obj2.rigidity_modulated,n2_flux_scaled,'y--',label=n_obj2.name+", Model")
    plt.plot(n_obj3.rigidity_modulated,n3_flux_scaled,'g--',label=f'{Decimal(external_factor):2.0E} x '+n_obj3.name+", Model")
    #plt.plot(n_obj2.rigidity_modulated,n_obj2.flux_rigidity_modulated,'y--',label=n_obj2.name+", Model")
    #plt.plot(n_obj3.rigidity_modulated,n_obj3.flux_rigidity_modulated,'g--',label=n_obj3.name+", Model")
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux "r'[m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]',fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.legend(loc='lower left', fontsize=fnt)
    plt.title("Spectra of Cosmic Ray Nuclei with Model L="+str(L)+",D="+str(D), fontsize=fnt)
    if which_error==1:
        plt.savefig(filepaths.images_path+"_multi_"+name1+"_"+name2+"_"+name3+"_totserror_ams_data_andmodelamplitude"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+"_multi_"+name1+"_"+name2+"_"+name3+"_ams_data_andmodelamplitude"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()


### Plot the data and an example model and save to filepaths declared directory ###
### Order matters, B,C,O preferrably
def make_plot_of_multifluxdata_and_model(df1,name1,df2,name2,df3,name3,log_show,which_error,n_obj1,n_obj2,n_obj3,L,D):
    fnt=28
    x1=0.5*df1.rigidity.values[0]
    x2=1.5*df3.rigidity.values[-1]
    y2=1.5*df3.flux.values[0]
    y1=0.05*df1.flux.values[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(df1.rigidity.values,df1.flux.values,xerr=df1.rigidity_binsize.values,yerr=df1.flux_errors.values,fmt='o',label=name1+", AMS")
    plt.errorbar(df2.rigidity.values,df2.flux.values,xerr=df2.rigidity_binsize.values,yerr=df2.flux_errors.values,fmt='o',label=name2+", AMS")
    plt.errorbar(df3.rigidity.values,df3.flux.values,xerr=df3.rigidity_binsize.values,yerr=df3.flux_errors.values,fmt='o',label=name3+", AMS")
    plt.plot(n_obj1.rigidity_modulated,n_obj1.flux_rigidity_modulated,'b--',label=n_obj1.name+", Model")
    plt.plot(n_obj2.rigidity_modulated,n_obj2.flux_rigidity_modulated,'y--',label=n_obj2.name+", Model")
    #plt.plot(n_obj3.rigidity_modulated,n_obj3.flux_rigidity_modulated,'g--',label=n_obj3.name+", Model")
    plt.plot(n_obj3.rigidity_modulated,n_obj3.flux_rigidity_modulated,'g--',label=n_obj3.name+", Model")
    plt.xscale("log")
    plt.xlabel("Rigidity [GV]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel("Flux "r'm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$',fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    #plt.legend(loc='lower right', fontsize=fnt-4)
    plt.legend(loc='lower left', fontsize=fnt)
    plt.title("Spectra of Cosmic Ray Nuclei with Model L="+str(L)+",D="+str(D), fontsize=fnt)
    if which_error==1:
        plt.savefig(filepaths.images_path+"_multi_"+name1+"_"+name2+"_"+name3+"_totserror_ams_data_andmodel"+str(L)+"_"+str(D)+".png")
    else:
        plt.savefig(filepaths.images_path+"_multi_"+name1+"_"+name2+"_"+name3+"_ams_data_andmodel"+str(L)+"_"+str(D)+".png")
    #don't show on supercomputer
    #plt.show()

def load_B_C_ratio_all_data():
# lets try saving the fluxes per element as a dataframe?
    extension='.csv'
    read_file=filepaths.data_path+"Ratios/"+'B_C_test_unmodulated_voyager_ams'+extension
    multi_exp_df=pd.read_csv(read_file)
    #print(multi_exp_df.head())
    column_names=['experiment','kinetic','kinetic_binsize','ratio','ratio_errors']
    # get just the voyager1- B_C ratio data.
    names=multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))]._experiment.values
    kinetic=np.array((multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))].EK_low.values,
                      multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))].EK_high.values.T))
    kinetic_mp=multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))].EK_mid.values

    kinetic_binsize=(kinetic[1,:]-kinetic[0,:])/2.0
    ratio_name='_B'+'_'+'C_ratio'
    ratio=np.array(multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))][ratio_name].values)
    ratio_sys_errors=np.array(multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))]._sys_plus.values)
    ratio_stat_errors=np.array(multi_exp_df[(multi_exp_df['_experiment'].str.contains("Voyager1-")) | 
                       (multi_exp_df['_experiment'].str.contains("AMS02"))]._stat_plus.values)
    ratio_errors=np.sqrt(np.square(ratio_stat_errors)+np.square(ratio_sys_errors))
    #print(kinetic_mp.shape)
    #print(ratio.shape)
    #print(ratio_errors.shape)
    ams_data_formatted_df = pd.DataFrame(data=np.column_stack((names,kinetic_mp,kinetic_binsize,ratio,ratio_errors)),
                   columns=column_names)
    return ams_data_formatted_df

def make_plot_of_B_C_all_data(numerator,denominator,log_show,energy,B_C_ratio_models,L,D):
    fnt=28
    df=load_B_C_ratio_all_data()
    x1=0.9*df.kinetic.min()
    x2=4*df.kinetic.max()
    y1=10**-2
    y2=3*10**-1
    plt.figure(figsize=(14,10))
    plt.errorbar(df[df['experiment'].str.contains("Voyager1-")].kinetic.values,
                 df[df['experiment'].str.contains("Voyager1-")].ratio.values,
                 xerr=df[df['experiment'].str.contains("Voyager1-")].kinetic_binsize.values,
                 yerr=df[df['experiment'].str.contains("Voyager1-")].ratio_errors.values,
                 marker="o",ms=10,linestyle='None',label="Voyager LIS")
    plt.errorbar(df[(df['experiment'].str.contains("AMS02")) & (df['kinetic']>20)].kinetic.values,
                 df[(df['experiment'].str.contains("AMS02"))  & (df['kinetic']>20)].ratio.values,
                 xerr=df[(df['experiment'].str.contains("AMS02"))  & (df['kinetic']>20)].kinetic_binsize.values,
                 yerr=df[(df['experiment'].str.contains("AMS02"))  & (df['kinetic']>20)].ratio_errors.values,
                 marker="x",ms=10,linestyle='None',label="AMS-02")
    plt.plot(energy[0],B_C_ratio_models[0],'--',linewidth=3,label=f'model L={L[0]}, D={D[0]}')
    plt.plot(energy[1],B_C_ratio_models[1],'-.',linewidth=3,label=f'model L={L[1]}, D={D[1]}')
    plt.plot(energy[2],B_C_ratio_models[2],'-',linewidth=3,label=f'model L={L[2]}, D={D[2]}')
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)  
    if log_show==1:
        plt.yscale("log")
    plt.ylabel(numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)  
    plt.xlim([x1,x2])
    plt.ylim([y1,y2])
    plt.legend(loc='upper right', fontsize=fnt-4)
    #plt.setp(plt.axis.tick_params(axis=both, width=3,length=3))
    plt.title("Boron Carbon Ratio, data and models", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_ams_and_voyager_data.png")
    #don't show on supercomputer
    #plt.show()

def make_plot_of_Beisotope_data_and_models(numerator,denominator,log_show,energy,be_10_be_9_model_ratio,L,D):
    fnt=28
    df=load_Be10_Be9_ratio()
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
    plt.plot(energy[0],be_10_be_9_model_ratio[0],'g--',linewidth=3,label=f'model L={L[0]}, D={D[0]}')
    plt.plot(energy[1],be_10_be_9_model_ratio[1],'r-.',linewidth=3,label=f'model L={L[1]}, D={D[1]}')
    plt.plot(energy[2],be_10_be_9_model_ratio[2],'m-',linewidth=3,label=f'model L={L[2]}, D={D[2]}')
    plt.xscale("log")
    plt.xlabel("Kinetic Energy per nucleon [GeV/n]",fontsize=fnt)
    plt.xticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    if log_show==1:
        plt.yscale("log")
    plt.ylabel(numerator+"/"+denominator,fontsize=fnt)
    plt.yticks(fontsize=fnt-4)
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xlim([x1,x2])
    #plt.ylim([y1,y2])
    plt.legend(loc='lower right', fontsize=fnt-8)
    plt.title("Be-10/Be-9 ratio data and models ", fontsize=fnt)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_all_data_and_models.png")
    #don't show on supercomputer
    #plt.show()
