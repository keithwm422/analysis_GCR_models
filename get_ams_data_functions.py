#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)


# In[1]:


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


# In[2]:


def read_in_data(numerator,denominator,path):
    extension='ams_data.csv'
    read_file=path+numerator+'_'+denominator+'_'+extension
    ams=pd.read_csv(read_file)
    print(ams.head())
    return ams


# In[3]:


def make_plot_of_data(numerator,denominator,rigidity,ratio,rigidity_binsize,ratio_errors,log_show):
    fnt=20
    x1=rigidity[0]-0.1
    x2=1.5*rigidity[-1]
    #y1=ratio[0]
    #y2=5*10**-1
    plt.figure(figsize=(10,10))
    plt.errorbar(rigidity,ratio,xerr=rigidity_binsize,yerr=ratio_errors,fmt='o',label="AMS")
    #plt.plot(energy,he_3_4_2,'-o',label="L=2")
    #plt.plot(energy,he_3_4_3,'-o',label="L=3")
    #plt.plot(energy,he_3_4_4,'-o',label="L=4")
    #plt.plot(energy,he_3_4_5,'-o',label="L=5")
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
    plt.savefig(numerator+"_"+denominator+"_ams_data.png")
    plt.show()
