#!/usr/bin/env python3
# this code opens a file of chi_square values and plots them
#color map of L vs D for those values
#author Keith McBride, 2021-1

import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
#from astropy.io import fits
#from scipy.interpolate import splev, splrep
from matplotlib.colors import LogNorm
import seaborn as sns


def make_plot(seq):
    with open("chisquarefile.txt", 'r') as file:
        read_data = file.read()
    file.close()
    #print(f'value read: {read_data}')
    print(len(read_data))
    print(type(read_data))
    values=read_data.split()
    print(values[10])
    print(values[11])
    print(f'length of the values read {len(values)}')
    # odd values are the filenames, so parse those to get the L and the D
    i=0
    chi_squares_full=[]
    L=[]
    D=[]
    chi_squares=[]
    while i<len(values):
        # the first value is L=10, D=1, the second value is L=11, D=1. So the first 20*2 will be D=1 and L in a strange order.
        if (i>0) and (((i % 40)==0) or (i==799)):  # once its the last value which wont be taken into account with mod 40
            # then we are at the next diffusion value
            #change the order of the chi_squares
            #j=0
            temp=chi_squares.copy()
            #print(f'i={i} and temp before adjustments {temp}')
            #while j<len(chi_squares):
                #if j<=9: #this is L=10 so set it 9th position in array, j==1 L=11 set it to 10 pos in array... j==9 L=19 set it to 18 pos in array
                    #temp[j+9]=chi_squares[j]
                    #print(f'j={j} and j should be leq 9')
                #elif j==10: # this is L=1 so set it to zeroth pos
                    #temp[0]=chi_squares[j]
                    #print(f'j={j} and j should be 10')
                #elif j==11: #this is L=20 so set it to last pos
                    #temp[19]=chi_squares[j]
                    #print(f'j={j} and j should be 11')
                #elif j>=12:  # these are j>=12 where j==12 is L=2 so set to 1 pos in array, j==13 set it 2 in array
                    #temp[j-11]=chi_squares[j]
                    #print(f'j={j} and j should be geq 12')
                    #print(f'j-11={j-11} and temp should be {chi_squares[j]} but temp is {temp[j-11]}')                
                #j+=1
            chi_squares_full.append(temp)
            #print(f'temp after adjustments {temp}')
            chi_squares=[]
        if (i % 2) == 0:
            chi_squares.append(float(values[i]))        
        else: 
            L_D_values=values[i].split("_")
            #print(L_D_values)
            L.append(int(L_D_values[-3]))
            D.append(int(L_D_values[-1]))        
        i+=1
    print(f'num rows: {len(chi_squares_full)}')
    print(f'num columns: {len(chi_squares_full[0])}')
    #print(L)
    #print(D)
    L_values=np.arange(1,10,10)
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
    plt.title("Chi-Square")
    plt.savefig("heatmap_example_chisquarefileB_C.png",dpi=400)

