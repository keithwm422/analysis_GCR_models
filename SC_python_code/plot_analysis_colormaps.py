import numpy as np
import pandas as pd
# Set up matplotlib and use a nicer set of plot parameters
import matplotlib
#matplotlib.rc_file("../../templates/matplotlibrc")
matplotlib.use('agg')
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import filepaths
from scipy import interpolate

def plot_B_C_fit_index(seq):
    spectral_index_nparray=np.empty([20,20])
    spectral_index_nparray = np.loadtxt('spectralindexfits_-0.33nominal.txt')
    fnt=24
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(np.transpose(spectral_index_nparray),cmap='plasma',origin='lower',vmax=-0.315, vmin=-0.345)
    #print(chi_new.shape)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([i for i in range(0,20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    ax.set_yticks([i for i in range(0,20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    ax.set_xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)', fontsize = fnt-2)
    ax.set_ylabel("Halo Size (kpc)", fontsize = fnt-2)
    cb=fig.colorbar(cax)
    #cb = colorbar() # grab the Colorbar instance
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fnt-4)
    plt.title("Power Law Index, "r'$\Delta$'" Fit to B/C",fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"spectral_index_colormap_tuning.png",dpi=400)

def plot_input_diffusion_R_index(seq):
    fnt=24
    spectral_index_nparray=np.empty([20,20])
    spectral_index_nparray =np.genfromtxt(open("diffusion_index_LUT.csv", "rb"), delimiter=",",dtype=float)
    #spectral_index_nparray = np.loadtxt('diffusion_index_LUT.csv')
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(spectral_index_nparray,cmap='plasma',origin='lower',vmax=0.5, vmin=0.1)
    #print(chi_new.shape)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([i for i in range(0,20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    ax.set_yticks([i for i in range(0,20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    ax.set_xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)', fontsize = fnt-2)
    ax.set_ylabel("Halo Size (kpc)", fontsize = fnt-2)
    cb=fig.colorbar(cax)
    #cb = colorbar() # grab the Colorbar instance
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fnt-4)
    plt.title("Power Law Index, "r'$\gamma$'" for Diffusion R-dependence",fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"spectral_index_colormap_diffusionR.png",dpi=400)

def plot_ratio_chisquare_with_interpolation(seq,numerator,denominator):
    # now make an interpolation on these values L and D values of chi square
    fnt=24
    chi_nparray=np.empty([20,20])
    chi_nparray =np.loadtxt('B_C_chisquare_error1.txt')
    chi_transposed_nparray=np.transpose(chi_nparray).copy()
    #print(chi_transposed_nparray.shape)
    #chi_real_transposed_1=np.transpose(chi_real.copy())
    #model_real_transposed=np.transpose(model_real)
    L=np.arange(1,21,1)
    D=np.arange(1,21,1)
    #chi_interp = interpolate.interp2d(D, L, chi_real_transposed)
    smoothing=0
    degree_x=3
    degree_y=3
    chi_real_transposed=np.log10(chi_transposed_nparray)
    print(chi_real_transposed.shape)
    chi_interp = interpolate.RectBivariateSpline(D,L,chi_real_transposed,kx=degree_x,ky=degree_y,s=smoothing)
    L_new=np.arange(1,20.1,0.1)
    D_new=np.arange(1,20.1,0.1)
    #chi_new=chi_interp(D_new,L_new)
    chi_new=chi_interp(D_new,L_new)
    print(np.where(chi_new==0))
    #print(chi_new)
    #better yet do the colormap to see
    chi_new=10**(chi_new)
    print(chi_new)
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(chi_new,cmap='plasma',origin='lower',norm=LogNorm(vmin=10, vmax=10**5))
    print(chi_new.shape)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([10*i for i in range(0,20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    ax.set_yticks([10*i for i in range(0,20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    ax.set_xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)', fontsize = fnt-2)
    ax.set_ylabel("Halo Size (kpc)", fontsize = fnt-2)
    cb=fig.colorbar(cax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fnt-4)
    plt.title(numerator + "/" + denominator+r' $\chi ^{2}$',fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"chi_square_colormap_spline.png",dpi=400)
