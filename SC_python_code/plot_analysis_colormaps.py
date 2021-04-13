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
    chi_nparray =np.loadtxt(numerator+'_'+denominator+'_chisquare_error1.txt')
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
    chi_real_transposed=10**(chi_real_transposed)
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
    return chi_real_transposed, chi_new
def plot_bery_ratio_chisquare_with_interpolation(seq):
    # now make an interpolation on these values L and D values of chi square
    fnt=24
    chi_nparray=np.empty([20,20])
    #chi_nparray =np.loadtxt('bery_chisquare_error1.txt')
    chi_nparray =np.loadtxt('chisquare_cutoff_-0.33nominal_error_1be.txt')
    chi_transposed_nparray=np.transpose(chi_nparray).copy()
    print(np.amin(chi_transposed_nparray))
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
    print(np.min(chi_new))
    #print(chi_new)
    #better yet do the colormap to see
    chi_new=10**(chi_new)
    chi_real_transposed=10**(chi_real_transposed)
    print(chi_new)
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(chi_new,cmap='plasma',origin='lower',norm=LogNorm(vmin=1.1, vmax=10**2))
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
    plt.title("Be-10/Be-9"+r' $\chi ^{2}$',fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"bery_chi_square_colormap_spline.png",dpi=400)
    return chi_real_transposed, chi_new

def plot_bery_ratio_chisquare(seq):
    chi_nparray=np.empty([20,20])
    chi_nparray = np.loadtxt('bery_chisquare_error1.txt')
    fnt=24
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(np.transpose(chi_nparray),cmap='plasma',origin='lower',vmax=100, vmin=10)
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
    plt.title("Be-10/Be-9 "r'$\chi ^{2}$',fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"bery_chi_square_colormap.png",dpi=400)

#plot_ratio_chisquare_with_interpolation(seq,numerator,denominator)
def plot_CI_ratio(seq,numerator,denominator):
# find the chi square values that are within 6.14 of the minimum model chi_square
    fnt=24
    chi_real_transposed_1,chi_new=plot_ratio_chisquare_with_interpolation(seq,numerator,denominator)
    delta_chi=6.14
    #models_new=model_real_transposed.copy()
    min_chi_square=np.amin(chi_real_transposed_1)
    min_chi_tuple=np.where(chi_real_transposed_1==min_chi_square)
    print(min_chi_square)
    tuples_found=np.where(chi_new<(min_chi_square+delta_chi))
    #models_new=model_real_transposed.copy()
    print(tuples_found[0])
    L_s_allowed=tuples_found[0].copy()
    print(L_s_allowed)
    # change to actual halo size values
    L_s_allowed=np.true_divide(L_s_allowed,10)+1
    D_s_allowed=tuples_found[1].copy()
    D_s_allowed=np.true_divide(D_s_allowed,10)+1
    plt.figure(figsize=(12,12))
    # these are for the minimum model add a nice big star to the minimum chi model found
    x=np.array(8)
    y=np.array(11)
    z=1
    plt.plot(D_s_allowed,L_s_allowed,color="red",marker="p",ms=5,linestyle='None',label="95\%")
    plt.scatter(x,y,s=400,marker="x",label='Minimum')
    plt.legend(loc='upper right',fontsize=fnt-4)
    plt.xticks(fontsize=fnt-4)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([1,20])
    plt.ylim([1,20])
    plt.ylabel("halo size (kpc)",fontsize=fnt-4)
    plt.xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)',fontsize=fnt-4)
    plt.title(numerator + "/" + denominator+r' $\Delta chi ^{2} \leq 6.14$',fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_halo-diffusion_confidence_interval.png",dpi=400)

#plot_ratio_chisquare_with_interpolation(seq,numerator,denominator)
def plot_CI_bery_ratio(seq):
# find the chi square values that are within 6.14 of the minimum model chi_square
    fnt=24
    chi_real_transposed_1,chi_new=plot_bery_ratio_chisquare_with_interpolation(seq)
    delta_chi=6.14
    #models_new=model_real_transposed.copy()
    min_chi_square=np.amin(chi_real_transposed_1)
    min_chi_tuple=np.where(chi_real_transposed_1==min_chi_square)
    print(min_chi_square)
    print(min_chi_tuple)
    tuples_found=np.where(chi_new<(min_chi_square+delta_chi))
    #models_new=model_real_transposed.copy()
    print(tuples_found[0])
    L_s_allowed=tuples_found[0].copy()
    print(L_s_allowed)
    # change to actual halo size values
    L_s_allowed=np.true_divide(L_s_allowed,10)+1
    D_s_allowed=tuples_found[1].copy()
    D_s_allowed=np.true_divide(D_s_allowed,10)+1
    plt.figure(figsize=(12,12))
    # these are for the minimum model add a nice big star to the minimum chi model found
    x=np.array(min_chi_tuple[1]+1)
    y=np.array(min_chi_tuple[0]+1)
    print(x)
    print(y)
    z=1
    plt.plot(D_s_allowed,L_s_allowed,color="red",marker="p",ms=5,linestyle='None',label="95\%")
    plt.scatter(x,y,s=400,marker="x",label='Minimum')
    plt.legend(loc='upper right',fontsize=fnt-4)
    plt.xticks(fontsize=fnt-4)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([1,20])
    plt.ylim([1,20])
    plt.ylabel("halo size (kpc)",fontsize=fnt-4)
    plt.xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)',fontsize=fnt-4)
    plt.title("Be-10/Be-9 "r'$\Delta \chi ^{2} \leq 6.14$',fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"Be10_Be9"+"_halo-diffusion_confidence_interval.png",dpi=400)
