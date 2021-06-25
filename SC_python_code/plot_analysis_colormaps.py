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
from scipy.interpolate import splev, splrep
def plot_B_C_fit_index(seq):
    spectral_index_nparray=np.empty([20,20])
    spectral_index_nparray = np.loadtxt(filepaths.outputs_path+'spectralindexfits_-0.33nominal.txt')
    fnt=28
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
    fnt=28
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
    fnt=28
    chi_nparray=np.empty([20,20])
    #chi_nparray =np.loadtxt(numerator+'_'+denominator+'_chisquare_error1.txt')
    chi_nparray =np.loadtxt(filepaths.outputs_path+'chisquare_cutoff_-0.33nominal_error_1b_c.txt')
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
    min_chi_square=np.amin(chi_real_transposed)
    min_chi_tuple=np.where(chi_real_transposed==min_chi_square)
    print(min_chi_square)
    x=np.array(min_chi_tuple[1]+1)
    y=np.array(min_chi_tuple[0]+1)
    print(x)
    print(y)
    #print(f'new::{L_new}')
    #print(f'len:{len(L_new)}')
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
    fnt=28
    chi_nparray=np.empty([20,20])
    #chi_nparray =np.loadtxt('bery_chisquare_error1.txt')
    chi_nparray =np.loadtxt(filepaths.outputs_path+'chisquare_cutoff_-0.33nominal_error_1be.txt')
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


def plot_ratio_chisquare(seq,numerator,denominator):
    # now make an interpolation on these values L and D values of chi square
    fnt=28
    chi_nparray=np.empty([20,20])
    #chi_nparray =np.loadtxt(numerator+'_'+denominator+'_chisquare_error1.txt')
    chi_nparray =np.loadtxt(filepaths.outputs_path+'chisquare_cutoff_-0.33nominal_error_1b_c.txt')
    chi_transposed_nparray=np.transpose(chi_nparray).copy()
    #print(chi_transposed_nparray.shape)
    #chi_real_transposed_1=np.transpose(chi_real.copy())
    #model_real_transposed=np.transpose(model_real)
    L=np.arange(1,21,1)
    D=np.arange(1,21,1)
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(chi_transposed_nparray,cmap='plasma',origin='lower',norm=LogNorm(vmin=10, vmax=10**5))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks([i for i in range(0,20)])
    ax.set_xticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    ax.set_yticks([i for i in range(0,20)])
    ax.set_yticklabels([str(i) for i in range(1,21)], fontsize=fnt-4)
    #ax.set_yticklabels([Categories[i] for i in range(20)],fontsize=14)
    #ax.colorbar()
    ax.set_xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)', fontsize = fnt-2)
    ax.set_ylabel("Halo Size (kpc)", fontsize = fnt-2)
    cb=fig.colorbar(cax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fnt-4)
    plt.title(numerator + "/" + denominator+r' $\chi ^{2}$',fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"chi_square_colormap.png",dpi=400)
    return chi_transposed_nparray

def plot_bery_ratio_chisquare(seq):
    chi_nparray=np.empty([20,20])
    chi_nparray = np.loadtxt(filepaths.outputs_path+'bery_chisquare_error1.txt')
    fnt=28
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
    fnt=28
    chi_real_transposed_1,chi_new=plot_ratio_chisquare_with_interpolation(seq,numerator,denominator)
    delta_chi=34
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
    x=np.array(min_chi_tuple[1]+1)
    y=np.array(min_chi_tuple[0]+1)
    print(x)
    print(y)
    z=1
    plt.plot(D_s_allowed,L_s_allowed,color="red",marker="p",ms=1,linestyle='None',label="95\%")
    plt.scatter(x,y,s=400,marker="x",label='Minimum')
    plt.legend(loc='upper right',fontsize=fnt-4)
    plt.xticks(fontsize=fnt-4)
    plt.yticks(fontsize=fnt-4)
    plt.xlim([1,20])
    plt.ylim([1,20])
    plt.ylabel("halo size (kpc)",fontsize=fnt-4)
    plt.xlabel("Diffusion Coefficient "r'($10^{28}$cm$^{2}$s$^{-1}$)',fontsize=fnt-4)
    plt.title(numerator + "/" + denominator+r' $\Delta \chi ^{2} \leq $'+str(delta_chi),fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+numerator+"_"+denominator+"_halo-diffusion_confidence_interval"+str(delta_chi)+".png",dpi=400)
    return L_s_allowed,D_s_allowed,x,y,chi_new,min_chi_square,delta_chi
#plot_ratio_chisquare_with_interpolation(seq,numerator,denominator)
def plot_CI_bery_ratio(seq):
# find the chi square values that are within 6.14 of the minimum model chi_square
    fnt=28
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
    plt.plot(D_s_allowed,L_s_allowed,color="red",marker="p",ms=1,linestyle='None',label="95\%")
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


def plot_bery_ratio_KE(seq):
    chi_nparray=np.empty([20,20])
    chi_nparray = np.loadtxt(filepaths.outputs_path+'be10_be9_ratio_KE_2.txt')
    fnt=28
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(np.transpose(chi_nparray),cmap='plasma',origin='lower',vmax=np.max(chi_nparray), vmin=np.min(chi_nparray))
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
    plt.title("Be-10/Be-9 at 2 GeV/n",fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"bery_ratio_colormap.png",dpi=400)

def plot_bery_ratio_KE_with_chisquare(seq):
    ls,ds,x,y=plot_CI_ratio(1,'B','C')
    chi_nparray=np.empty([20,20])
    chi_nparray = np.loadtxt(filepaths.outputs_path+'be10_be9_ratio_KE_2.txt')
    fnt=28
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(np.transpose(chi_nparray),cmap='plasma',origin='lower',vmax=np.max(chi_nparray), vmin=np.min(chi_nparray))
    plt.scatter(x-1,y-1,s=400,marker="x",label='Minimum',zorder=1)
    plt.plot(ds-1,ls-1,color="red",marker="p",ms=1,linestyle='None',label="95\%",zorder=1,alpha=0.25)
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
    plt.title("Be-10/Be-9 at 2 GeV/n",fontsize=fnt,y=1)
    plt.savefig(filepaths.images_path+"bery_ratio_colormap_with_chisquare.png",dpi=400)
# what if we find the borders and plot those only?
def plot_bery_ratio_KE_with_chisquareborders(seq,KE):
    ls,ds,x,y,chi_new,min_chi_square,delta_chi=plot_CI_ratio(1,'B','C')
    #print(f'ls: {ls-1}')
    #print(f'ds: {ds-1}')
    #print(f'{len(ls)}')
    #allowed_space=np.zeros(chi_new.shape)
    #print(allowed_space.shape)
    #print(chi_new<(min_chi_square+delta_chi))
    allowed_space=np.where(chi_new<(min_chi_square+delta_chi),1,0)
    print(allowed_space[0,0])
    num_ls=len(chi_new[0]) # to get the L,D model chi use chi_new[L,D]
    # for constant L, find the first and last D (if any) and
    l_border=[]
    d_border=[]
    i_l=0
    while i_l<num_ls:
        i_d=0
        while i_d<num_ls:
            if allowed_space[i_l,i_d]==1:
                l_border.append(float(i_l+1)/10)
                d_border.append(float(i_d+1)/10)
                break
            i_d+=1
        i_l+=1
    # now go backwards
    i_l=num_ls-1
    while i_l>=0:
        i_d=num_ls-1
        while i_d>=0:
            if allowed_space[i_l,i_d]==1:
                l_border.append(float(i_l+1)/10)
                d_border.append(float(i_d+1)/10)
                break
            i_d-=1
        i_l-=1
    d_border_array=np.array(d_border)
    l_border_array=np.array(l_border)
    print(d_border_array)
    chi_nparray=np.empty([20,20])
    chi_nparray = np.loadtxt(filepaths.outputs_path+'be10_be9_ratio_KE_'+str(KE)+'.txt')
    fnt=28
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(np.transpose(chi_nparray),cmap='summer',origin='lower',vmax=0.35, vmin=0.01)
    #cax=ax.matshow(np.transpose(chi_nparray),cmap='summer',origin='lower',vmax=np.max(chi_nparray)-0.05, vmin=np.min(chi_nparray))
    plt.scatter(x-1,y-1,s=400,marker="x",color="black",label='Minimum',zorder=1)
    #plt.plot(ds-1,ls-1,color="red",marker="p",ms=1,linestyle='None',label="95\%",zorder=1,alpha=0.5)
    plt.plot(d_border_array,l_border_array,color="white",marker="o",ms=1.5,label="B/C region",zorder=1)
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
    plt.title("Be-10/Be-9 at "+str(KE)+" GeV/n",fontsize=fnt,y=1)
    plt.legend(loc='upper right',fontsize=fnt-4)
    plt.savefig(filepaths.images_path+"bery_ratio_colormap_with_chisquareborders"+str(KE)+".png",dpi=400)

#it is assumed that x and y are understood by the user (no extra prep is done to x or y)
def splineHELIX(x,y,num_steps,minimum,maximum,smoothin):
    spl = splrep(x,y,s=smoothin)
    x_cont = np.linspace(minimum, maximum, num_steps)
    y_spline = splev(x_cont, spl)
    return x_cont, y_spline

# with above, we got res as delta_m over m
# need delta_m per mass (10 and 9)
def find_sigmas(mass_1,mass_2,resolution_of_point):
    #mass_be_10=10
    #mass_be_9=9
    delta_m2=resolution_of_point*(mass_2)
    sigma_m2=0.84932*delta_m2
    delta_m1=resolution_of_point*(mass_1)
    sigma_m1=0.84932*delta_m1;
    print(delta_m1)
    print(sigma_m1)
    return sigma_m1,sigma_m2

def read_inHELIX(seq,smoothin):
    helix_mass_res_df=pd.read_csv(filepaths.outputs_path+'helix_mass_res.csv')
    helix_mass_res_df.tail()
    energy_array_logged=np.log10(helix_mass_res_df.Energy.values)
    mass_array_logged=np.log10(helix_mass_res_df.mass_res.values)
    # Now do a spline and plot that on these points
    # separate the regions of the spline
    num_steps=2000
    tof_start=0
    tof_end=0
    max_differ=0
    j=0
    while j<len(mass_array_logged)-1:
        difference=abs(mass_array_logged[j+1]-mass_array_logged[j])
        if(difference>max_differ):
            max_differ=difference
            tof_end=j
        j+=1
    print(tof_end)
    rich_start=tof_end+1
    spline_x_TOF,spline_y_TOF=splineHELIX(energy_array_logged[tof_start:tof_end],mass_array_logged[tof_start:tof_end],
                                 num_steps,energy_array_logged[tof_start],energy_array_logged[tof_end],smoothin)
    spline_x_RICH,spline_y_RICH=splineHELIX(energy_array_logged[rich_start:],mass_array_logged[rich_start:],
                                 num_steps,energy_array_logged[rich_start],energy_array_logged[-1],smoothin)
    # Try taking the logarithm
    energy_array_logged=np.log10(helix_mass_res_df.Energy.values)
    mass_array_logged=np.log10(helix_mass_res_df.mass_res.values)
    # unlog and plot again
    spline_x_TOF_reg=10**(spline_x_TOF)
    spline_y_TOF_reg=10**(spline_y_TOF)
    spline_x_RICH_reg=10**(spline_x_RICH)
    spline_y_RICH_reg=10**(spline_y_RICH)
    fnt=28
    y1=0.009
    y2=1*10**-1
    x1=8*10**-2
    x2=2*10**1
    rs=np.array([x1,x2])
    ys=np.array([0.025,0.025])
    plt.figure(figsize=(14,10))
 #   plt.plot(helix_mass_res_df.Energy.values,helix_mass_res_df.mass_res.values,'bo',label="HELIX simulation")
    plt.plot(spline_x_TOF_reg,spline_y_TOF_reg,'b--',linewidth=2, label="HELIX simulation")
    plt.plot(spline_x_TOF_reg,spline_y_TOF_reg,'b--',linewidth=2)
    plt.plot(spline_x_RICH_reg,spline_y_RICH_reg,'b--',linewidth=2)
    plt.plot(rs,ys,'k-.',linewidth=1,label=r'$2.5$\% limit')
    plt.xticks(fontsize=fnt-4)
    plt.xlabel("Kinetic Energy [GeV/n]",fontsize=fnt-2) 
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.yticks(fontsize=fnt-4)
    plt.ylabel(r'$\delta m/m$',fontsize=fnt-2) 
    ax=plt.gca()
    ax.tick_params(which='both',width=2, length=7)
    ax.tick_params(which='minor',width=1.2, length=4)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='upper left',fontsize=fnt-4)
    plt.ylim([y1,y2])
    plt.xlim([x1,x2])
    plt.title("HELIX Mass Resolution",fontsize=fnt)
    plt.savefig(filepaths.images_path+"mass_resolutions_helix_data.png",dpi=400)
    #use energy per nucleon
    KE_point=2 #GeV/nucleon
    #convert to energy per nucleon
    E_point=KE_point+0.938
    # find the closest point
    array=spline_x_RICH_reg.copy()
    value=E_point
    absolute_val_array = np.abs(array - value)
    smallest_difference_index = absolute_val_array.argmin()
    closest_x_value=array[smallest_difference_index]
    resolution_of_point=spline_y_RICH_reg[smallest_difference_index]
    mass_be_9=9
    mass_be_10=10
    sigma_be_9,sigma_be_10=find_sigmas(mass_be_9,mass_be_10,resolution_of_point)
    print(sigma_be_9)

# what if we find the borders and plot those only?
def plot_bery_ratio_KE_with_chisquareborders_smooth(seq,KE):
    ls,ds,x,y,chi_new,min_chi_square,delta_chi=plot_CI_ratio(1,'B','C')
    #print(f'ls: {ls-1}')
    #print(f'ds: {ds-1}')
    #print(f'{len(ls)}')
    #allowed_space=np.zeros(chi_new.shape)
    #print(allowed_space.shape)
    #print(chi_new<(min_chi_square+delta_chi))
    allowed_space=np.where(chi_new<(min_chi_square+delta_chi),1,0)
    print(allowed_space[0,0])
    num_ls=len(chi_new[0]) # to get the L,D model chi use chi_new[L,D]
    # for constant L, find the first and last D (if any) and
    l_border=[]
    d_border=[]
    l_border_left=[]
    d_border_left=[]
    l_border_right=[]
    d_border_right=[]
    i_l=0
    while i_l<num_ls:
        i_d=0
        while i_d<num_ls:
            if allowed_space[i_l,i_d]==1:
                l_border.append(float(i_l+1)/10)
                d_border.append(float(i_d+1)/10)
                l_border_left.append(float(i_l+1)/10)
                d_border_left.append(float(i_d+1)/10)
                break
            i_d+=1
        i_l+=1
    # now go backwards
    i_l=num_ls-1
    while i_l>=0:
        i_d=num_ls-1
        while i_d>=0:
            if allowed_space[i_l,i_d]==1:
                l_border.append(float(i_l+1)/10)
                d_border.append(float(i_d+1)/10)
                l_border_right.append(float(i_l+1)/10)
                d_border_right.append(float(i_d+1)/10)
                break
            i_d-=1
        i_l-=1
    d_border_array=np.array(d_border) # note these values are reported as -1 of the actual D values (so that they line up on a grid starting from 0. 
    l_border_array=np.array(l_border)
    d_border_array_left=np.array(d_border_left)
    l_border_array_left=np.array(l_border_left)
    d_border_array_right=np.array(d_border_right)
    l_border_array_right=np.array(l_border_right)
    # put them in the same order by reversing the right array
    d_border_array_right=np.flip(d_border_array_right)
    l_border_array_right=np.flip(l_border_array_right)
    print(l_border_array_left)
    print(l_border_array_right)

    # for each halo size, in between 2.5 and 3.5, then 3.5 and 4.5, etc find the largest diff in diffusion
    l_start=2.5 # these need to be translated by 1 to give a gridspace value recorded in the arrays
    l_last=20.25
    l_step=0.5 # step from l_start by l_step until l_last
    l_grid_start=l_start-1
    l_grid_last=l_last-1
    new_l_vals_left=[]
    new_l_vals_right=[]
    new_d_vals_left=[]
    new_d_vals_right=[]
    while l_grid_start<l_grid_last:
        print(l_grid_start)
        l_templ=l_border_array_left[np.abs(l_border_array_left-l_grid_start-l_step/2.0)<=l_step/2.0] # get a temp array for left and right halo sizes that is within this region
        l_tempr=l_border_array_right[np.abs(l_border_array_right-l_grid_start-l_step/2.0)<=l_step/2.0] # get a temp array for left and right halo sizes that is within this region
        d_templ=d_border_array_left[np.abs(l_border_array_left-l_grid_start-l_step/2.0)<=l_step/2.0]   # diffusion uses conditions for halo size regions
        d_tempr=d_border_array_right[np.abs(l_border_array_right-l_grid_start-l_step/2.0)<=l_step/2.0] # get the diffusion values for these as well
        d_diff_temp=d_tempr-d_templ # get the difference in D for this region
        diff_amt=np.max(d_diff_temp) # max difference in this region
        new_d_vals_left.append(d_templ[np.argmax(d_diff_temp)])
        new_d_vals_right.append(d_tempr[np.argmax(d_diff_temp)])
        new_l_vals_left.append(l_templ[np.argmax(d_diff_temp)])
        new_l_vals_right.append(l_tempr[np.argmax(d_diff_temp)])
        l_grid_start+=l_step
    l_grid_left=np.array(new_l_vals_left)
    l_grid_right=np.array(new_l_vals_right)
    d_grid_left=np.array(new_d_vals_left)
    d_grid_right=np.array(new_d_vals_right)
    # wow this worked, okay then
    # spline the left gridspace and the right grid space
    #print(l_grid_left)
    #print(d_grid_left)
    #spline inverted since mutliple d values in the d array
    smoothing_param=2
    spl_left = splrep(l_grid_left,d_grid_left,s=smoothing_param)
    l_cont_left = np.linspace(np.min(l_grid_left), np.max(l_grid_left), 200)
    print(l_cont_left)
    d_spline_left = splev(l_cont_left, spl_left)
    #print(d_spline_left)
    spl_right = splrep(l_grid_right,d_grid_right,s=smoothing_param)
    l_cont_right = np.linspace(np.min(l_grid_right), np.max(l_grid_right), 200)
    print(l_cont_right)
    d_spline_right = splev(l_cont_right, spl_right)
    #horizontal line or a curve arrays for closing the interval
    dh=np.linspace(d_spline_left[0],d_spline_right[0],10)
    lh=dh.copy()
    lh.fill(l_cont_left[0]) # right now its a hline really
    i=1
    parab_k=lh[0]-0.2    # if you want a parabola
    midpoint=(dh[-1]-dh[0])/2.0 + dh[0]    # low point of parabola
    while i<len(lh)-1:
        lh[i]=(np.abs(lh[0]-parab_k)/(dh[0]-midpoint)**2)*((dh[i]-midpoint)**2)+parab_k
        i+=1
    # then reverse them and concatenate,
    d_spline_right=np.flip(d_spline_right)
    l_cont_right=np.flip(l_cont_right)
    ltots=np.append(l_cont_left,l_cont_right)
    dtots=np.append(d_spline_left,d_spline_right)
    # plot with lines to connect
    #print(d_border_array)
    chi_nparray=np.empty([20,20])
    chi_nparray = np.loadtxt(filepaths.outputs_path+'be10_be9_ratio_KE_'+str(KE)+'.txt')
    fnt=28
    step=5
    start=2
    fig, ax = plt.subplots(figsize=(10,10)) 
    cax=ax.matshow(np.transpose(chi_nparray),cmap='hsv',origin='lower',vmax=0.35, vmin=0.01)
    #cax=ax.matshow(np.transpose(chi_nparray),cmap='summer',origin='lower',vmax=np.max(chi_nparray)-0.05, vmin=np.min(chi_nparray))
    plt.scatter(x-1,y-1,s=40,marker="x",color="black",label='Minimum',zorder=1)
    #plt.plot(ds-1,ls-1,color="red",marker="p",ms=1,linestyle='None',label="95\%",zorder=1,alpha=0.5)
    #plt.plot(d_border_array,l_border_array,color="white",marker="o",ms=1.5,label="B/C region",zorder=1)
    plt.plot(dtots,ltots,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    # add a horizontal line for the first and last values
    plt.plot(dh,lh,color="black",marker="o",ms=1.5,zorder=1,alpha=0.5)
    # old plot examples
    #plt.plot(d_spline_left,l_cont_left,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_spline_right,l_cont_right,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_grid_left,l_grid_left,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_grid_right,l_grid_right,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_border_array_left[start::step],l_border_array_left[start::step],color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_border_array_right[start::step],l_border_array_right[start::step],color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
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
    plt.title("Be-10/Be-9 at "+str(KE)+" GeV/n",fontsize=fnt,y=1)
    plt.legend(loc='upper right',fontsize=fnt-4)
    plt.savefig(filepaths.images_path+"bery_ratio_colormap_with_chisquareborders_smooth"+str(KE)+".png",dpi=400)


# what if we find the borders and plot those only?
def plot_bery_ratio_KE_with_chisquareborders_smooth_HELIX(seq,KE,events_Be9,frac,save):
    ls,ds,x,y,chi_new,min_chi_square,delta_chi=plot_CI_ratio(1,'Be','C')
    allowed_space=np.where(chi_new<(min_chi_square+delta_chi),1,0)
    print(allowed_space[0,0])
    num_ls=len(chi_new[0]) # to get the L,D model chi use chi_new[L,D]
    # for constant L, find the first and last D (if any) and
    l_border=[]
    d_border=[]
    l_border_left=[]
    d_border_left=[]
    l_border_right=[]
    d_border_right=[]
    i_l=0
    while i_l<num_ls:
        i_d=0
        while i_d<num_ls:
            if allowed_space[i_l,i_d]==1:
                l_border.append(float(i_l+1)/10)
                d_border.append(float(i_d+1)/10)
                l_border_left.append(float(i_l+1)/10)
                d_border_left.append(float(i_d+1)/10)
                break
            i_d+=1
        i_l+=1
    # now go backwards
    i_l=num_ls-1
    while i_l>=0:
        i_d=num_ls-1
        while i_d>=0:
            if allowed_space[i_l,i_d]==1:
                l_border.append(float(i_l+1)/10)
                d_border.append(float(i_d+1)/10)
                l_border_right.append(float(i_l+1)/10)
                d_border_right.append(float(i_d+1)/10)
                break
            i_d-=1
        i_l-=1
    d_border_array=np.array(d_border) # note these values are reported as -1 of the actual D values (so that they line up on a grid starting from 0. 
    l_border_array=np.array(l_border)
    d_border_array_left=np.array(d_border_left)
    l_border_array_left=np.array(l_border_left)
    d_border_array_right=np.array(d_border_right)
    l_border_array_right=np.array(l_border_right)
    # put them in the same order by reversing the right array
    d_border_array_right=np.flip(d_border_array_right)
    l_border_array_right=np.flip(l_border_array_right)
    print(l_border_array_left)
    print(l_border_array_right)

    # for each halo size, in between 2.5 and 3.5, then 3.5 and 4.5, etc find the largest diff in diffusion
    l_start=2.5 # these need to be translated by 1 to give a gridspace value recorded in the arrays
    l_last=20.25
    l_step=0.5 # step from l_start by l_step until l_last
    l_grid_start=l_start-1
    l_grid_last=l_last-1
    new_l_vals_left=[]
    new_l_vals_right=[]
    new_d_vals_left=[]
    new_d_vals_right=[]
    while l_grid_start<l_grid_last:
        print(l_grid_start)
        l_templ=l_border_array_left[np.abs(l_border_array_left-l_grid_start-l_step/2.0)<=l_step/2.0] # get a temp array for left and right halo sizes that is within this region
        l_tempr=l_border_array_right[np.abs(l_border_array_right-l_grid_start-l_step/2.0)<=l_step/2.0] # get a temp array for left and right halo sizes that is within this region
        d_templ=d_border_array_left[np.abs(l_border_array_left-l_grid_start-l_step/2.0)<=l_step/2.0]   # diffusion uses conditions for halo size regions
        d_tempr=d_border_array_right[np.abs(l_border_array_right-l_grid_start-l_step/2.0)<=l_step/2.0] # get the diffusion values for these as well
        d_diff_temp=d_tempr-d_templ # get the difference in D for this region
        diff_amt=np.max(d_diff_temp) # max difference in this region
        new_d_vals_left.append(d_templ[np.argmax(d_diff_temp)])
        new_d_vals_right.append(d_tempr[np.argmax(d_diff_temp)])
        new_l_vals_left.append(l_templ[np.argmax(d_diff_temp)])
        new_l_vals_right.append(l_tempr[np.argmax(d_diff_temp)])
        l_grid_start+=l_step
    l_grid_left=np.array(new_l_vals_left)
    l_grid_right=np.array(new_l_vals_right)
    d_grid_left=np.array(new_d_vals_left)
    d_grid_right=np.array(new_d_vals_right)
    # wow this worked, okay then
    # spline the left gridspace and the right grid space
    #print(l_grid_left)
    #print(d_grid_left)
    #spline inverted since mutliple d values in the d array
    smoothing_param=2
    spl_left = splrep(l_grid_left,d_grid_left,s=smoothing_param)
    l_cont_left = np.linspace(np.min(l_grid_left), np.max(l_grid_left), 200)
    print(l_cont_left)
    d_spline_left = splev(l_cont_left, spl_left)
    #print(d_spline_left)
    spl_right = splrep(l_grid_right,d_grid_right,s=smoothing_param)
    l_cont_right = np.linspace(np.min(l_grid_right), np.max(l_grid_right), 200)
    print(l_cont_right)
    d_spline_right = splev(l_cont_right, spl_right)
    #horizontal line or a curve arrays for closing the interval
    dh=np.linspace(d_spline_left[0],d_spline_right[0],10)
    lh=dh.copy()
    lh.fill(l_cont_left[0]) # right now its a hline really
    i=1
    parab_k=lh[0]-0.2    # if you want a parabola
    midpoint=(dh[-1]-dh[0])/2.0 + dh[0]    # low point of parabola
    while i<len(lh)-1:
        lh[i]=(np.abs(lh[0]-parab_k)/(dh[0]-midpoint)**2)*((dh[i]-midpoint)**2)+parab_k
        i+=1
    # then reverse them and concatenate,
    d_spline_right=np.flip(d_spline_right)
    l_cont_right=np.flip(l_cont_right)
    ltots=np.append(l_cont_left,l_cont_right)
    dtots=np.append(d_spline_left,d_spline_right)
    # plot with lines to connect
#########################BERYLLIUM STUFF##############################
    #print(d_border_array)
    ratio_nparray=np.empty([20,20])
    ratio_nparray = np.loadtxt(filepaths.outputs_path+'be10_be9_ratio_KE_'+str(KE)+'.txt')
    # with this chi_array, spline again ugh?
    ratio_nparray=np.transpose(ratio_nparray)
    L=np.arange(1,21,1)
    D=np.arange(1,21,1)
    #chi_interp = interpolate.interp2d(D, L, chi_real_transposed)
    smoothing=0
    degree_x=3
    degree_y=3
    ratio_interp = interpolate.RectBivariateSpline(D,L,ratio_nparray,kx=degree_x,ky=degree_y,s=smoothing)
    delta_=0.1
    L_new=np.arange(1,20.1,delta_)
    D_new=np.arange(1,20.1,delta_)
    #chi_new=chi_interp(D_new,L_new)
    nominal_res=0.02 # sigma from the proposal plot
    # to get this value do the following
    #calc_res=frac*np.sqrt(frac*events_Be9 + events_Be9)/((1+frac)*events_Be9)
    calc_res=frac*0.05
    ratio_splined=ratio_interp(D_new,L_new)
    allowed_ratios=np.where(np.abs(ratio_splined-frac-calc_res/2.0)<=calc_res/2.0,1,0)
    tuples_found=np.where(np.abs(ratio_splined-frac-calc_res/2.0)<=calc_res/2.0)
    #models_new=model_real_transposed.copy()
    #print(tuples_found[0])
    L_s_allowed=tuples_found[0].copy()
    #print(L_s_allowed)
    # change to actual halo size values
    L_s_allowed=np.true_divide(L_s_allowed,1.0/delta_)+1
    D_s_allowed=tuples_found[1].copy()
    D_s_allowed=np.true_divide(D_s_allowed,1.0/delta_)+1
    #print(np.any(allowed_ratios==1))
    l_border=[]
    d_border=[]
    l_border_left=[]
    d_border_left=[]
    l_border_right=[]
    d_border_right=[]
    i_l=0
    while i_l<num_ls:
        i_d=0
        while i_d<num_ls:
            if allowed_ratios[i_l,i_d]==1:
                l_border.append(float(i_l+1)*delta_)
                d_border.append(float(i_d+1)*delta_)
                l_border_left.append(float(i_l+1)*delta_)
                d_border_left.append(float(i_d+1)*delta_)
                break
            i_d+=1
        i_l+=1
    # now go backwards
    i_l=num_ls-1
    while i_l>=0:
        i_d=num_ls-1
        while i_d>=0:
            if allowed_ratios[i_l,i_d]==1:
                l_border.append(float(i_l+1)*delta_)
                d_border.append(float(i_d+1)*delta_)
                l_border_right.append(float(i_l+1)*delta_)
                d_border_right.append(float(i_d+1)*delta_)
                break
            i_d-=1
        i_l-=1
    d_border_array=np.array(d_border) # note these values are reported as -1 of the actual D values (so that they line up on a grid starting from 0. 
    l_border_array=np.array(l_border)
    d_border_array_left=np.array(d_border_left)
    l_border_array_left=np.array(l_border_left)
    d_border_array_right=np.array(d_border_right)
    l_border_array_right=np.array(l_border_right)
    # put them in the same order by reversing the right array
    d_border_array_right=np.flip(d_border_array_right)
    l_border_array_right=np.flip(l_border_array_right)
    print(l_border_array_left)
    print(l_border_array_right)


    # now find the edges, making sure there is a cutoff in D to only get the interesting piece
    D_cutoff=3
    # do rest at test site
    #num_ls=len(chi_new[0]) # to get the L,D model chi use chi_new[L,D]
    # cut the D_s_allowed and L_s_allowed for HELIX
    
    #print(np.where(chi_new==0))
    #print(np.min(chi_new))
    fnt=28
    step=5
    start=2
    fig, ax = plt.subplots(figsize=(10,10)) 
    # jim likes this one
    #cax=ax.matshow(ratio_nparray,cmap='gist_rainbow',origin='lower',vmax=np.max(ratio_nparray)-0.185, vmin=0.14)
    # i prefer this
    #cax=ax.matshow(ratio_nparray,cmap='hsv',origin='lower',vmax=0.4, vmin=0.12)
    # beacom prefers this
    cax=ax.matshow(ratio_nparray,cmap='RdYlBu',origin='lower',vmax=np.max(ratio_nparray)-0.185, vmin=0.14)
    #cax=ax.matshow(np.transpose(chi_nparray),cmap='summer',origin='lower',vmax=np.max(chi_nparray)-0.05, vmin=np.min(chi_nparray))
    plt.plot(x-1,y-1,ms=8,marker="+",linestyle='None',color="black",label='Minimum',zorder=1)
    #plt.plot(ds-1,ls-1,color="red",marker="p",ms=1,linestyle='None',label="95\%",zorder=1,alpha=0.5)
    #plt.plot(d_border_array[d_border_array>2.5],l_border_array[d_border_array>2.5],color="gray",marker="o",ms=1.5,linestyle='dashed',label="HELIX region",zorder=1)
    #plt.plot(d_border_array,l_border_array,color="gray",marker="o",ms=1.5,linestyle='dashed',label="HELIX region",zorder=1)
    plt.plot(dtots,ltots,color="black",linestyle='dashed', linewidth =2, marker=",",ms=0.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(dtots,ltots,color="blue",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    plt.plot(D_s_allowed-1,L_s_allowed-1,color="black",marker="_",ms=0.75,linestyle='None',label="HELIX region",zorder=1,alpha=0.75)
    # add a horizontal line for the first and last values
    plt.plot(dh,lh,color="black",marker=",",linestyle='dashed', linewidth=2,ms=0.5,zorder=1,alpha=0.5)
    # old plot examples
    #plt.plot(d_spline_left,l_cont_left,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_spline_right,l_cont_right,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_grid_left,l_grid_left,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_grid_right,l_grid_right,color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_border_array_left[start::step],l_border_array_left[start::step],color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
    #plt.plot(d_border_array_right[start::step],l_border_array_right[start::step],color="black",marker="o",ms=1.5,label="B/C region",zorder=1,alpha=0.5)
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
    plt.title("Be-10/Be-9 at "+str(KE)+" GeV/n",fontsize=fnt,y=1)
    lgnd= plt.legend(loc='lower right',fontsize=fnt-2)
    lgnd.legendHandles[0]._legmarker.set_markersize(8)
    lgnd.legendHandles[1]._legmarker.set_markersize(0.5)
    lgnd.legendHandles[2]._legmarker.set_markersize(10)
    if(save==1):
        plt.savefig(filepaths.images_path+"bery_ratio_colormap_with_chisquareborders_smooth"+str(KE)+"_"+str(events_Be9)+"_"+str(frac)+"_.png",dpi=400)
    plt.close()
