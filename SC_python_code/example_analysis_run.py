# this code takes about an hour of comp time btw
from all_models_analysis import *
import matplotlib
matplotlib.use('agg') 
#matplotlib.rc('text', usetex=True)
#matplotlib.rc_file("../../templates/matplotlibrc")
import matplotlib.pyplot as plt
#plt.rcParams.update({"text.usetex": True})
#run_chi_square_test(1)
#index,amp,cov=run_analysis_test(0,2000,65,0.6)
run_analysis_test(0,2000,65,0.6) # not used, number spline steps, rigidity (GV) cutoff for analysis, solar modulation factor (GV)
#print(index)
#print(amp)
#print(cov)
