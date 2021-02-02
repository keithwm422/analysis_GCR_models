#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
# to calculate residuals 
# residuals of y data from model evaluated at x values (finds closest values)
# actually returns the residuals, splined-x values closest to data points and splined-y values at those x values
def calculate_residuals(data_x,data_y,model_x, model_y):
    i=0
    closest_x_value=[]
    #closest_x_index=[]
    residual=[]
    expected_values=[]
    while i<len(data_y):
        # no matter what, find the closest rigidity value in the spline.
        array=model_x.copy()
        value=data_x[i]
        absolute_val_array = np.abs(array - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_x_value.append(array[smallest_difference_index]) # the actual model x-axis value near the data x-axis point
        #closest_x_index.append(smallest_difference_index)
        residual.append(data_y[i]-model_y[smallest_difference_index])
        expected_values.append(model_y[smallest_difference_index]) # the actual model y-axis value near the data x-axis point
        i+=1
    return np.array(residual),np.array(closest_x_value),np.array(expected_values)
# Once residual is known, use this with y-errors (and pass number of fit parameters in degrees of freedom)
# calls the residuals and calculates the reduced chi-square (based on integer params) and its pvalue from the SF of the distribution. 
def calculate_reduced_chi_square(data_x,data_y,model_x,model_y,data_error,params):
    residual,x_expected,y_expected=calculate_residuals(data_x,data_y,model_x,model_y)
    degrees_of_freedom=len(residual)-params # 2 is the L and D (parameters trying to fit?)
    #print(len(residuals))
    chi_reduced=np.sum(np.true_divide(np.square(residual),np.square(data_error)))/degrees_of_freedom
    p_value=chi2.sf(chi_reduced,degrees_of_freedom) # calculates a p-value for the ch-square reduced based on number of degrees of freedom (data size-parameters to fit)    
    return chi_reduced, p_value
# Once residual is known, use this with y-errors (and pass number of fit parameters in degrees of freedom)
# calls the residuals and calculates the regular chi-square (based on data size) and its pvalue from the scipy package 
def calculate_chi_square(data_x,data_y,model_x,model_y,cutoff):
    residual,x_expected,y_expected=calculate_residuals(data_x,data_y,model_x,model_y)
    degrees_of_freedom=len(residual)-1 # 1 less than data 
    #print(len(residuals))
    stats_obj=chisquare(f_obs=data_y, f_exp=y_expected)    
    return stats_obj

