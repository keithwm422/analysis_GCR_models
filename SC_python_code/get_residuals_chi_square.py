#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.interpolate import splev, splrep  # included now to evaluate interpolation at exact data point. 

# calculate residuals 
# residuals of y data from model evaluated at x data using interpolation of the logged values, and then unlogged
# actually returns the residuals
# if there is a cutoff in the data, you do not need to pass the cutoff, just the data above the cutoff
# b/c interpolation is evaluated at the data, always give the full model arrays and don't think about it. 
def calculate_residuals(data_x,data_y,model_x, model_y):
    model_x_logged = np.log10(model_x)
    model_y_logged = np.log10(model_y)
    spl = splrep(model_x_logged,model_y_logged) # these should be logged, however, as that helps interpolation
    data_x_logged = np.log10(data_x.copy())
    model_y_spline = splev(data_x_logged, spl)
    model_y_spline=10**(model_y_spline)
    #now find residuals, as array
    res=model_y_spline-data_y
    return np.array(res)

## find the chi-square and reduced for the given residuals (use the errors as the weights by doing np.true_divide)
def calculate_chi_squares(data_x,data_y,model_x,model_y,data_error):
    residual=calculate_residuals(data_x,data_y,model_x,model_y)
    res_weighted=np.true_divide(residual,data_error)
    # now calculate the sums and such
    params=2 # ???
    degrees_of_freedom=len(residual)-params # 2 is the L and D (parameters trying to fit?)
    chi_reduced=np.sum(np.square(res_weighted))/degrees_of_freedom
    chi_square=np.sum(np.square(res_weighted))
    return residual,chi_square,chi_reduced
###OLD###
# to calculate residuals 
# residuals of y data from model evaluated at x values (finds closest values)
# actually returns the residuals, splined-x values closest to data points and splined-y values at those x values
'''def calculate_residuals(data_x,data_y,model_x, model_y):
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
# if cutoff was not zero, apply it to the calculation 
def calculate_reduced_chi_square(data_x,data_y,model_x,model_y,data_error,params,cutoff):
    residual,x_expected,y_expected=calculate_residuals(data_x,data_y,model_x,model_y)
    #print(len(residuals))
    # impose the cutoff *if cutoff is 0 obvi no cutoff.
    #find the values allowed after the masking
    if cutoff > 0:
        #model_x_indices_allowed=x_expected>=cutoff
        #x_expected_masked=x_expected[model_x_indices_allowed]
        #y_expected_masked=y_expected[model_x_indices_allowed]
        data_x_allowed=data_x>=cutoff
        #data_x_masked=data_x[data_x_allowed]
        #data_y_masked=data_y[data_x_allowed]
        data_error_masked=data_error[data_x_allowed]
        residual_masked=residual[data_x_allowed]
        degrees_of_freedom=len(residual_masked)-params # 2 is the L and D (parameters trying to fit?)
        chi_reduced=np.sum(np.true_divide(np.square(residual_masked),np.square(data_error_masked)))/degrees_of_freedom
    else:
        degrees_of_freedom=len(residual)-params # 2 is the L and D (parameters trying to fit?)
        chi_reduced=np.sum(np.true_divide(np.square(residual),np.square(data_error)))/degrees_of_freedom
    p_value=chi2.sf(chi_reduced,degrees_of_freedom) # calculates a p-value for the ch-square reduced based on number of degrees of freedom (data size-parameters to fit)    
    return chi_reduced, p_value
# Once residual is known, use this with y-errors (and pass number of fit parameters in degrees of freedom)
# calls the residuals and calculates the regular chi-square (based on data size) and its pvalue from the scipy package 
def calculate_chi_square(data_x,data_y,model_x,model_y,data_error,cutoff):
    residual,x_expected,y_expected=calculate_residuals(data_x,data_y,model_x,model_y)
    #print(len(residuals))
    # impose the cutoff *if cutoff is 0 obvi no cutoff.
    #find the values allowed after the masking
    if cutoff > 0:
        #model_x_indices_allowed=x_expected>=cutoff
        #x_expected_masked=x_expected[model_x_indices_allowed]
        #y_expected_masked=y_expected[model_x_indices_allowed]
        data_x_allowed=data_x>=cutoff
        #data_x_masked=data_x[data_x_allowed]
        #data_y_masked=data_y[data_x_allowed]
        data_error_masked=data_error[data_x_allowed]
        residual_masked=residual[data_x_allowed]
        degrees_of_freedom=len(residual_masked)-2 # 2 is the L and D (parameters trying to fit?)
        chi_square=np.sum(np.true_divide(np.square(residual_masked),np.square(data_error_masked)))
    else:
        degrees_of_freedom=len(residual)-2 # 2 is the L and D (parameters trying to fit?)
        chi_square=np.sum(np.true_divide(np.square(residual),np.square(data_error)))
    p_value=chi2.sf(chi_square,degrees_of_freedom) # calculates a p-value for the ch-square based on number of degrees of freedom (data size-parameters to fit)    
    return chi_square, p_value
    #residual,x_expected,y_expected=calculate_residuals(data_x,data_y,model_x,model_y)
    #degrees_of_freedom=len(residual)-1 # 1 less than data 
    #print(len(residuals))
    #stats_obj=chisquare(f_obs=data_y, f_exp=y_expected)    
    #return stats_obj
'''
