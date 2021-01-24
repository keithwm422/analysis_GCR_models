#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import chisquare
#now make functions to get residuals
# to calculate residuals 
#find the elements of the spline energy array closest to the ams energy value:
# for every data point so len(data_y)
# add ability to impost cutoff (calculate above certain rigidity or kinetic energy per nucleon)
# Return alot of stuff in case its needed for plotting
def calculate_chi_square(data_x,data_y,model_x,model_y,cutoff):
    i=0
    closest_x_value=[]
    closest_x_index=[]
    residual=[]
    expected_values=[]
    residual_masked=[]
    while i<len(data_y):
        # no matter what, find the closest rigidity value in the spline.
        array=model_x.copy()
        value=data_x[i]
        absolute_val_array = np.abs(array - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_x_value.append(array[smallest_difference_index]) # the actual model x-axis value near the data x-axis point
        closest_x_index.append(smallest_difference_index)
        residual.append(data_y[i]-model_y[smallest_difference_index])
        expected_values.append(model_y[smallest_difference_index]) # the actual model y-axis value near the data x-axis point
        i+=1
    #with residuals and these values (indices) calculate the total chi_square
    #make square residuals
    sq_res=np.square(residual)
    sq_res_div=np.true_divide(sq_res,expected_values)
    stats_obj=chisquare(f_obs=data_y, f_exp=expected_values)
    #find the rigidity bin that starts the cutoff
    #masking arrays correctly
    # make them into numpy arrays
    closest_x_value=np.array(closest_x_value)
    expected_values=np.array(expected_values)
    #find the values allowed after the masking
    model_x_values_allowed=closest_x_value>=cutoff
    closest_x_masked=closest_x_value[model_x_values_allowed]
    expected_values_masked=expected_values[model_x_values_allowed]
    data_x_allowed=data_x>=cutoff
    data_x_masked=data_x[data_x_allowed]
    data_y_masked=data_y[data_x_allowed]
    # now calculate new statistics
    stats_obj_masked=chisquare(f_obs=data_y_masked, f_exp=expected_values_masked)

    return residual,np.sum(sq_res_div),stats_obj,stats_obj_masked,data_x_masked,data_y_masked,closest_x_value,expected_values,closest_x_masked,expected_values_masked


