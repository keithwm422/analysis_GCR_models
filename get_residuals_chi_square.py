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


#now make functions to get residuals
# to calculate residuals 
#find the elements of the spline energy array closest to the ams energy value:
# for every data point so len(data_y)
def calculate_chi_square(data_x,data_y,model_x,model_y):
    i=0
    closest_x_value=[]
    closest_x_index=[]
    residual=[]
    expected_values=[]
    while i<len(data_y):
        array=model_x
        value=data_x[i]
        absolute_val_array = np.abs(array - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_x_value.append(array[smallest_difference_index])
        closest_x_index.append(smallest_difference_index)
        residual.append(data_y[i]-model_y[smallest_difference_index])
        expected_values.append(model_y[smallest_difference_index])
        i+=1
    #with residuals and these values (indices) calculate the total chi_square
    #make square residuals
    sq_res=np.square(residual)
    sq_res_div=np.true_divide(sq_res,expected_values)
    return residual,np.sum(sq_res_div)

