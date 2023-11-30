import pandas as pd
from scipy import io
import numpy as np

file = '../record2/raw/force/FORCE1700832156440'
force_csv = pd.read_csv( file+'.csv')

# signal = np.array(force["1"].values)
column_names = ['3', '4', '5', '6', '7', '8']
selected_columns = force_csv[column_names]
force = selected_columns.to_numpy()
column_names = ['time']
selected_columns = force_csv[column_names]
time = selected_columns.to_numpy()

io.savemat(file + '.mat',  {"force": force, "force_time": time})