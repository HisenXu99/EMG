import pandas as pd
from scipy import io
import numpy as np

file = '../record2/raw/emg/EMG1700832558675'
emg_csv = pd.read_csv( file+'.csv')

# signal = np.array(force["1"].values)
column_names = ['1', '2', '3', '4', '5', '6']
selected_columns = emg_csv[column_names]
emg = selected_columns.to_numpy()
column_names = ['0']
selected_columns = emg_csv[column_names]
time = selected_columns.to_numpy()

io.savemat(file + '.mat',  {"emg": emg, "emg_time": time})