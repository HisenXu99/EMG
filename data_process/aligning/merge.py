import pandas as pd
from scipy import io
import numpy as np


#EMG
file = '../record2/raw/merge/merge_emg123'
emg_csv = pd.read_csv( file+'.csv')

column_names = ['1', '2', '3', '4', '5', '6']
selected_columns = emg_csv[column_names]
emg = selected_columns.to_numpy()
# column_names = ['0']
# selected_columns = emg_csv[column_names]
# time = selected_columns.to_numpy()



#force
file = '../record2/raw/merge/merge_force123'
force_csv = pd.read_csv( file+'.csv')

column_names = ['3', '4', '5', '6', '7', '8']
selected_columns = force_csv[column_names]
force = selected_columns.to_numpy()
# column_names = ['time']
# selected_columns = force_csv[column_names]
# time = selected_columns.to_numpy()

io.savemat('../record2/merge/merge123.mat',  {"emg": emg, "force": force})