import pandas as pd
from scipy import io
import numpy as np
from sklearn.preprocessing import StandardScaler

path ="../record2/raw/emg/"
file_name = "EMG1700832558675"
csv_file = path + file_name +'.csv'
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = csv_data.transpose()
csv_df.to_csv(path +file_name + '.csv', index=None)
# column_names = ['1', '2', '3', '4', '5', '6']
# selected_columns = emg_csv[column_names]
# emg = selected_columns.to_numpy()
# column_names = ['0']
# selected_columns = emg_csv[column_names]
# time = selected_columns.to_numpy()

# io.savemat(path + file_name + '.mat',  {"emg": emg, "emg_time": time})