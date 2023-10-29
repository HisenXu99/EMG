import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

class Data():
    def __init__(self, emg_channel, force_channel):
        self.emg_signal = pd.DataFrame()
        self.force_signal = pd.DataFrame()
        self.emg_channel = emg_channel
        self.force_channel = force_channel
        self.data_num = 0

    def get_data(self, path, file):
        mat = loadmat(os.path.join(path,file))
        self.emg_signal = pd.DataFrame(mat['emg'])
        self.force_signal = pd.DataFrame(mat['force'])
        self.data_num = len(self.emg_signal)-1
        # return data,force

    def crop_data(self):  #对数据进行裁剪，针对情况自己写
        pass


    def normalise(self):  #!没写完，标准化变成均值为0
        print("Not yet!!!!!!!!!!!!!!!!!!!!")
        
        scaler = StandardScaler(with_mean=True,
                                    with_std=True,
                                    copy=False).fit(self.emg_signal.iloc[:, :])
        
        scaled = scaler.transform(self.emg_signal.iloc[:,:])
        self.emg_signal = pd.DataFrame(scaled)
    

    def filter_data(self, f, butterworth_order = 4, btype = 'lowpass'):
        #力并没有进行滤波，因为后面窗口内的取均值作为真值
        emg_data = self.emg_signal.values[:,:]
   
        f_sampling = 2000
        nyquist = f_sampling/2
        if isinstance(f, int):
            fc = f/nyquist
        else:
            fc = list(f)
            for i in range(len(f)):
                fc[i] = fc[i]/nyquist
                
        b,a = signal.butter(butterworth_order, fc, btype=btype)
        transpose = emg_data.T.copy()
        
        for i in range(len(transpose)):
            transpose[i] = (signal.lfilter(b, a, transpose[i]))
        
        self.emg_signal = pd.DataFrame(transpose.T)

    def rectify_data(self):
        self.emg_signal = abs(self.emg_signal)

    def windowing_data(self, win_len, win_stride):

        idx=  [i for i in range(win_len, len(self.emg_signal), win_stride)]
        
        x = np.zeros([len(idx), win_len, self.emg_channel])
        force_win=np.zeros([win_len, self.force_channel])
        y = np.zeros([len(idx), self.force_channel])
        
        for i,end in enumerate(idx):
            start = end - win_len
            x[i] = self.emg_signal.iloc[start:end, :].values
            force_win = self.force_signal.iloc[start:end, :].values
            y[i] = np.average(force_win,axis=0)
        return x, y
    
# data=Data(12,6)
# data.get_data('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
# data.emg_signal[:100000].plot(figsize = (15,10))
# data.filter_data(f=20, butterworth_order=4, btype='lowpass')
# data.emg_signal[:100000].plot(figsize = (15,10))
# data.rectify_data()
# data.emg_signal[:100000].plot(figsize = (15,10))
# data.windowing_data(200, 100)