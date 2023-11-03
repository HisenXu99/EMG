from data_process.pre_process import *
from data_process.feature import *
from model.LSTM import *

import torch

# data=Data(12,6)
# data.get_data('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
# data.normalise()
# data.filter_data(f=(20,50), butterworth_order=4, btype='bandpass')
# data.rectify_data()
# x,y = data.windowing_data(200, 100)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyModel(data.emg_channel, 200, data.force_channel)
# loader = DatatoTorch(x.astype(np.float32), y.astype(np.float32), device)
# train(model, loader, device, 20000)

#parameter
win_len = 200
win_stride = 20
batch = 32
epoch = 10000


data=Data(12,6)
data.get_data('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
data.crop_data('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
data.normalise()
data.filter_data(f=(20,50), butterworth_order=4, btype='bandpass')
data.rectify_data()

x,y = data.windowing_data(win_len, win_stride)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(12, win_len, 6)
#loader = DatatoTorch(feature.time_features_matrix.astype(np.float32), y[:,0,None].astype(np.float32), 32, device)
loader = DatatoTorch(x.astype(np.float32), y.astype(np.float32), batch, device)
train(model, loader, device, epoch, 'model_path/')


#参数快照编写，应该放在训练之前




# # 准备数据
# x=np.linspace(-2*np.pi,2*np.pi,400)
# y=np.sin(x)
# # 将数据做成数据集的模样
# X=np.expand_dims(x,axis=1)
# Y=y.reshape(400,-1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 使用批训练方式
# model = MyModel(1,1)
# loader = DatatoTorch(X.astype(np.float32), Y.astype(np.float32), device)
# train(model, loader, device, 1000)