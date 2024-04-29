from data_process.pre_process import *
from data_process.feature import *
from utils.save_config import *
from model.LSTM import *
import datetime
import torch


#parameter
win_len = 200
win_stride = 20
batch = 32
epoch = 10000


data=Data(3,6)  #这里就可以改变力和emg的通道数
data.get_data('/remote-home/2230728/project/EMG/myDataset/', 'data.mat')
# data.get_data('/remote-home/2230728/project/EMG/myDataset/', 'merge123.mat')
# data_crop = data.crop_data1('/remote-home/2230728/project/EMG/myDataset/', 'Merge2023-11-18-14-24-13.mat')
data.normalise()
data.filter_data(f=(20,50), butterworth_order=4, btype='bandpass')
data.rectify_data()
# x,y = data.windowing_data2(200, 20, 210000)
x,y = data.windowing_data2(200, 20, 236600)
print(x.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DatatoTorch(x.astype(np.float32), y.astype(np.float32), batch, device)
model = MyModel(next(iter(loader))[0].shape[-1], win_len,next(iter(loader))[1].shape[-1])
#loader = DatatoTorch(feature.time_features_matrix.astype(np.float32), y[:,0,None].astype(np.float32), 32, device)

name = 'model_path/' + 'LSTM' + '-b'+ str(loader.batch_size) + datetime.datetime.now().strftime('-%d:%H:%M') +\
    '-i'+ str(next(iter(loader))[0].shape[-1]) + 'o' + str(next(iter(loader))[1].shape[-1])
print(name)
# save_config(name+'.txt', 'FCNN', data_crop, win_len, win_stride, 0.0001, str(loader.batch_size))
train(model, loader, device, epoch, name+'.pth')