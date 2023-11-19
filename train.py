from data_process.pre_process import *
from data_process.feature import *
from utils.save_config import *
from model.FCNN import *
import datetime
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
data_crop = data.crop_data1('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
data.normalise()
data.filter_data(f=(20,50), butterworth_order=4, btype='bandpass')
data.rectify_data()
x,y = data.windowing_data(200, 20)
feature=feature(x)
feature.time_features_estimation(x, 200, data.emg_raw)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(8,6)
#loader = DatatoTorch(feature.time_features_matrix.astype(np.float32), y[:,0,None].astype(np.float32), 32, device)
loader = DatatoTorch(feature.time_features_matrix.astype(np.float32), y.astype(np.float32), 32, device)
name = 'model_path/' + 'FCNN' + '-b'+ str(loader.batch_size) + datetime.datetime.now().strftime('-%d:%H:%M') +\
        '-i'+ str(next(iter(loader))[0].shape[-1]) + 'o' + str(next(iter(loader))[1].shape[-1])
print(name)
save_config(name+'.txt', 'FCNN', data_crop, win_len, win_stride, 0.0001, str(loader.batch_size))
train(model, loader, device, 100000, name+'.pth')


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