from DataProcess_emg import *
from FCNN import *
from Feature import *
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


data=Data(12,6)
data.get_data('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
data.normalise()
data.filter_data(f=(20,50), butterworth_order=4, btype='bandpass')
data.rectify_data()
x,y = data.windowing_data(200, 100)
feature=feature(x)
feature.time_features_estimation(x, 200, 100, data.emg_raw)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(8,6)
loader = DatatoTorch(feature.time_features_matrix.astype(np.float32), y.astype(np.float32), device)
train(model, loader, device, 10000)