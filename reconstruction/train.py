from DataProcess_emg import *
from LSTM import *
import torch

data=Data(12,6)
data.get_data('/remote-home/2230728/project/EMG/NinaPro/DB2', 'S5_E3_A1.mat')
data.normalise()
data.filter_data(f=(20,50), butterworth_order=4, btype='bandpass')
data.rectify_data()
x,y = data.windowing_data(200, 100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(data.emg_channel, 200, data.force_channel)
loader = DatatoTorch(x.astype(np.float32), y.astype(np.float32), device)
train(model, loader, device, 20000)