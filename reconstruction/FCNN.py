import math
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from statistics import mean
from torch.utils.data import Dataset, DataLoader

class MyModel(nn.Module):
    def __init__(self, feature_num, force_num):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(feature_num, 128)
        self.dropout_1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, force_num)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear2(x)
        # x = self.activation(x)
        x = self.dropout_2(x)
        x = self.linear3(x)
        #x = torch.softmax(x, dim=1)
        return x
    
class Dst(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    
def DatatoTorch(x, y, device):
    print(x)
    print(y)
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    train_dst = Dst(x, y)
    loader = DataLoader(train_dst, batch_size=32, shuffle=True)
    return loader


def train(model, loader, device, epoch):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    model.to(device)
    loss_list=[]

    name = 'model_' + datetime.datetime.now().strftime('%d-%H:%M') + '.pth'

    for epoch in range(epoch):
        for i, data in enumerate(loader):
            model.train()
            x, y = data
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

            model.eval()  #dropout

            # print('Epoch: %d, Loss: %.3f' % (epoch, loss.item()))

        torch.save(model.state_dict(), name)
        print('Epoch: %d, Loss: %.3f' % (epoch, mean(loss_list)))
        if(math.isnan(mean(loss_list))):
            print("Something wrong!")
            break
        loss_list=[]
    pass