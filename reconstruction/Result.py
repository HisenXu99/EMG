import torch
import pandas as pd

def result(model, x, y):
    list=[[]]
    print(x.shape)
    for i, data in enumerate(x):
        with torch.no_grad():
            outputs = model(torch.unsqueeze(x[i], dim=0)).cpu()
            print(outputs.size())
            list.append(outputs[0].numpy().tolist())
    data=pd.DataFrame(list)
    return data