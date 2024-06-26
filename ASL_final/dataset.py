import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset,DataLoader
   
   
df=pd.read_csv('landmarks_.csv')

class CordDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 13180## number of hand landmarks captured!!!
    def __getitem__(self, index) :
       row=df.iloc[index]
       input_data=torch.tensor(row[:63] ,dtype=torch.float)
       input_data=input_data.reshape(1,63)
       input_label=torch.tensor(row[63:],dtype=torch.int64)
       input_label=input_label-1
       return input_data,input_label

train_set=CordDataset()
train_loader=DataLoader(train_set,batch_size=20,shuffle=True)