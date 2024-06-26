import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

class CordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_network_stack=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.Linear_network_stack=nn.Sequential(
            nn.Linear(32*14,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,26),
            nn.Softmax(dim=1)
        )
    def forward(self,input):
        self.output=self.Conv_network_stack(input)
        self.output=self.output.reshape(1,32*14)
        self.output=self.Linear_network_stack(self.output)
        return self.output