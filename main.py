# encoding=utf-8
# 各种预测代码
import pandas as pd
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time


class Model_pre(nn.Module):
    def __init__(self,hidden_size=4,num_layers=2,acivate_f='sigmoid'):
        super(Model_pre, self).__init__()
        self.input_size=1
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.Ls_dir=nn.Sequential(
            nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        )

        self.lin_dir=nn.Sequential(
            nn.Flatten(),
            nn.Linear(120*self.hidden_size,20),
            nn.Sigmoid()
        )

    def forward(self, x):
        f0=  self.Ls_dir(x.unsqueeze(2))
        f0 = self.lin_dir(f0[0] )
        return f0
