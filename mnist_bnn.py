from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d,binarized
from models.binarized_modules import  Binarize,HingeLoss
# Training settings

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = BinarizeLinear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.htanh1 = nn.Hardtanh()

        self.fc2 = BinarizeLinear(512, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.htanh2 = nn.Hardtanh()

        
        self.fc3 =BinarizeLinear(32,10)

        self.drop=nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)

        # x = self.drop(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        
        # x = self.drop(x)

        x = self.fc3(x)
        return x
        # return self.logsoftmax(x)

