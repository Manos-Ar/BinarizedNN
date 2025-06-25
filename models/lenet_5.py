import torch.nn as nn
import torch.nn.functional as F
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

class BinarizedLeNet5_BN(nn.Module):
    def __init__(self):
        super(BinarizedLeNet5_BN, self).__init__()

        # Conv layers + BatchNorm2d
        self.conv1 = BinarizeConv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.htanh1 = nn.Hardtanh()
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = BinarizeConv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.htanh2 = nn.Hardtanh()
        self.pool2 = nn.AvgPool2d(2)

        # Linear layers + BatchNorm1d
        self.fc1 = BinarizeLinear(16 * 4 * 4, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.htanh3 = nn.Hardtanh()

        self.fc2 = BinarizeLinear(120, 84)
        self.bn_fc2 = nn.BatchNorm1d(84)
        self.htanh4 = nn.Hardtanh()

        self.fc3 = BinarizeLinear(84, 10)  # binarized output
        self.bn_fc3 = nn.BatchNorm1d(10)   # optional: stabilize logits

    def forward(self, x):
        x = self.pool1(self.htanh1(self.bn1(self.conv1(x))))
        x = self.pool2(self.htanh2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)

        x = self.htanh3(self.bn_fc1(self.fc1(x)))
        x = self.htanh4(self.bn_fc2(self.fc2(x)))
        x = self.bn_fc3(self.fc3(x))  # No Hardtanh at the end
        return x

class BinarizedLeNet5_NoBN(nn.Module):
    def __init__(self):
        super(BinarizedLeNet5_NoBN, self).__init__()

        self.conv1 = BinarizeConv2d(1, 6, kernel_size=5)
        self.htanh1 = nn.Hardtanh()
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = BinarizeConv2d(6, 16, kernel_size=5)
        self.htanh2 = nn.Hardtanh()
        self.pool2 = nn.AvgPool2d(2)

        self.fc1 = BinarizeLinear(16 * 4 * 4, 120)
        self.htanh3 = nn.Hardtanh()

        self.fc2 = BinarizeLinear(120, 84)
        self.htanh4 = nn.Hardtanh()

        self.fc3 = BinarizeLinear(84, 10)  # binarized output

    def forward(self, x):
        x = self.pool1(self.htanh1(self.conv1(x)))
        x = self.pool2(self.htanh2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        x = self.htanh3(self.fc1(x))
        x = self.htanh4(self.fc2(x))
        x = self.fc3(x)  # No Hardtanh
        return x