import torch.nn as nn
import torch.nn.functional as F
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
from .cim_pytorch_modules import BinarizeLinearInference, BinarizeConv2dInference

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

padding = 0
class BinarizedLeNet5_NoBN(nn.Module):
    def __init__(self):
        super(BinarizedLeNet5_NoBN, self).__init__()

        self.conv1 = BinarizeConv2d(1, 6, kernel_size=5,padding=padding)
        self.htanh1 = nn.Hardtanh()
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = BinarizeConv2d(6, 16, kernel_size=5,padding=padding)
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

class BinarizedLeNet5_BN_CIM(nn.Module):
    def __init__(self,Num_rows,Num_Columns, mode="cs", checkboard=False, workers=8):
        super(BinarizedLeNet5_BN_CIM, self).__init__()
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.checkboard  = checkboard
        self.workers     = workers


        # Conv layers + BatchNorm2d
        # self.conv1 = BinarizeConv2d(1, 6, kernel_size=5)
        self.conv1 = BinarizeConv2dInference(1,6, kernel_size=5,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,checkboard=self.checkboard,workers=self.workers)
        self.bn1 = nn.BatchNorm2d(6)
        self.htanh1 = nn.Hardtanh()
        self.pool1 = nn.AvgPool2d(2)

        # self.conv2 = BinarizeConv2d(6, 16, kernel_size=5)
        self.conv2 = BinarizeConv2dInference(6,16, kernel_size=5,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,checkboard=self.checkboard,workers=self.workers)
        self.bn2 = nn.BatchNorm2d(16)
        self.htanh2 = nn.Hardtanh()
        self.pool2 = nn.AvgPool2d(2)

        # Linear layers + BatchNorm1d
        # self.fc1 = BinarizeLinear(16 * 4 * 4, 120)
        self.fc1 = BinarizeLinearInference(16 * 4 * 4, 120,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,checkboard=self.checkboard,workers=self.workers)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.htanh3 = nn.Hardtanh()

        # self.fc2 = BinarizeLinear(120, 84)
        self.fc2 = BinarizeLinearInference(120, 84,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,checkboard=self.checkboard,workers=self.workers)

        self.bn_fc2 = nn.BatchNorm1d(84)
        self.htanh4 = nn.Hardtanh()

        self.fc3 = BinarizeLinearInference(84, 10,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,checkboard=self.checkboard,workers=self.workers)
        # self.fc3 = BinarizeLinear(84, 10)  # binarized output

        self.bn_fc3 = nn.BatchNorm1d(10)   # optional: stabilize logits

    def forward(self, x):
        x = self.pool1(self.htanh1(self.bn1(self.conv1(x))))
        x = self.pool2(self.htanh2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)

        x = self.htanh3(self.bn_fc1(self.fc1(x)))
        x = self.htanh4(self.bn_fc2(self.fc2(x)))
        x = self.bn_fc3(self.fc3(x))  # No Hardtanh at the end
        return x
    
    def set_weights(self, model:BinarizedLeNet5_BN):
        self.conv1.weight = model.conv1.weight
        self.conv2.weight = model.conv2.weight
        self.fc1.weight = model.fc1.weight
        self.fc2.weight = model.fc2.weight
        self.fc3.weight = model.fc3.weight

        # Set biases if they exist
        if hasattr(model.conv1, 'bias'):
            self.conv1.bias = model.conv1.bias
        if hasattr(model.conv2, 'bias'):
            self.conv2.bias = model.conv2.bias
        if hasattr(model.fc1, 'bias'):
            self.fc1.bias = model.fc1.bias
        if hasattr(model.fc2, 'bias'):
            self.fc2.bias = model.fc2.bias
        if hasattr(model.fc3, 'bias'):
            self.fc3.bias = model.fc3.bias

        # Set batch normalization parameters
        self.bn1.weight = model.bn1.weight
        self.bn1.bias = model.bn1.bias
        self.bn1.running_mean = model.bn1.running_mean
        self.bn1.running_var = model.bn1.running_var

        self.bn2.weight = model.bn2.weight
        self.bn2.bias = model.bn2.bias
        self.bn2.running_mean = model.bn2.running_mean
        self.bn2.running_var = model.bn2.running_var

        self.bn_fc1.weight = model.bn_fc1.weight
        self.bn_fc1.bias = model.bn_fc1.bias
        self.bn_fc1.running_mean = model.bn_fc1.running_mean
        self.bn_fc1.running_var = model.bn_fc1.running_var

        self.bn_fc2.weight = model.bn_fc2.weight
        self.bn_fc2.bias = model.bn_fc2.bias
        self.bn_fc2.running_mean = model.bn_fc2.running_mean
        self.bn_fc2.running_var = model.bn_fc2.running_var

        self.bn_fc3.weight = model.bn_fc3.weight
        self.bn_fc3.bias = model.bn_fc3.bias
        self.bn_fc3.running_mean = model.bn_fc3.running_mean
        self.bn_fc3.running_var = model.bn_fc3.running_var

        # Ensure the model is in evaluation
        # mode to use the running statistics of batch normalization
        self.eval()
