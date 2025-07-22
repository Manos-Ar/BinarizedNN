import torch.nn as nn
import torch.nn.functional as F
from models.binarized_modules import  BinarizeLinear
# from models.cim_pytorch_modules import BinarizeLinearInference, BinarizeConv2dInference
from models.cim_pytorch_modules import BinarizeLinearInference, BinarizeConv2dInference
import time

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
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

class MLP_CIM(nn.Module):
    def __init__(self,Num_rows,Num_Columns, mode="cs", workers=8, transient=False,checkboard=True,mapping=True):
        super(MLP_CIM, self).__init__()
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.workers     = workers
        self.transient   = transient
        self.checkboard  = checkboard
        self.mapping     = mapping

        self.fc1 = BinarizeLinearInference(784, 512,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,transient=self.transient,workers=self.workers,checkboard=self.checkboard,mapping=self.mapping)
        self.bn1 = nn.BatchNorm1d(512)
        self.htanh1 = nn.Hardtanh()

        self.fc2 = BinarizeLinearInference(512,32,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,transient=self.transient,workers=self.workers,checkboard=self.checkboard,mapping=self.mapping)
        
        self.bn2 = nn.BatchNorm1d(32)
        self.htanh2 = nn.Hardtanh()

        
        self.fc3 =  BinarizeLinearInference(32,10,Num_rows=self.Num_rows,Num_Columns=self.Num_Columns,mode=self.mode,transient=self.transient,workers=self.workers,checkboard=self.checkboard,mapping=self.mapping)

        self.drop=nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        # print(x.shape)
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
    