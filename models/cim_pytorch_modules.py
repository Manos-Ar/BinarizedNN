import os 
import torch.nn as nn
from models.cim_conv import con2d
from models.cim_fc import fc
from models.binarized_modules import binarized


class BinarizeConv2dInference(nn.Conv2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 Num_rows, Num_Columns,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True,
                 mode="ideal", workers=8,transient=False,checkboard=True,mapping=True):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        # parameters for custom tiled conv
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.workers     = workers
        self.transient   = transient
        self.checkboard  = checkboard
        self.mapping     = mapping

    def forward(self, input):
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b = input

        weight_b = binarized(self.weight)

        padding = self.padding[0] if isinstance(self.padding, tuple) else self.padding

        input_b = input_b.detach()
        weight_b = weight_b.detach()
        out = con2d(input_b,weight_b,self.Num_rows,self.Num_Columns,self.mode,self.workers,self.transient,self.checkboard,self.mapping)
        # add bias if present
        if self.bias is not None:
            # store original bias for potential gradient updates, etc.
            self.bias.org = self.bias.data.clone()
            out = out + self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class BinarizeLinearInference(nn.Linear):

    def __init__(self, in_features, out_features,Num_rows,Num_Columns,mode="ideal",workers=8, bias=True,transient=False,checkboard=False,mapping=False):
        super().__init__(in_features, out_features, bias=bias)
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.workers     = workers
        self.transient   = transient
        self.checkboard  = checkboard
        self.mapping     = mapping
    def forward(self, input):
        # print(input.size(1))

        # if input.size(1) != 784:
        input_b=binarized(input)
        weight_b=binarized(self.weight)

        input_b = input_b.detach()
        weight_b = weight_b.detach()
        weight_b = weight_b.T
        out = fc(input_b,weight_b,self.Num_rows,self.Num_Columns,mode=self.mode,max_workers=self.workers,transient=self.transient,checkboard=self.checkboard,mapping=self.mapping)
        # out = nn.functional.linear(input_b,weight_b)6
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        # print(out)

        return out
    

    
