from .cim_modules import conv_inferenece, linear_inferenece
from .binarized_modules import binarized
import torch.nn as nn
import torch
import os
import numpy as np
class BinarizeConv2dInference(nn.Conv2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 Num_rows, Num_Columns,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True,
                 mode="gs", checkboard=False, workers=8):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        # parameters for custom tiled conv
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.checkboard  = checkboard
        self.workers     = workers

    def forward(self, input):
        # binarize inputs (but keep first 3-channel inputs full-precision)
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b = input

        # binarize weights
        weight_b = binarized(self.weight)

        # use custom inference routine instead of F.conv2d
        padding = self.padding[0] if isinstance(self.padding, tuple) else self.padding
        out = conv_inferenece(
            input_b, weight_b,
            self.Num_rows, self.Num_Columns,
            padding=padding,
            mode=self.mode,
            checkboard=self.checkboard,
            workers=self.workers
        )

        # add bias if present
        if self.bias is not None:
            # store original bias for potential gradient updates, etc.
            self.bias.org = self.bias.data.clone()
            out = out + self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class BinarizeLinearInference(nn.Linear):

    def __init__(self, in_features, out_features,Num_rows,Num_Columns,mode="gs",checkboard=False,workers=8, bias=True):
        # super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        super().__init__(in_features, out_features, bias=bias)
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.checkboard  = checkboard
        self.workers     = workers
    def forward(self, input):
        # print(input.size(1))

        # if input.size(1) != 784:
        input_b=binarized(input)
        weight_b=binarized(self.weight)
        out = linear_inferenece(input_b, weight_b, self.Num_rows, self.Num_Columns, mode=self.mode, checkboard=self.checkboard, workers=self.workers)
        # out = nn.functional.linear(input_b,weight_b)6
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        # print(out)

        return out

if __name__ == "__main__":
    from lenet_5 import BinarizedLeNet5_BN as Net

    # model_path = os.path.join(models_path,f"epoch_7.pth")
    model_idx = 1
    models_path = os.path.abspath(f"/home/earapidis/BinarizedNN/saved_models/lenet_5/model_{model_idx}")
    model_path = os.path.join(models_path,f"best.pth")
    model = Net()
    cuda = False

    model.load_state_dict(torch.load(model_path))
    if cuda:
        torch.cuda.set_device(0)
        model.cuda()
    # print(model)

    N = 1
    Hi = 28
    Wi = 28
    padding = 0 
    CIN = 1

    Num_rows = 32
    Num_Columns = 32                    

############################################

    inputs_conv = torch.randn(N, CIN, Hi, Wi)
    # inputs_conv = binarized(inputs_conv)

    filters_conv = model.conv1.weight.data

    COUT, _, Kh, Kw = filters_conv.shape

    ref_conv = nn.functional.conv2d(binarized(inputs_conv), binarized(filters_conv), padding=0)

    shape = model.fc1.weight.shape

    fc_inputs = torch.randn(N, shape[1])
    ref_fc = nn.functional.linear(binarized(fc_inputs), binarized(model.fc1.weight))


    bias = False
    checkboard = True
    mode = "cs"  

    modes = ["cs", "gs"]
    checkboards = [True, False]

    for mode in modes:
        for checkboard in checkboards:
            print(f" mode: {mode}, checkboard: {checkboard}")
    
            model_conv = BinarizeConv2dInference(CIN, COUT, kernel_size=Kh,Num_rows=Num_rows, Num_Columns=Num_Columns,bias=bias, mode=mode, checkboard=checkboard, workers=8)
            model_conv.weight = nn.Parameter(model.conv1.weight)
            out_conv = model_conv(inputs_conv)
            with np.printoptions(threshold=np.inf):

                diff = out_conv.detach().numpy()-ref_conv.detach().numpy()
                print("Conv output diff:",np.mean(np.abs(diff)))

            ########################################

            model_fc = BinarizeLinearInference(shape[1], shape[0],bias=False, Num_rows=Num_rows, Num_Columns=Num_Columns, mode=mode, checkboard=checkboard, workers=8)
            model_fc.weight = nn.Parameter(model.fc1.weight)
            out_cim_fc = model_fc(fc_inputs)
            with np.printoptions(threshold=np.inf):
                diff_fc = out_cim_fc.detach().numpy()-ref_fc.detach().numpy()

                print("fc output diff:",np.mean(np.abs(diff_fc)))

