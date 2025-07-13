import os 
import torch.nn as nn
from models.cim_conv2d_mapping import con2d
from models.cim_fc_mapping import fc
from models.binarized_modules import binarized


class BinarizeConv2dInference(nn.Conv2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 Num_rows, Num_Columns,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True,
                 mode="cs", workers=8,transient=False):
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

    def forward(self, input):
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b = input

        weight_b = binarized(self.weight)

        padding = self.padding[0] if isinstance(self.padding, tuple) else self.padding

        input_b = input_b.detach()
        weight_b = weight_b.detach()
        out = con2d(input_b,weight_b,self.Num_rows,self.Num_Columns,self.mode,self.workers,self.transient)
        # add bias if present
        if self.bias is not None:
            # store original bias for potential gradient updates, etc.
            self.bias.org = self.bias.data.clone()
            out = out + self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class BinarizeLinearInference(nn.Linear):

    def __init__(self, in_features, out_features,Num_rows,Num_Columns,mode="cs",workers=8, bias=True,transient=False):
        super().__init__(in_features, out_features, bias=bias)
        self.Num_rows    = Num_rows
        self.Num_Columns = Num_Columns
        self.mode        = mode
        self.workers     = workers
        self.transient   = transient
    def forward(self, input):
        # print(input.size(1))

        # if input.size(1) != 784:
        input_b=binarized(input)
        weight_b=binarized(self.weight)

        input_b = input_b.detach()
        weight_b = weight_b.detach()
        weight_b = weight_b.T
        out = fc(input_b,weight_b,self.Num_rows,self.Num_Columns,self.mode,self.workers,self.transient)
        # out = nn.functional.linear(input_b,weight_b)6
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        # print(out)

        return out
    
if __name__ == "__main__":
    from lenet_5 import BinarizedLeNet5_BN as Net
    from torchvision import datasets, transforms
    import torch


    cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    test_batch_size=1
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)


    model_idx = 3
    models_path = os.path.abspath(f"/shares/bulk/earapidis/dev/BinarizedNN/saved_models/lenet_5/model_{model_idx}")
    # models_path = os.path.abspath(f"/home/earapidis/BinarizedNN/saved_models/lenet_5/model_{model_idx}")
    model_path = os.path.join(models_path,f"best.pth")
    model = Net()
    model.load_state_dict(torch.load(model_path))
    if cuda:
        torch.cuda.set_device(0)
        model.cuda()
    
    images, labels = next(iter(test_loader))

    
