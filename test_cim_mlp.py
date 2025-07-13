from models.binarized_modules import binarized
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import os
import numpy as np
from models.mlp import MLP , MLP_CIM

from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
from collections import defaultdict
import time
test_batch_size=10
cuda = False
# cuda = True
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)
lr=0.001
criterion = nn.CrossEntropyLoss()

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            break

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)
    return acc

if __name__ == "__main__":

    model_idx = 1
    models_path = os.path.abspath(f"/shares/bulk/earapidis/dev/BinarizedNN/saved_models/mlp/model_{model_idx}")
    model_path = os.path.join(models_path,f"best.pth")
    model = MLP()
    cuda = False

    model.load_state_dict(torch.load(model_path))
    if cuda:
        torch.cuda.set_device(0)
        model.cuda()
    # print(model)


    images, labels = next(iter(test_loader))
    model.eval()
    if cuda:
        images, labels = images.cuda(), labels.cuda()
    images, labels = Variable(images), Variable(labels)

    output = model(images)
    print(output)
    original_prediction = torch.argmax(output, dim=1)
    print(original_prediction)

    modes = ["cs", "gs"]
    checkboards = [True]
    # checkboards = [True, False]
    workers = 20
    predictions = defaultdict(lambda: defaultdict())
    differences = defaultdict(lambda: defaultdict())
    for mode in modes:
        print(f" mode: {mode}")
        model_cim = MLP_CIM(Num_rows=32, Num_Columns=32, mode=mode,workers=workers,transient=False)
        model_cim.set_weights(model)
        start_time = time.time()
        # x = self.pool1(self.htanh1(self.bn1(self.conv1(x))))

        # x = model.pool1(model.htanh1(model.bn1(model.conv1(images))))
        # x = model.pool2(model.htanh2(model.bn2(model.conv2(x))))
        # x = x.view(x.size(0), -1)

        # x = model_cim.htanh3(model_cim.bn_fc1(model_cim.fc1(x)))

        # output_cim = model_cim(images)
        # images = images.view(-1, 28*28)
        # print(images.shape)
        output_cim = model_cim(images)

        end_time = time.time()
        print(f"Time taken for inference: {end_time - start_time} seconds")
        print(output_cim)
        print(output_cim-output)
        prediction = torch.argmax(output_cim, dim=1)
        predictions[mode] = prediction
        diff = output_cim - output
        avg = torch.mean(torch.abs(diff))
        differences[mode] = avg
        print(avg)
        
    print("Predictions:")
    original_prediction = original_prediction.cpu().numpy()
    for mode, prediction in predictions.items():
            print(f"Mode: {mode}")
            prediction = prediction.cpu().numpy()
            print(f"Predictions:   {prediction}")
            print(f"Original:      {original_prediction}")
            num_correct = (prediction == original_prediction ).sum().item()
            acc = num_correct / len(original_prediction) * 100.0
            print(f"Accuracy: {acc:.2f}%")
            print()