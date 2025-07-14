import os
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from collections import defaultdict
import time
import random
import argparse
from tqdm import tqdm

from models.lenet_5 import BinarizedLeNet5_BN as Lenet5
from models.lenet_5 import BinarizedLeNet5_BN_CIM as Lenet5_CIM

def test(model, data_loader):
    model.eval()
    results = []

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            # if cuda:
            #     data, target = data.cuda(), target.cuda()

            output = model(data).detach()
            results.append(output.cpu())

    results = torch.stack(results)
    return results


if __name__ == "__main__":

    parsher = argparse.ArgumentParser()
    parsher.add_argument("save_path",type=str)
    parsher.add_argument('-l',"--load", action='store_true',help="load sim from target save_path")

    args = parsher.parse_args()

    save_path = os.path.abspath(args.save_path)
    load_indices = args.load

    model_idx = 2
    models_path = os.path.abspath(f"/shares/bulk/earapidis/dev/BinarizedNN/saved_models/lenet_5/model_{model_idx}")
    # model_path = os.path.join(models_path,f"epoch_1.pth")
    model_path = os.path.join(models_path,f"best.pth")
    model = Lenet5()

    test_batch_size = 10
    num_batches = 1
    num_samples = num_batches * test_batch_size

    test_dataset = datasets.MNIST(
        'data', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    os.makedirs(save_path,exist_ok=True)
    indices_path = os.path.join(save_path,"random_indices.pt")
    
    if load_indices and os.path.exists(indices_path):

         random_indices = torch.load(indices_path)
         print("loaded random indices")
    else:
        random_indices = random.sample(range(len(test_dataset)), num_samples)
        torch.save(torch.tensor(random_indices), indices_path)

    subset = Subset(test_dataset, random_indices)
    subset_loader = DataLoader(subset, batch_size=test_batch_size, shuffle=False)

    model.load_state_dict(torch.load(model_path))
    original_output = test(model,subset_loader)
    original_prediction = torch.argmax(original_output, dim=2)
    
    original_prediction_path = os.path.join(save_path,"original_prediction.pt")
    original_last_layer_path = os.path.join(save_path,"original_last_layer.pt")
    
    torch.save(original_output,original_last_layer_path)
    torch.save(original_prediction,original_prediction_path)
    # print(original_prediction)


    # modes = ["ideal"]
    modes = ["cs", "gs"]

    workers = 16
    predictions = defaultdict(lambda: defaultdict())


    for mode in modes:
        print(f" mode: {mode}")
        model_cim = Lenet5_CIM(Num_rows=32, Num_Columns=32, mode=mode,workers=workers,transient=False)
        # model_cim.set_weights(model)
        model_cim.load_state_dict(model.state_dict())
        model_cim.eval()
        start_time = time.time()
        output_cim = test(model_cim,subset_loader)
        end_time = time.time()

        mode_dir = os.path.join(save_path,mode)
        os.makedirs(mode_dir,exist_ok=True)
        last_layer_path = os.path.join(mode_dir,"last_layer.pt")
        torch.save(output_cim, last_layer_path)
        prediction = torch.argmax(output_cim, dim=2)
        prediction_path = os.path.join(mode_dir,"prediction.pt")
        torch.save(prediction,prediction_path)

        predictions[mode] = prediction
        # diff = output_cim - output
        # avg = torch.mean(torch.abs(diff))
        # print(avg)
        print(f"Time taken for inference: {end_time - start_time} seconds")
        
    print("Predictions:")
    for mode, prediction in predictions.items():
            print(f"\tMode: {mode}")
            # print(f"Predictions:   {prediction}")
            # print(f"Original:      {original_prediction}")
            num_correct = (prediction == original_prediction ).sum().item()
            # print(num_correct)
            total = torch.numel(original_prediction)
            acc = (num_correct /total)  * 100.0
            print(f"\tAccuracy: {acc:.2f}%")
            print()