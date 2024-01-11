import sys
import torch
import torch.nn as nn

from models import *
from datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import save, load
from constants import merged_data

def get_threshold(args, model, criterion):
    # Create the DataLoader
    transform = transforms.Compose([
        # Add any desired transformations here
    ])
    dataset = eval(model.dataset)(pcap_file=args.traindata_file, max_iterations=sys.maxsize, transform=transform)
    dataloader = DataLoader(dataset, batch_size=model.input_dim * args.batch_size, shuffle=False, drop_last=True)

    reconstruction_errors = []

    for packets in dataloader:
        reshaped_packets = packets.reshape(args.batch_size, 1, model.input_dim, model.input_dim).to(torch.float)
        outputs = model(reshaped_packets)

        # Compute the loss
        loss = criterion(outputs, reshaped_packets)
        reconstruction_errors.append(loss.data)

    # finding the 95th percentile of the reconstruction error distribution for threshold
    reconstruction_errors.sort(reverse=True)
    ninety_fifth_percentile_index = int(0.90 * len(reconstruction_errors))
    threshold = reconstruction_errors[ninety_fifth_percentile_index]

    return threshold

def infer(args):
    """
    Infer using the model using args provided by the user
    """
    # Load the model
    model = eval(args.model_name)()
    model.load_state_dict(torch.load(f"../artifacts/models/{args.model_name}/model.pth"))
    model = model.to(args.device)
    model.eval()
    print(f"Loaded the model in eval mode!!!")

    transform = transforms.Compose([
        # Add any desired transformations here
    ])

    criterion = getattr(nn, args.loss)()

    # get threshold
    if args.threshold is not None:
        threshold = -1 * args.threshold
    elif args.get_threshold:
        threshold = -1 * get_threshold(args, model, criterion)
    else:
        print("Neither any threshold provided or the get-threshold flag is set!!! Overriding to calculating threshold")
        threshold = -1 * get_threshold(args, model, criterion)
    print(f"Threshold for the Anomaly Detector: {threshold}!!!")

    y_true, y_pred = [], []
    for pcap_path in merged_data:
        # Create the DataLoader
        dataset = eval(model.dataset)(pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform)
        dataloader = DataLoader(dataset, batch_size=model.input_dim * args.batch_size, shuffle=False, drop_last=True)

        anomaly_scores = []

        for packets in dataloader:
            reshaped_packets = packets.reshape(args.batch_size, 1, model.input_dim, model.input_dim).to(torch.float)
            outputs = model(reshaped_packets)

            # Compute the loss
            loss = criterion(outputs, reshaped_packets)
            anomaly_score = -1 * loss.data
            anomaly_scores.append(anomaly_score)

            y_true.append(1 if "malicious" in pcap_path else 0)
            y_pred.append(1 if anomaly_score > threshold else 0)

        avg_anomaly_score = sum(anomaly_scores)/ len(anomaly_scores)
        print(f"Average anomaly score for {pcap_path.split('/')[-1]} is: {avg_anomaly_score}")

    # save y_true, y_pred, and anomaly_scores as corresponding objects
    save(path="../artifacts/objects/anomaly_detectors/autoencoder/anomaly_scores", params={"anomaly_scores": anomaly_scores})
    save(path="../artifacts/objects/anomaly_detectors/autoencoder/y_true", params={"y_true": y_true})
    save(path="../artifacts/objects/anomaly_detectors/autoencoder/y_pred", params={"y_pred": y_pred})

    return y_true, y_pred, anomaly_scores
