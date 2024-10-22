import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from constants import merged_data
from datasets import *  # noqa
from feature_extractor import net_stat as ns
from models import *  # noqa
from torch.nn import *  # noqa
from torch.utils.data import DataLoader
from torchvision import transforms
from train import RMSELoss  # noqa
from utils import save


def get_threshold(args, model, criterion):
    # Create the DataLoader
    transform = transforms.Compose(
        [
            # Add any desired transformations here
        ]
    )
    dataset = eval(model.dataset)(
        pcap_file=args.traindata_file, max_iterations=sys.maxsize, transform=transform
    )
    if not model.raw:
        batch_size = model.input_dim * args.batch_size
    else:
        batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    reconstruction_errors = []
    maxHost = 100000000000
    maxSess = 100000000000
    nstat = ns.netStat(np.nan, maxHost, maxSess)
    args.print_interval = 100

    for i, packet in enumerate(dataloader):
        if model.raw:
            x = nstat.updateGetStats(
                packet["IPtype"].item(),
                packet["srcMAC"][0],
                packet["dstMAC"][0],
                packet["srcIP"][0],
                packet["srcproto"][0],
                packet["dstIP"][0],
                packet["dstproto"][0],
                int(packet["framelen"]),
                float(packet["timestamp"]),
            )
            # concatenate with the tensors
            reshaped_packets = torch.cat(
                (packet["packet_tensor"][0], torch.tensor(x))
            ).to(torch.float)
        else:
            reshaped_packets = packet.reshape(
                (batch_size // model.input_dim),
                1,
                model.input_dim,
                model.input_dim,
            ).to(torch.float)

        # Move the data to the device that is being used
        model = model.to(args.device)
        reshaped_packets = reshaped_packets.to(args.device)
        # below line is for regular models
        # outputs = model(reshaped_packets)
        # loss = criterion(outputs, reshaped_packets)

        # below line is for loopback pgd
        outputs, tails = model(reshaped_packets)
        loss = torch.log(criterion(outputs, tails))  # average loss over the batch
        reconstruction_errors.append(loss.data)

        if (i + 1) % args.print_interval == 0:
            print(f"Processed {i+1} for calculating threshold")

    # finding the 90th percentile of the reconstruction error distribution for threshold
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
    model.load_state_dict(
        torch.load(f"../artifacts/models/{args.model_name}/model.pth")
    )
    model = model.to(args.device)
    model.eval()
    print("Loaded the model in eval mode!!!")

    transform = transforms.Compose(
        [
            # Add any desired transformations here
        ]
    )

    # criterion = getattr(nn, args.loss)()
    criterion = eval(args.loss)()

    # get threshold
    if not model.raw:
        if args.threshold is not None:
            threshold = -1 * args.threshold
        elif args.get_threshold:
            threshold = -1 * get_threshold(args, model, criterion)
        else:
            print(
                "Neither any threshold provided or the get-threshold flag is set!!! Overriding to calculating threshold"
            )
            threshold = -1 * get_threshold(args, model, criterion)
    if model.raw and args.threshold is None:
        threshold = get_threshold(args, model, criterion)
    if model.raw and args.threshold is not None:
        threshold = args.threshold
    print(f"Threshold for the Anomaly Detector: {threshold}!!!")

    if not model.raw:
        args.batch_size = model.input_dim * args.batch_size
    y_true, y_pred = [], []
    for pcap_path in merged_data:
        # Create the DataLoader
        print(f"Processing {pcap_path}!!!")
        dataset = eval(model.dataset)(
            pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )

        anomaly_scores = []
        maxHost = 100000000000
        maxSess = 100000000000
        nstat = ns.netStat(np.nan, maxHost, maxSess)

        start = time.time()
        for packet in dataloader:
            if model.raw:
                tensors = []
                for j in range(len(packet["IPtype"])):
                    x = nstat.updateGetStats(
                        packet["IPtype"][j].item(),
                        packet["srcMAC"][j],
                        packet["dstMAC"][j],
                        packet["srcIP"][j],
                        packet["srcproto"][j],
                        packet["dstIP"][j],
                        packet["dstproto"][j],
                        int(packet["framelen"][j]),
                        float(packet["timestamp"][j]),
                    )
                    tensors.append(torch.tensor(x))
                # concatenate with the tensors
                reshaped_packets = torch.cat(
                    (packet["packet_tensor"], torch.stack(tensors)), dim=1
                ).to(torch.float)
            else:
                reshaped_packets = packet.reshape(
                    (args.batch_size // model.input_dim),
                    1,
                    model.input_dim,
                    model.input_dim,
                ).to(torch.float)

            # Move the data to the device that is being used
            model = model.to(args.device)
            reshaped_packets = reshaped_packets.to(args.device)

            # below line is for regular models
            # outputs = model(reshaped_packets)
            # loss = criterion(outputs, reshaped_packets)

            # below line is for loopback pgd
            outputs, tails = model(reshaped_packets)
            loss = criterion(outputs, tails)  # average loss over the batch

            if model.raw:
                anomaly_score = loss.data
            else:
                anomaly_score = -1 * loss.data
            anomaly_scores.append(anomaly_score)

            y_true.append(1 if "malicious" in pcap_path else 0)
            y_pred.append(1 if anomaly_score > threshold else 0)

        avg_anomaly_score = sum(anomaly_scores) / len(anomaly_scores)
        print(
            f"Average anomaly score for {pcap_path.split('/')[-1]} is: {avg_anomaly_score}"
        )

        # print time taken to process the pcap file upto 4 decimal places with units
        end = time.time()
        time_taken = end - start
        if time_taken < 60:
            print(f"Time taken: {time_taken:.4f} seconds")
        elif time_taken < 3600:
            print(f"Time taken: {time_taken/60:.4f} minutes")
        else:
            print(f"Time taken: {time_taken/3600:.4f} hours")

        _, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)
        x_val = np.arange(len(anomaly_scores))

        try:
            ax1.scatter(x_val, anomaly_scores, s=1, c="#00008B")
        except Exception as e:
            print(f"Error: {e}")
            ax1.scatter(x_val, anomaly_scores, s=1, alpha=1.0, c="#FF8C00")

        ax1.axhline(y=threshold, color="r", linestyle="-")
        ax1.set_yscale("log")
        ax1.set_title("Anomaly Scores from Kitsune Execution Phase")
        ax1.set_ylabel("RMSE (log scaled)")
        ax1.set_xlabel("Packet index")

        # Show or save the plot
        plt.savefig(f"../artifacts/plots/{pcap_path.split('/')[-1][:-5]}_re.png")
        plt.close()

    # save y_true, y_pred, and anomaly_scores as corresponding objects
    save(
        path="../artifacts/objects/anomaly_detectors/autoencoder/anomaly_scores",
        params={"anomaly_scores": anomaly_scores},
    )
    save(
        path="../artifacts/objects/anomaly_detectors/autoencoder/y_true",
        params={"y_true": y_true},
    )
    save(
        path="../artifacts/objects/anomaly_detectors/autoencoder/y_pred",
        params={"y_pred": y_pred},
    )

    return y_true, y_pred, anomaly_scores
