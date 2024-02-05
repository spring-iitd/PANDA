import argparse

import matplotlib.pyplot as plt
import numpy as np
import scapy.all as scapy
import torch.nn as nn
from attacks import Attack

from utils import save

def update_timestamps(pcap_file, inter_arrival_times, adv_pcap_path):
    """
    Updates the timestamps of the packets in the pcap file
    The difference in length of pcap file and inter_arrival_times
    is due to the fact that the drop_last parameter in the dataloader
    is set to True. This is done to ensure that the last batch is
    dropped because last batch doesn't form an image.
    """
    packets = scapy.rdpcap(pcap_file)
    for i, packet in enumerate(packets):
        if i == 0:
            new_timestamp = packet.time
            continue
        elif i < len(inter_arrival_times):
            new_timestamp = new_timestamp + inter_arrival_times[i]
        else:
            break

        packet.time = new_timestamp

    scapy.wrpcap(adv_pcap_path, packets)

def update_timestamps_raw(pcap_file, adv_timestamps, adv_pcap_path):
    """
    Updates the timestamps of the packets in the pcap file
    The difference in length of pcap file and inter_arrival_times
    is due to the fact that the drop_last parameter in the dataloader
    is set to True. This is done to ensure that the last batch is
    dropped because last batch doesn't form an image.
    """
    packets = scapy.rdpcap(pcap_file)
    print("Original timestamp of 5th packet:", packets[5].time)
    print("Expected new timestamp of 5th packet:", adv_timestamps[5])
    same = 0
    for i, packet in enumerate(packets):
        if packet.time == adv_timestamps[i]:
            same += 1
        packet.time = adv_timestamps[i]
    
    print("Number of same timestamps:", same, "out of", i)
    print("Updated timestamp of 5th packet:", packets[5].time)
    scapy.wrpcap(adv_pcap_path, packets)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "PANDA: Adversarial Attack",
        add_help=False,
    )
    parser.add_argument(
        "--root-dir",
        default="../",
        help="folder where all the code, data, and artifacts lie",
    )
    parser.add_argument(
        "--pcap-path", default="../data/malicious/Port_Scanning_SmartTV.pcap", type=str
    )
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument(
        "--device", default="cuda", help="device to use for training/ testing"
    )
    parser.add_argument(
        "--surrogate-model",
        default="Autoencoder",
        type=str,
        help="Name of the surrogate model",
    )
    parser.add_argument(
        "--target-model", default="kitsune", type=str, help="Name of the target model"
    )
    parser.add_argument(
        "--attack",
        default="fgsm",
        type=str,
        help="Name of the attack to perform or inference",
    )
    parser.add_argument("--selected-columns", nargs="+", default=list(range(32)), type=list)
    parser.add_argument(
        "--eval", action="store_true", default=False, help="perform attack inference"
    )

    return parser


criterion = nn.BCELoss()


def main(args):
    # create an object of the attack class
    args.adv_pcap_path = (
        f"../data/adversarial/{args.attack}/Adv_{args.pcap_path.split('/')[-1]}"
    )
    attack = Attack(args=args)
    attack_method = getattr(attack, args.attack)
    re, adv_re, y_true, y_pred, taus, adv_timestamps, adv_sizes, actual_sizes = attack_method(epsilon=0.05)

    print(f"Pcap file: {args.pcap_path.split('/')[-1][:-5]}")
    print(f"Mean RE for malicious packets: {sum(re)/ len(re)}")
    print(f"Mean RE for adversarial malicious packets: {sum(adv_re)/ len(adv_re)}")

    save(
        path="../artifacts/objects/attacks/loopback_pgd/adv_sizes",
        params={"adv_sizes": adv_sizes},
    )

    evasion_rate = 1 - (sum(y_pred) / len(y_pred))
    print(f"Evasion Rate: {evasion_rate}")

    # create adversarial packets
    if args.surrogate_model == "AutoencoderRaw":
        update_timestamps_raw(args.pcap_path, adv_timestamps, args.adv_pcap_path)
    else:
        update_timestamps(args.pcap_path, taus, args.adv_pcap_path)

    # Plot the reconstruction error curve
    re = [np.array(elem.to("cpu")) for elem in re][20:100]
    adv_re = [np.array(elem.to("cpu")) for elem in adv_re][20:100]

    # Generate x-axis values (image indices)
    image_indices = np.arange(len(re))


    # Create a line curve (line plot)
    plt.figure(figsize=(10, 6))
    plt.plot(
        image_indices, re, marker="o", linestyle="-", color="b", label="Clean Data"
    )
    image_indices = np.arange(len(adv_re))
    plt.plot(
        image_indices,
        adv_re,
        marker="o",
        linestyle="-",
        color="r",
        label="Advesarial Data",
    )
    # TODO: #absolute value for threshold. REMOVE!!!!
    plt.axhline(y=765.83, color="green", linestyle="--", label="Threshold")
    plt.title(
        f"Reconstruction Error Curve: {args.pcap_path.split('/')[-1][:-5]}_{args.attack}"
    )
    plt.xlabel("Image Index")
    plt.ylabel("Reconstruction Error")

    # Add legend
    plt.legend()
    plt.grid(True)

    # Show or save the plot
    plt.savefig(
        f"../artifacts/plots/{args.pcap_path.split('/')[-1][:-5]}_{args.attack}.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Adversarial Attack on PANDA", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
