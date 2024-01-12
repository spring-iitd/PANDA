import sys
import logging
import argparse

import struct
import scapy.all as scapy

from constants import benign_data, malicious_data, merged_data
from utils import set_logger, save, load

import torch
import torch.nn as nn

from models import Autoencoder
from attacks import Attack
from datasets import PcapDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_curve, roc_auc_score

#plotting
import numpy as np
import matplotlib.pyplot as plt

# pcap_path = "../data/malicious/Port_Scanning_SmartTV.pcap"
# pcap_path = "../data/malicious/Service_Detection_Smartphone_1.pcap"
# pcap_path = "../data/malicious/ARP_Spoofing_Google-Nest-Mini_1.pcap"
# pcap_path="../data/malicious/ACK_Flooding_Smart_Clock_1.pcap"
# pcap_path = "../data/malicious/SYN_Flooding_SmartTV.pcap"
# pcap_path = "../data/malicious/UDP_Flooding_Lenovo_Bulb_1.pcap"

# adv_pcap_path = "../data/adversarial/fgsm/Adv_" + pcap_path.split('/')[-1]


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

def get_args_parser():
    parser = argparse.ArgumentParser('PANDA: Adversarial Attack', add_help=False)
    parser.add_argument('--root-dir', default="../",
                        help="folder where all the code, data, and artifacts lie")
    parser.add_argument('--pcap-path', default="../data/malicious/Port_Scanning_SmartTV.pcap",
                        type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--device', default='cuda',
                        help="device to use for training/ testing")
    parser.add_argument('--surrogate-model', default='AutoencoderInt', type=str,
                        help="Name of the surrogate model")
    parser.add_argument('--target-model', default='kitsune', type=str,
                        help="Name of the target model")
    parser.add_argument('--attack', default='fgsm', type=str,
                        help="Name of the attack to perform or inference")
    parser.add_argument('--selected-columns', nargs='+', default=[0], type=list)
    parser.add_argument('--eval', action='store_true', default=False,
                        help='perform attack inference')

    return parser

criterion = nn.BCELoss()

def main(args):
    # create an object of the attack class
    args.adv_pcap_path = f"../data/adversarial/{args.attack}/Adv_{args.pcap_path.split('/')[-1]}"
    args.selected_columns = [x for x in range(80)]
    attack = Attack(args=args)
    attack_method = getattr(attack, args.attack)
    re, adv_re, y_true, y_pred, taus = attack_method(epsilon=0.8)

    print(f"Pcap file: {args.pcap_path.split('/')[-1][:-5]}")
    print(f"Mean RE for malicious packets: {sum(re)/ len(re)}")
    print(f"Mean RE for adversarial malicious packets: {sum(adv_re)/ len(adv_re)}")

    evasion_rate = 1 - sum(y_pred) / len(y_pred)
    print(f"Evasion Rate: {evasion_rate}")

    # create adversarial packets
    update_timestamps(args.pcap_path, taus, args.adv_pcap_path)

    # Generate x-axis values (image indices)
    image_indices = np.arange(len(re))

    # Create a line curve (line plot)
    plt.figure(figsize=(10, 6))
    plt.plot(image_indices, re, marker='o', linestyle='-', color='b', label="Clean Data")
    plt.plot(image_indices, adv_re, marker='o', linestyle='-', color='r', label="Advesarial Data")
    plt.axhline(y=0.2661, color='green', linestyle='--', label='Threshold')
    plt.title(f"Reconstruction Error Curve: {args.pcap_path.split('/')[-1][:-5]}_{args.attack}")
    plt.xlabel('Image Index')
    plt.ylabel('Reconstruction Error')

    # Add legend
    plt.legend()
    plt.grid(True)

    # Show or save the plot
    plt.savefig(f"../artifacts/plots/{args.pcap_path.split('/')[-1][:-5]}_{args.attack}.png")

    image_indices = np.arange(len(adv_re))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Adversarial Attack on PANDA', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
 