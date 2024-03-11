import sys

import numpy as np
import torch
from feature_extractor import net_stat as ns
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = "PcapDatasetRaw"


# pcap_path = "../data/benign/weekday.pcap"
# pcap_path = "../data/malicious/Port_Scanning_SmartTV_Filtered_500.pcap"
pcap_path = (
    "../data/adversarial/loopback_pgd/Adv_Port_Scanning_SmartTV_Filtered_500.pcap"
)

transform = transforms.Compose(
    [
        # Add any desired transformations here
    ]
)

dataset = eval(dataset)(pcap_file=pcap_path, max_iterations=sys.maxsize, transform=None)
# TODO: Remove #absolute batch_size = 1
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
)

# nstat stuff
maxHost = 100000000000
maxSess = 100000000000
nstat = ns.netStat(np.nan, maxHost, maxSess)

features = []

for packet in dataloader:
    # Get the clean input
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
    reshaped_packets = (
        torch.cat((packet["packet_tensor"][0], torch.tensor(x)))
        .to(torch.float)
        .to("cpu")
    )

    features.append(reshaped_packets.numpy())

features_array = np.vstack(features)

file_path = pcap_path.replace(".pcap", ".csv")
np.savetxt(file_path, features_array, delimiter=",")
