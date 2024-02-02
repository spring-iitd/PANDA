# Import dependencies
import numpy as np
import torch
from datasets import PcapDatasetRaw
from feature_extractor import net_stat as ns
from scapy.all import ARP
from scapy.all import ICMP
from scapy.all import IP
from scapy.all import IPv6
from scapy.all import random
from scapy.all import sys
from scapy.all import TCP
from scapy.all import UDP
from torch.utils.data import DataLoader
from torchvision import transforms


def packet_parser(packet):
    IPtype = np.nan
    timestamp = packet.time
    framelen = len(packet)
    if packet.haslayer(IP):  # IPv4
        srcIP = packet[IP].src
        dstIP = packet[IP].dst
        IPtype = 0
    elif packet.haslayer(IPv6):  # ipv6
        srcIP = packet[IPv6].src
        dstIP = packet[IPv6].dst
        IPtype = 1
    else:
        srcIP = ""
        dstIP = ""

    if packet.haslayer(TCP):
        srcproto = str(packet[TCP].sport)
        dstproto = str(packet[TCP].dport)
    elif packet.haslayer(UDP):
        srcproto = str(packet[UDP].sport)
        dstproto = str(packet[UDP].dport)
    else:
        srcproto = ""
        dstproto = ""

    srcMAC = packet.src
    dstMAC = packet.dst
    if srcproto == "":  # it's a L2/L1 level protocol
        if packet.haslayer(ARP):  # is ARP
            srcproto = "arp"
            dstproto = "arp"
            srcIP = packet[ARP].psrc  # src IP (ARP)
            dstIP = packet[ARP].pdst  # dst IP (ARP)
            IPtype = 0
        elif packet.haslayer(ICMP):  # is ICMP
            srcproto = "icmp"
            dstproto = "icmp"
            IPtype = 0
        elif srcIP + srcproto + dstIP + dstproto == "":  # some other protocol
            srcIP = packet.src  # src MAC
            dstIP = packet.dst  # dst MAC

    return IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, framelen, timestamp


def isEvade(x):
    return random.choice([True, False])


pcap_path = "../data/benign/weekday_100k.pcap"


transform = transforms.Compose(
    [
        # Add any desired transformations here
    ]
)

dataset = PcapDatasetRaw(
    pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
)

maxHost = 100000000000
maxSess = 100000000000
nstat = ns.netStat(np.nan, maxHost, maxSess)

print("Reading PCAP file ...")
for packet in dataloader:
    # print(packet)
    while True:
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
        features = torch.cat((packet["packet_tensor"][0], torch.tensor(x)))
        print(x)
        if isEvade(x):
            print("Evade! Exiting...")
            break
        else:
            print("No Evade! Reverting...")
            nstat.revertUpdate()
            print("Reverted!")
            packet[i].time = packet[i].time + 0000
            print("Old time: " + str(timestamp) + " New time: " + str(packets[i].time))

        n = int(input("Enter next packet sequence:"))
        if n == 1:
            continue
        else:
            break


# print("Reading PCAP file via Scapy...")
# packets = rdpcap(path)
# limit = len(packets)
# print("Loaded " + str(len(packets)) + " Packets.")

# maxHost = 100000000000
# maxSess = 100000000000
# nstat = ns.netStat(np.nan, maxHost, maxSess)

# for i in range(limit):
#     nstat.updatePreviousStats()
#     while True:
#         IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, framelen, timestamp = packet_parser(packets[i])
#         print(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, framelen, timestamp)
#         try:
#             x = nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
#                                                 int(framelen),
#                                                 float(timestamp))
#         except Exception as e:
#             print(e)
#             x = []

#         print(x)
#         if isEvade(x):
#             print("Evade! Exiting...")
#             break
#         else:
#             print("No Evade! Reverting...")
#             nstat.revertUpdate()
#             print("Reverted!")
#             packets[i].time = packets[i].time + 0000
#             print("Old time: " + str(timestamp) + " New time: " + str(packets[i].time))

#     n = int(input("Enter next packet sequence:"))
#     if n == 1:
#         continue
#     else:
#         break
