import scapy
import torch

from torchvision import transforms
from torch.utils.data import Dataset

from preprocessing import extract_bits_from_packets

class PacketDataset(Dataset):
    def __init__(self, pcap_file, transform=None):
        self.packets = scapy.rdpcap(pcap_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.packets)
    
    def __getitem__(self, index):
        packet = self.packets[index]
        packet_tensor = extract_bits_from_packets(packet)

        if self.transform:
            packet_tensor = self.transform(packet_tensor)

        return packet_tensor
