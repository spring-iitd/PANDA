import sys

from constants import PCAP_PATH
from datasets import PcapDatasetRaw
from torch.utils.data import DataLoader

dataset = PcapDatasetRaw(pcap_file=PCAP_PATH, max_iterations=sys.maxsize)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
for i, packets in enumerate(dataloader):
    print(packets.time)
    if i == 20:
        break
