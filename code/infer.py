import sys
import torch
import torch.nn as nn

from models import Autoencoder
from datasets import PcapDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    # Add any desired transformations here
])

# pcap_path="../data/malicious/ACK_Flooding_Smart_Clock_1.pcap"
# pcap_path="../data/malicious/Port_Scanning_SmartTV.pcap"
pcap_path="../data/benign/weekday_06.pcap"
batch_size = 1
criterion = nn.BCELoss()

# Create the DataLoader
dataset = PcapDataset(pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform)
dataloader = DataLoader(dataset, batch_size=235 * batch_size, shuffle=False, drop_last=False)

model = Autoencoder()
model.load_state_dict(torch.load('../artifacts/models/autoencoder_model_best.pth'))
model.eval()

re = []
for packets in dataloader:
    reshaped_packets = packets.reshape(batch_size, 1, 235, 235).to(torch.float)
    outputs = model(reshaped_packets)

    loss = criterion(outputs, reshaped_packets)

    re.append(loss.data)

print(re)
print(sum(re)/ len(re))
