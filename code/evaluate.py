import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             auc)

from constants import PCAP_PATH
from models import Autoencoder
from datasets import PcapDataset

# Define transformations (if needed)
transform = transforms.Compose([
    # Add any desired transformations here
])
# Define loss function (Binary Cross-Entropy Loss for binary data)
criterion = nn.BCELoss()

import matplotlib.pyplot as plt
import numpy as np

def metrics(y_true, y_preds):
    precision = precision_score(y_true, y_preds)
    recall    = recall_score(y_true, y_preds)
    f1        = f1_score(y_true, y_preds)

    # roc


def plot_re(pcap_path, dataset_name):
    # Load the trained autoencoder model
    model = Autoencoder()
    model.load_state_dict(torch.load('../artifacts/models/autoencoder_model_best.pth'))
    model.eval()

    batch_size = 1

    # Create the DataLoader
    dataset = PcapDataset(pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform)
    dataloader = DataLoader(dataset, batch_size=235 * batch_size, shuffle=False, drop_last=True)

    reconstruction_errors = []

    for packets in dataloader:
        reshaped_packets = packets.reshape(batch_size, 1, 235, 235).to(torch.float)
        outputs = model(reshaped_packets)

        # Compute the loss
        loss = criterion(outputs, reshaped_packets)
        reconstruction_errors.append(loss.data)
    
    # Generate x-axis values (image indices)
    image_indices = np.arange(len(reconstruction_errors))

    # Create a line curve (line plot)
    plt.figure(figsize=(10, 6))
    plt.plot(image_indices, reconstruction_errors, marker='o', linestyle='-', color='b')
    plt.title(f'Reconstruction Error Curve: {dataset_name}')
    plt.xlabel('Image Index')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)

    # Show or save the plot
    plt.show()
    plt.savefig("../artifacts/plots/RE_plot")

plot_re("../data/malicious/SYN_Flooding_SmartTV.pcap", "SYN Flooding Smart TV")
