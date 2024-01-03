import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Autoencoder
from datasets import PcapDataset

# Define transformations (if needed)
transform = transforms.Compose([
    # Add any desired transformations here
])
# Define loss function (Binary Cross-Entropy Loss for binary data)
criterion = nn.BCELoss()
mask = torch.zeros(235, 235)
mask[:, :32] = 1

def get_timegaps(packet):
    first_32_bits = packet[:, :32]
    integer_values = []
    for row in first_32_bits:
        binary_string = ''.join(str(int(x)) for x in row)
        integer_value = int(binary_string, 2)
        integer_values.append(integer_value/ 1000000)

    return integer_values, sum(integer_values)

def evaluate_perturbed_packets():
    ...

def fgsm(pcap_path, epsilon):
    # Load the trained autoencoder model
    model = Autoencoder()
    model.load_state_dict(torch.load('../artifacts/models/autoencoder_model_best.pth'))
    model.eval()

    batch_size = 1

    # Create the DataLoader
    dataset = PcapDataset(pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform)
    dataloader = DataLoader(dataset, batch_size=235 * batch_size, shuffle=False, drop_last=True)

    re = []
    adv_re = []
    y_true, y_pred = [], []
    taus = []
    total_time = 0
    adv_total_time = 0
    for packets in dataloader:
        _, sum_clean_timegaps = get_timegaps(packets)
        total_time = total_time + sum_clean_timegaps
        reshaped_packets = packets.reshape(batch_size, 1, 235, 235).to(torch.float)
        reshaped_packets.requires_grad = True
        outputs = model(reshaped_packets)

        # Compute the loss
        loss = criterion(outputs, reshaped_packets)
        re.append(loss.data)
        model.zero_grad()

        loss.backward()

        sign_data_grad = reshaped_packets.grad.data.sign()

        # Create the perturbed image by adjusting the original image
        perturbed_packets = reshaped_packets + (epsilon * sign_data_grad) * mask

        # Clip the perturbed image to ensure it stays within valid data range
        perturbed_packets = torch.clamp(perturbed_packets, 0, 1)

        adv_outputs = model(perturbed_packets)
        adv_loss = criterion(adv_outputs, perturbed_packets)
        adv_anomaly_score = -1 * adv_loss.data
        y_true.append(1 if "malicious" in pcap_path else 0)
        y_pred.append(1 if adv_anomaly_score > -0.2661924958229065 else 0)

        adv_re.append(adv_loss.data)

        perturbed_packets = perturbed_packets.squeeze()
        adv_timegaps, sum_adv_timegaps = get_timegaps(perturbed_packets)
        adv_total_time = adv_total_time + sum_adv_timegaps
        taus = taus + adv_timegaps

    print(f"Total time: {total_time}, Adv: {adv_total_time}")

    return re, adv_re, y_true, y_pred, taus

def pgd(pcap_path, epsilon):
    ...

import torch
from torch.utils.data import DataLoader
import sys

class Attack:
    def __init__(self, args):
        self.model_path = args.root_dir + "artifacts/models/" + args.surrogate_model + ".pth"
        self.batch_size = args.batch_size
        self.mask = torch.zeros(235, 235)
        self.pcap_path = args.pcap_path
        self.mask[:, :32] = 1
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = args.device
        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def fgsm(self, epsilon):
        dataset = PcapDataset(pcap_file=self.pcap_path, max_iterations=sys.maxsize, transform=transform)
        dataloader = DataLoader(dataset, batch_size=235 * self.batch_size, shuffle=False, drop_last=True)

        re = []
        adv_re = []
        y_true, y_pred = [], []
        taus = []
        total_time = 0
        adv_total_time = 0

        for packets in dataloader:
            _, sum_clean_timegaps = get_timegaps(packets)
            total_time = total_time + sum_clean_timegaps
            reshaped_packets = packets.reshape(self.batch_size, 1, 235, 235).to(torch.float).to(self.device)
            reshaped_packets.requires_grad = True
            outputs = self.model(reshaped_packets)

            # Compute the loss
            loss = criterion(outputs, reshaped_packets)
            re.append(loss.data)
            self.model.zero_grad()

            loss.backward()

            sign_data_grad = reshaped_packets.grad.data.sign()

            # Create the perturbed image by adjusting the original image
            perturbed_packets = reshaped_packets + (epsilon * sign_data_grad) * self.mask

            # Clip the perturbed image to ensure it stays within valid data range
            perturbed_packets = torch.clamp(perturbed_packets, 0, 1)

            adv_outputs = self.model(perturbed_packets)
            adv_loss = criterion(adv_outputs, perturbed_packets)
            adv_anomaly_score = -1 * adv_loss.data
            y_true.append(1 if "malicious" in self.pcap_path else 0)
            y_pred.append(1 if adv_anomaly_score > -0.2661924958229065 else 0)

            adv_re.append(adv_loss.data)

            perturbed_packets = perturbed_packets.squeeze()
            adv_timegaps, sum_adv_timegaps = get_timegaps(perturbed_packets)
            adv_total_time = adv_total_time + sum_adv_timegaps
            taus = taus + adv_timegaps

        print(f"Total time: {total_time}, Adv: {adv_total_time}")

        return re, adv_re, y_true, y_pred, taus
