import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Autoencoder
from datasets import PcapDataset

def get_timegaps(packet):
    first_32_bits = packet[:, :32]
    integer_values = []
    for row in first_32_bits:
        binary_string = ''.join(str(int(x)) for x in row)
        integer_value = int(binary_string, 2)
        integer_values.append(integer_value/ 1000000)

    return integer_values, sum(integer_values)


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
        self.device = torch.device(args.device)

        self.transform = transforms.Compose([
            # Add any desired transformations here
        ])
        self.dataset = PcapDataset(pcap_file=self.pcap_path, max_iterations=sys.maxsize, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=235 * self.batch_size, shuffle=False, drop_last=True)
        self.criterion = nn.BCELoss()
        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def fgsm(self, epsilon):
        re = []
        adv_re = []
        y_true, y_pred = [], []
        taus = []
        total_time = 0
        adv_total_time = 0

        for packets in self.dataloader:
            _, sum_clean_timegaps = get_timegaps(packets)
            total_time = total_time + sum_clean_timegaps
            reshaped_packets = packets.reshape(self.batch_size, 1, 235, 235).to(torch.float).to(self.device)
            reshaped_packets.requires_grad = True
            outputs = self.model(reshaped_packets)

            # Compute the loss
            loss = self.criterion(outputs, reshaped_packets)
            re.append(loss.data)
            self.model.zero_grad()

            loss.backward()

            sign_data_grad = reshaped_packets.grad.data.sign()

            # Create the perturbed image by adjusting the original image
            perturbed_packets = reshaped_packets + (epsilon * sign_data_grad) * self.mask

            # Clip the perturbed image to ensure it stays within valid data range
            perturbed_packets = torch.clamp(perturbed_packets, 0, 1)

            adv_outputs = self.model(perturbed_packets)
            adv_loss = self.criterion(adv_outputs, perturbed_packets)
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
