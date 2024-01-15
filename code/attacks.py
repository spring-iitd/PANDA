import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


# Define transformations (if needed)
transform = transforms.Compose(
    [
        # Add any desired transformations here
    ]
)


def get_timegaps(packet):
    first_32_bits = packet[:, :1]
    integer_values = []
    for row in first_32_bits:
        # binary_string = ''.join(str(int(x)) for x in row)
        # integer_value = int(binary_string, 2)
        integer_values.append(row.item() / 1000000)

    return integer_values, sum(integer_values)


class Attack:
    def __init__(self, args):
        self.model = eval(args.surrogate_model)().to(args.device)
        self.model.load_state_dict(
            torch.load(f"../artifacts/models/{args.surrogate_model}/model.pth")
        )
        # self.model_path = args.root_dir + "artifacts/models/AutoencoderInt/" + args.surrogate_model + ".pth"
        self.batch_size = args.batch_size
        self.pcap_path = args.pcap_path

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = args.device
        # TODO: store models along with loss function and optimizer and load here.
        self.criterion = nn.BCELoss()
        # self.model = AutoencoderInt().to(self.device)
        # self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        args.selected_columns = args.selected_columns + [self.model.input_dim - 1]
        self.mask = torch.zeros(self.model.input_dim, self.model.input_dim).to(
            self.device
        )
        self.mask[:, args.selected_columns] = 1

    def fgsm(self, epsilon):
        dataset = eval(self.model.dataset)(
            pcap_file=self.pcap_path, max_iterations=sys.maxsize, transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.model.input_dim * self.batch_size,
            shuffle=False,
            drop_last=True,
        )

        re = []
        adv_re = []
        y_true, y_pred = [], []
        taus = []
        total_time = 0
        adv_total_time = 0

        for packets in dataloader:
            _, sum_clean_timegaps = get_timegaps(packets)
            total_time = total_time + sum_clean_timegaps
            reshaped_packets = (
                packets.reshape(
                    self.batch_size, 1, self.model.input_dim, self.model.input_dim
                )
                .to(torch.float)
                .to(self.device)
            )
            reshaped_packets.requires_grad = True
            outputs = self.model(reshaped_packets)

            # Compute the loss
            loss = self.criterion(outputs, reshaped_packets)
            re.append(loss.data)
            self.model.zero_grad()

            loss.backward()

            sign_data_grad = reshaped_packets.grad.data.sign()

            # Create the perturbed image by adjusting the original image
            perturbed_packets = (
                reshaped_packets + (epsilon * sign_data_grad) * self.mask
            )

            # Clip the perturbed image to ensure it stays within valid data range
            perturbed_packets = torch.clamp(perturbed_packets, 0, 1)

            adv_outputs = self.model(perturbed_packets)
            adv_loss = self.criterion(adv_outputs, perturbed_packets)
            adv_anomaly_score = -1 * adv_loss.data
            y_true.append(1 if "malicious" in self.pcap_path else 0)
            y_pred.append(1 if adv_anomaly_score > -0.20200824737548828 else 0)

            adv_re.append(adv_loss.data)

            perturbed_packets = perturbed_packets.squeeze()
            adv_timegaps, sum_adv_timegaps = get_timegaps(perturbed_packets)
            adv_total_time = adv_total_time + sum_adv_timegaps
            taus = taus + adv_timegaps

        print(f"Total time: {total_time}, Adv: {adv_total_time}")

        return re, adv_re, y_true, y_pred, taus

    def pgd(self, epsilon):
        """
        Performs PGD attack on the autoencoder model.
        Args:
            epsilon (float): The maximum perturbation that can be applied to each pixel.
        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: The clean and adversarial reconstruction errors, the true and predicted labels, and the timegaps.
        """
        # Define the PGD attack parameters
        num_steps = 10

        dataset = eval(self.model.dataset)(
            pcap_file=self.pcap_path, max_iterations=sys.maxsize, transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.model.input_dim * self.batch_size,
            shuffle=False,
            drop_last=True,
        )

        re = []
        adv_re = []
        y_true, y_pred = [], []
        taus = []
        total_time = 0
        adv_total_time = 0

        for packets in dataloader:
            _, sum_clean_timegaps = get_timegaps(packets)
            total_time = total_time + sum_clean_timegaps

            # Get the clean input
            reshaped_packets = (
                packets.reshape(
                    self.batch_size, 1, self.model.input_dim, self.model.input_dim
                )
                .to(torch.float)
                .to(self.device)
            )
            reshaped_packets.requires_grad = True

            # Generate the adversarial example
            adversarial_packets = reshaped_packets.clone().detach()
            adversarial_packets.requires_grad = True

            # clean loss
            outputs = self.model(reshaped_packets)
            clean_loss = self.criterion(outputs, reshaped_packets)
            for _ in range(num_steps):
                # Forward pass through the autoencoder
                reconstructed_output = self.model(adversarial_packets)

                # Compute the loss
                loss = self.criterion(reconstructed_output, adversarial_packets)

                # Backward pass and gradient ascent
                self.model.zero_grad()
                loss.backward()

                adversarial_packets.data = (
                    adversarial_packets
                    + epsilon * torch.sign(adversarial_packets.grad.data) * self.mask
                )
                adversarial_packets = torch.clamp(adversarial_packets, 0, 1).detach()
                adversarial_packets.requires_grad = True

            re.append(clean_loss.data)
            adv_outputs = self.model(adversarial_packets)
            adv_loss = self.criterion(adv_outputs, adversarial_packets)
            adv_anomaly_score = -1 * adv_loss.data
            y_true.append(1 if "malicious" in self.pcap_path else 0)
            y_pred.append(1 if adv_anomaly_score > -0.20200824737548828 else 0)

            adv_re.append(adv_loss.data)

            adversarial_packets = adversarial_packets.squeeze()
            adv_timegaps, sum_adv_timegaps = get_timegaps(adversarial_packets)
            adv_total_time = adv_total_time + sum_adv_timegaps
            taus = taus + adv_timegaps

        print(f"Total time: {total_time}, Adv: {adv_total_time}")

        return re, adv_re, y_true, y_pred, taus
