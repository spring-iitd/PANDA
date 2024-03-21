import sys
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import *  # noqa
from feature_extractor import net_stat as ns
from models import *  # noqa
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose(
    [
        # Add any desired transformations here
    ]
)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def get_timegaps(packet):
    first_32_bits = packet[:, :1]
    integer_values = []
    for row in first_32_bits:
        # binary_string = ''.join(str(int(x)) for x in row)
        # integer_value = int(binary_string, 2)
        integer_values.append(row.item() / 1000000)

    return integer_values, sum(integer_values)


def denormalize_packet_size(normalized_size):
    min_size = 64  # minimum frame size
    max_size = 1518  # maximum frame size

    packet_size = normalized_size * (max_size - min_size) + min_size
    return packet_size


class Attack:
    def __init__(self, args):
        self.model = eval(args.surrogate_model)().to(args.device)
        self.model.load_state_dict(
            torch.load(f"../artifacts/models/{args.surrogate_model}/model.pth")
        )
        self.batch_size = args.batch_size
        self.pcap_path = args.pcap_path
        self.device = args.device
        # TODO: store models along with loss function and optimizer and load here.
        self.criterion = nn.BCELoss()
        self.model.eval()
        # args.selected_columns = args.selected_columns + [self.model.input_dim - 1]
        self.mask = torch.zeros(self.model.input_dim, self.model.input_dim).to(
            self.device
        )
        self.mask[:, args.selected_columns] = 1
        self.threshold = args.threshold

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
            y_pred.append(1 if adv_anomaly_score > self.threshold else 0)

            adv_re.append(adv_loss.data)

            adversarial_packets = adversarial_packets.squeeze()
            adv_timegaps, sum_adv_timegaps = get_timegaps(adversarial_packets)
            adv_total_time = adv_total_time + sum_adv_timegaps
            taus = taus + adv_timegaps

        print(f"Total time: {total_time}, Adv: {adv_total_time}")

        return re, adv_re, y_true, y_pred, taus

    def loopback_pgd(self, epsilon):
        # The attack is designed for one packet at a time
        self.criterion = RMSELoss()

        print(f"Attacking {self.pcap_path}")

        # creating mask for the timestamp and size
        # TODO: #absolute mask (make it diff for different attacks)
        self.mask = torch.zeros(self.model.input_dim).to(self.device)
        self.mask[0] = 1
        self.mask[1] = 1

        dataset = eval(self.model.dataset)(
            pcap_file=self.pcap_path, max_iterations=sys.maxsize, transform=transform
        )
        # TODO: Remove #absolute batch_size = 1
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )

        re = []
        adv_re = []
        y_true, y_pred = [], []
        taus = []
        adv_timestamps = []
        actual_sizes = []
        adv_sizes = []
        total_time = 0
        adv_total_time = 0

        # nstat stuff
        maxHost = 100000000000
        maxSess = 100000000000
        nstat = ns.netStat(np.nan, maxHost, maxSess)
        nstat_alias = ns.netStat(np.nan, maxHost, maxSess)

        start = time.time()
        for i, packet in enumerate(dataloader):
            # if i == 0:
            #     start_time = packet["timestamp"]
            # packet["timestamp"] = packet["timestamp"] - start_time

            total_time = total_time + packet["packet_tensor"][0][0].item()

            # updating the previous stats to till packet i-1
            # if the current packet is a malicious packet
            x_alias = nstat_alias.updateGetStats(
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
            reshaped_packets_alias = (
                torch.cat((packet["packet_tensor"][0], torch.tensor(x_alias)))
                .to(torch.float)
                .to(self.device)
            )
            outputs, tails = self.model(reshaped_packets_alias)
            clean_loss_alias = self.criterion(outputs, tails)
            re.append(clean_loss_alias.data)

            # if clean_loss_alias < self.threshold:
            #     print("Loss of malicious file less than the threshold, Evaded!!!")
            #     continue
            # else:
            #     # print("Loss of malicious file greater than the threshold, Attacking!!!")
            #     nstat.updatePreviousStats()

            nstat.updatePreviousStats()
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

            reshaped_packets = (
                torch.cat((packet["packet_tensor"][0], torch.tensor(x)))
                .to(torch.float)
                .to(self.device)
            )
            reshaped_packets.requires_grad = True
            # Generate the adversarial example
            adversarial_packets = reshaped_packets.clone().detach()
            adversarial_packets.requires_grad = True

            outputs, tails = self.model(reshaped_packets)
            clean_loss = self.criterion(outputs, tails)
            clean_anomaly_score = clean_loss.data

            original_size = int(packet["framelen"])
            original_timestamp = float(packet["timestamp"])
            original_iat = packet["packet_tensor"][0][0].item()

            adversarial_size = int(packet["framelen"])
            adversarial_timestamp = float(packet["timestamp"])
            adversarial_iat = packet["packet_tensor"][0][0].item()

            num_steps = 10
            for step in range(num_steps):
                # here the first adversarial packet is the original packet
                reconstructed_output, tails = self.model(adversarial_packets)
                loss = self.criterion(reconstructed_output, tails)

                if loss < self.threshold:
                    print("Loss of malicious file less than the threshold, Evaded!!!")
                    break

                # Backward pass and gradient ascent
                self.model.zero_grad()
                loss.backward()

                # calculate perturbations
                # delta = epsilon * torch.sign(adversarial_packets.grad.data) * self.mask
                # adversarial_packets.data = adversarial_packets - delta
                # # TODO: #absolute
                # denormalised_delta = delta[0].item() * (1 - 0.00001) + 0.00001
                # adversarial_timestamp = min((adversarial_timestamp - denormalised_delta), 0.00001)
                # denormalized_adv_size = denormalize_packet_size(
                #     adversarial_packets[1].item()
                # )
                iat_epsilon = 0.1
                size_epsilon = 2
                iat_delta = torch.sign(adversarial_packets.grad[0]) * iat_epsilon
                size_delta = torch.sign(adversarial_packets.grad[1]) * size_epsilon

                # pgd signs
                # torch.sign(adversarial_packets.grad[2:])

                # TODO: Adjust the entire if else loop also introduce new packets in between
                print(f"Timestamp before perturbation: {original_timestamp}")
                print(f"Original IAT: {original_iat}")
                # adversarial_iat = max((adversarial_iat - iat_delta), 0.00001)
                adversarial_iat = original_iat - iat_delta.item()
                adversarial_timestamp = original_timestamp - iat_delta.item()
                print(f"Adversarial IAT: {adversarial_iat}")
                print(f"Timestamp after perturbation: {adversarial_timestamp}")

                print(f"Size before perturbation: {original_size}")
                adversarial_size = adversarial_size - size_delta
                adversarial_size_normalized = (adversarial_size - 64) / (1518 - 64)
                print(f"Size after perturbation: {adversarial_size}")

                if step == 5:
                    last_three = int(packet["srcIP"][0].split(".")[-1]) + 1
                    packet["srcIP"][0] = packet["srcIP"][0][:-3] + str(last_three)

                # input("Enter something: ")

                print(
                    f"Loss: {clean_loss_alias}, Threshold: {self.threshold}, Adv Loss: {loss}"
                )

                # adjust other features according to the modified timestamp
                # step 1: revert the nstat update
                nstat.revertUpdate()

                # step 2: update the nstat with the new timestamp
                x = nstat.updateGetStats(
                    packet["IPtype"].item(),
                    packet["srcMAC"][0],
                    packet["dstMAC"][0],
                    packet["srcIP"][0],
                    packet["srcproto"][0],
                    packet["dstIP"][0],
                    packet["dstproto"][0],
                    adversarial_size,
                    # int(packet["framelen"]),
                    adversarial_timestamp,
                )

                # calculate loopback sign

                # step 3: concatenate with the tensors to get new adversarial packets
                adversarial_packets = (
                    torch.cat(
                        (
                            torch.tensor(adversarial_iat).reshape(
                                1,
                            ),
                            adversarial_size_normalized.reshape(
                                1,
                            ),
                            #  packet["packet_tensor"][0][1].reshape(1,),
                            torch.tensor(x),
                        )
                    )
                    .to(torch.float)
                    .to(self.device)
                )
                # adversarial_packets = (
                #     torch.cat(
                #         (adversarial_packets[0:2].detach().cpu(), torch.tensor(x))
                #     )
                #     .to(torch.float)
                #     .to(self.device)
                # )

                adversarial_packets.requires_grad = True

            print(
                f"Loss: {clean_loss.data}, Threshold: {self.threshold}, Adv Loss: {loss}"
            )

            # calculating adversarial loss (if loop exited for maximum iterations not evading)
            adv_outputs, tails = self.model(adversarial_packets)
            adv_loss = self.criterion(adv_outputs, tails)
            adv_re.append(adv_loss.data)

            # predicting the label
            adv_anomaly_score = adv_loss.data
            y_true.append(1 if clean_anomaly_score > self.threshold else 0)
            y_pred.append(1 if adv_anomaly_score > self.threshold else 0)

            adversarial_packets = adversarial_packets.squeeze()
            adv_timegaps = adversarial_packets[0].item()
            adv_total_time = adv_total_time + adv_timegaps
            taus.append(adv_timegaps)
            adv_timestamps.append(adversarial_timestamp)
            actual_sizes.append(packet["framelen"])
            adv_sizes.append(adversarial_size)

        print(f"Total time: {total_time}, Adv: {adv_total_time}")
        print(f"Actual sizes: {sum(actual_sizes)}, Adv sizes: {sum(adv_sizes)}")

        print(f"Time taken for the attack: {time.time() - start}")

        return re, adv_re, y_true, y_pred, taus, adv_timestamps, adv_sizes, actual_sizes
