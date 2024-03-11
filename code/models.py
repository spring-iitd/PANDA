import pickle

import numpy as np
import torch
import torch.nn as nn
from constants import clusters


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, z):
#         squared_difference = (x - z) ** 2
#         mean = torch.mean(squared_difference)
#         rmse = torch.sqrt(mean)
#         return rmse


class CNNAutoencoder(nn.Module):
    """
    CNN Autoencoder to cater the bit representation of the pcap files.
    Some of the fields are converted to bits and stacked together to make
    a 235x235 image.
    """

    def __init__(self):
        super().__init__()
        self.dataset = "PcapDataset"
        self.input_dim = 235
        self.raw = False

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Max pooling layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Max pooling layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2
            ),  # Deconvolutional layer 1
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2
            ),  # Deconvolutional layer 2
            nn.Sigmoid(),  # Sigmoid activation for output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderRaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.dataset = "PcapDatasetRaw"
        self.input_dim = 102
        self.raw = True

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),  # Linear layer 1
            nn.ReLU(),
            nn.Linear(64, 32),  # Linear layer 2
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),  # Linear layer 1
            nn.ReLU(),
            nn.Linear(64, self.input_dim),  # Linear layer 2
            nn.Sigmoid(),  # Sigmoid activation for output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x


class KitNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.dataset = "PcapDatasetRaw"
        self.input_dim = 102
        self.raw = True
        self.hr = 0.75
        self.rmse = RMSELoss()
        self.tails = nn.ModuleList(
            [BaseAutoencoder(len(c), int(np.ceil(len(c) * self.hr))) for c in clusters]
        )
        self.head = BaseAutoencoder(
            len(clusters), int(np.ceil(len(clusters) * self.hr))
        )
        with open(
            "../artifacts/objects/anomaly_detectors/kitsune/norm_params.pkl", "rb"
        ) as f:
            self.norm_params = pickle.load(f)

    def forward(self, x):
        x = x.view(-1, 102)

        x_clusters = []
        for c in clusters:
            norm_max = torch.tensor(self.norm_params[f"norm_max_{c[0]}"])
            norm_min = torch.tensor(self.norm_params[f"norm_min_{c[0]}"])

            x_cluster = torch.index_select(x, 1, torch.tensor(c))
            x_cluster = (x_cluster - norm_min) / (
                norm_max - norm_min + 0.0000000000000001
            )
            x_cluster = x_cluster.float()

            x_clusters.append(x_cluster)

        tail_losses = []
        for tail, c in zip(self.tails, x_clusters):
            output = tail(c)
            loss = self.rmse(output, c)
            if loss.data == 0:
                loss.data = torch.tensor(1e-2)
            tail_losses.append(loss)

        tails = torch.stack(tail_losses)

        norm_max = torch.tensor(self.norm_params["norm_max_output"])
        norm_min = torch.tensor(self.norm_params["norm_min_output"])
        tails = (tails - norm_min) / (norm_max - norm_min + 0.0000000000000001)
        tails = tails.float()
        x = self.head(tails)

        return x, tails
