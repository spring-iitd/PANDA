import torch
import torch.nn as nn
from constants import kitsune_clusters


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# Define the CNN Autoencoder architecture
class Autoencoder(nn.Module):
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


class AutoencoderInt(nn.Module):
    def __init__(self):
        super().__init__()
        self.dataset = "PcapDatasetInt"
        self.input_dim = 194
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
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Deconvolutional layer 1
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, output_padding=1
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
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size), nn.Sigmoid())

    def forward(self, x):
        # print(x.shape)
        # print(self.encoder)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class KitsuneAutoencoder(nn.Module):
    def __init__(self):
        self.dataset = "PcapDatasetRaw"
        self.input_dim = 100
        self.raw = True
        super().__init__()
        self.tails = [BaseAutoencoder(len(c), len(c) // 2) for c in kitsune_clusters]
        self.head = BaseAutoencoder(len(kitsune_clusters), len(kitsune_clusters) // 2)
        self.rmse = RMSELoss()

    def forward(self, x):
        # cluster x by using index from clusters
        x = x.view(-1, 100)
        # print(f"Before clustering: {x}")
        # x_clusters = [torch.index_select(x, 1, torch.tensor(c)) for c in clusters]
        x_clusters = [
            torch.index_select(x, 1, torch.tensor(c)) for c in kitsune_clusters
        ]

        tails = []
        for tail, c in zip(self.tails, x_clusters):
            output = tail(c)
            loss = torch.log(self.rmse(output, c))
            tails.append(loss)

        # print(tails)
        tails = torch.stack(tails)
        x = self.head(tails)
        return x, tails
