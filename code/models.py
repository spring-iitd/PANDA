import torch.nn as nn


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
