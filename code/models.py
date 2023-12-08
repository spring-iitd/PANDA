import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the CNN Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Max pooling layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Max pooling layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),  # Deconvolutional layer 1
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2),  # Deconvolutional layer 2
            nn.Sigmoid()  # Sigmoid activation for output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
