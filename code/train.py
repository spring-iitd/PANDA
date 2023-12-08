import sys
import torch

from models import Autoencoder
from datasets import PcapDataset
from constants import PCAP_PATH

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader

# Create an instance of the autoencoder
autoencoder = Autoencoder()

# Define loss function (Binary Cross-Entropy Loss for binary data)
criterion = nn.BCELoss()

# Define optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Define transformations (if needed)
transform = transforms.Compose([
    # Add any desired transformations here
])

batch_size = 8

best_loss = float('inf')
best_model_state = None
# Training loop
num_epochs = 30
print_interval = 5  # Print every 20000 packets

for epoch in range(num_epochs):
    running_loss = 0.0
    
    # Create the dataset
    dataset = PcapDataset(pcap_file=PCAP_PATH, max_iterations=sys.maxsize, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=235 * batch_size, shuffle=False, drop_last=True)

    for i, packets in enumerate(dataloader):

        reshaped_packets = packets.reshape(batch_size, 1, 235, 235).to(torch.float)
        outputs = autoencoder(reshaped_packets)

        # Compute the loss: we're getting average loss over the batch here
        loss = criterion(outputs, reshaped_packets)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Running loss before: {running_loss}")
        running_loss += loss.item()
        # print(f"Running loss after: {running_loss}")

        # Print the loss and update the progress bar
        if (i + 1) % print_interval == 0:
            avg_loss = running_loss / print_interval
            print(f"Epoch {(i+1)}/ {(epoch+1)} Average Loss: {avg_loss}")
    
    # Calculate average loss for the epoch
    avg_epoch_loss = running_loss / (i+1)
    
    # Check if this is the best model so far
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        print(f"BEST LOSS: {best_loss}")
        best_model_state = autoencoder.state_dict()

    # This is the average reconstruction error for the entire dataset. Use this to form the threshold
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss}")
    # Save the model checkpoint after every epoch
    # torch.save(autoencoder.state_dict(), f'autoencoder_model_epoch_{epoch+1}.pth')

# Save the best trained model
torch.save(best_model_state, '../artifacts/models/autoencoder_model_best.pth')
print(f"Best average reconstruction error over the entire dataset: {best_loss}")
