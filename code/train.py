import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import *  # noqa
from feature_extractor import net_stat as ns
from models import *  # noqa
from torch.nn import *  # noqa
from torch.utils.data import DataLoader
from torchvision import transforms


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def _train_one_epoch(model, criterion, optimizer, dataloader, epoch, args):
    """
    Train the model for one epoch
    """
    # Set the model to training mode
    model.train()
    maxHost = 100000000000
    maxSess = 100000000000
    nstat = ns.netStat(np.nan, maxHost, maxSess)
    losses = []
    datas = []
    for i, packet in enumerate(dataloader):
        running_loss = 0.0
        if model.raw:
            tensors = []
            for j in range(len(packet["IPtype"])):
                x = nstat.updateGetStats(
                    packet["IPtype"][j].item(),
                    packet["srcMAC"][j],
                    packet["dstMAC"][j],
                    packet["srcIP"][j],
                    packet["srcproto"][j],
                    packet["dstIP"][j],
                    packet["dstproto"][j],
                    int(packet["framelen"][j]),
                    float(packet["timestamp"][j]),
                )
                tensors.append(torch.tensor(x))
            # concatenate with the tensors
            reshaped_packets = torch.cat(
                (packet["packet_tensor"], torch.stack(tensors)), dim=1
            ).to(torch.float)
            # reshaped_packets = torch.stack(tensors).to(torch.float)

        else:
            reshaped_packets = packet.reshape(
                (args.batch_size // model.input_dim),
                1,
                model.input_dim,
                model.input_dim,
            ).to(torch.float)

        # collect and store reshaped packet to create a csv
        datas.append(reshaped_packets)

        # Move the data to the device that is being used
        model = model.to(args.device)
        reshaped_packets = reshaped_packets.to(args.device)

        # Forward pass
        # below line is for regular models
        # outputs = model(reshaped_packets)
        # loss = criterion(outputs, reshaped_packets)

        # below line is for loopback pgd
        outputs, tails = model(reshaped_packets)
        loss = torch.log(criterion(outputs, tails))  # average loss over the batch
        losses.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print the loss and update the progress bar
        if (i + 1) % args.print_interval == 0:
            avg_loss = running_loss / args.print_interval
            print(f"Epoch {(i+1)}/ {(epoch+1)} Average Loss: {avg_loss}")

    # Save the reshaped packets to a csv
    stacked_datas = torch.stack(datas)
    reshaped_datas = stacked_datas.view(stacked_datas.shape[0], -1)
    np_data = reshaped_datas.cpu().detach().numpy()
    np.savetxt("../data/malicious/Port_Scanning_SmartTV.csv", np_data, delimiter=",")
    # plt.plot(losses)
    # plt.show()

    return losses


def trainer(args):
    """
    Train the model using args provided by the user
    """
    # Create an instance of the model
    model = eval(args.model_name)()

    # Define loss function (Binary Cross-Entropy Loss for binary data)
    criterion = eval(args.loss)()

    # Define optimizer
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

    # Define transformations (if needed)
    transform = transforms.Compose(
        [
            # Add any desired transformations here
        ]
    )

    if not model.raw:
        args.batch_size = model.input_dim * args.batch_size
    best_loss = float("inf")
    best_model_state = None

    # Training loop
    for epoch in range(args.num_epochs):
        # Create the dataset
        dataset = eval(model.dataset)(
            pcap_file=args.traindata_file,
            max_iterations=sys.maxsize,
            transform=transform,
        )

        # Create the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )

        # Train the model for one epoch
        losses: List = _train_one_epoch(
            model, criterion, optimizer, dataloader, epoch, args
        )

        avg_epoch_loss = sum(losses) / len(losses)
        # TODO: Add validation loop with early stopping here

        # Check if this is the best model so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"BEST LOSS: {best_loss}")
            best_model_state = model.state_dict()

            # Check if the folder exists, if not create it
            folder_path = f"../artifacts/models/{args.model_name}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Define the file path for saving the best model
            # TODO: Add the epoch number to the file name and save an entire state dictionary
            file_path = os.path.join(folder_path, "model.pth")

            # Save the best trained model
            torch.save(best_model_state, file_path)

        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss}")

    print(f"Best average reconstruction error over all the epochs: {best_loss}")
