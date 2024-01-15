import datetime
import logging
import os
import pickle
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def set_logger(args):
    dataset_name = args.model_name.split("_")[0]
    log_folder = f"logs/{dataset_name}"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = f"{log_folder}/{args.attack_name}.log"
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(
        f"----------------------------- {current_time} -----------------------------"
    )
    logger.info("Initialised logger!!!!")

    return logger


def accuracy(Y: List, predY: List) -> float:
    """
    Get accuracy
    """
    Y = np.array(Y)
    predY = np.array(predY)
    accuracy = (Y == predY).sum() / float(len(Y))
    accuracy = np.round(accuracy * 100, 2)

    return accuracy


def save(path: str, params: Dict) -> None:
    """
    Save model to path
    """
    outfile = open(path, "wb")
    pickle.dump(params, outfile)
    outfile.close()


def load(path: str) -> Dict:
    """
    Load model from path
    """
    infile = open(path, "rb")
    params = pickle.load(infile)
    infile.close()

    return params


def save_checkpoint_autoencoder(
    epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path
):
    state = {
        "epoch": epoch,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }

    filename = path
    torch.save(state, filename)


def save_checkpoint_autoencoder_new(epoch, model, optimizer, path):
    state = {"epoch": epoch, "model": model, "optimizer": optimizer}

    filename = path
    torch.save(state, filename)


def visualize_cifar_reconstructions(input_imgs, reconst_imgs, file_name):
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    imgs = imgs.cpu().detach()
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=False, range=(0, 255))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title("Reconstructed image from the latent codes")
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig(f"../img/{file_name}.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(accuracy(a, b))
