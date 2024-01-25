import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import *  # noqa
from infer import infer
from models import *  # noqa
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from train import trainer
from utils import load

transform = transforms.Compose(
    [
        # Add any desired transformations here
    ]
)

# def run(pcap_path, dataset_name):
#     # Load the trained autoencoder model
#     model = Autoencoder()
#     model.load_state_dict(torch.load('../artifacts/models/autoencoder_model_best.pth'))
#     model.eval()

#     batch_size = 1

#     # Create the DataLoader
#     dataset = PcapDataset(pcap_file=pcap_path, max_iterations=sys.maxsize, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=235 * batch_size, shuffle=False, drop_last=True)

#     reconstruction_errors = []

#     for packets in dataloader:
#         reshaped_packets = packets.reshape(batch_size, 1, 235, 235).to(torch.float)
#         outputs = model(reshaped_packets)

#         # Compute the loss
#         loss = criterion(outputs, reshaped_packets)
#         reconstruction_errors.append(loss.data)

#     # Generate x-axis values (image indices)
#     image_indices = np.arange(len(reconstruction_errors))

#     # Create a line curve (line plot)
#     plt.figure(figsize=(10, 6))
#     plt.plot(image_indices, reconstruction_errors, marker='o', linestyle='-', color='b')
#     plt.title(f'Reconstruction Error Curve: {dataset_name}')
#     plt.xlabel('Image Index')
#     plt.ylabel('Reconstruction Error')
#     plt.grid(True)

#     # Show or save the plot
#     plt.show()
#     plt.savefig("../artifacts/plots/RE_plot")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "PANDA: Model Training and Inference", add_help=False
    )
    parser.add_argument(
        "--root-dir",
        default="../",
        help="folder where all the code, data, and artifacts lie",
    )
    # model related arguments
    parser.add_argument("--model-name", default="Autoencoder", type=str)
    parser.add_argument("--loss", default="BCELoss", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--lr", default=0.001, type=float)

    # training related arguments
    parser.add_argument("--num-epochs", default=30, type=int)
    parser.add_argument("--print-interval", default=5, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--traindata-file", default="../data/benign/weekday_06.pcap")
    # TODO: Incorporate traindata-len in the training loop (currently not used)
    parser.add_argument(
        "--traindata-len",
        default=10000,
        type=int,
        help="number of packets used to train the model",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    # inference related arguments
    parser.add_argument(
        "--eval", action="store_true", default=False, help="perform inference"
    )
    parser.add_argument(
        "--get-threshold",
        action="store_true",
        default=False,
        help="should the threshold be calculated or already provided",
    )
    parser.add_argument("--threshold", type=float, help="threshold for the autoencoder")

    return parser


criterion = nn.BCELoss()

# def get_threshold(args, model):
#     # Create the DataLoader
#     dataset = PcapDataset(pcap_file=args.traindata_file, max_iterations=sys.maxsize, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=194 * args.batch_size, shuffle=False, drop_last=True)

#     reconstruction_errors = []

#     for packets in dataloader:
#         reshaped_packets = packets.reshape(args.batch_size, 1, 194, 194).to(torch.float)
#         outputs = model(reshaped_packets)

#         # Compute the loss
#         loss = criterion(outputs, reshaped_packets)
#         reconstruction_errors.append(loss.data)

#     # finding the 95th percentile of the reconstruction error distribution for threshold
#     reconstruction_errors.sort(reverse=True)
#     ninety_fifth_percentile_index = int(0.90 * len(reconstruction_errors))
#     threshold = reconstruction_errors[ninety_fifth_percentile_index]

#     return threshold


def plot_recon(args):
    # Load the trained autoencoder model
    model = eval(args.model_name)()
    model.load_state_dict(
        torch.load(f"../artifacts/models/{args.model_name}/model.pth")
    )
    model = model.to(args.device)
    model.eval()
    print("Loaded the model in eval mode for plotting!!!")

    transform = transforms.Compose(
        [
            # Add any desired transformations here
        ]
    )

    # model = Autoencoder()
    # model.load_state_dict(torch.load('autoencoder_model_best.pth'))
    # model.eval()

    batch_size = 1

    # Create the DataLoader
    dataset = eval(model.dataset)(
        pcap_file=args.traindata_file, max_iterations=sys.maxsize, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=model.input_dim, shuffle=False, drop_last=True
    )

    for packets in dataloader:
        reshaped_packets = packets.reshape(
            batch_size, 1, model.input_dim, model.input_dim
        ).to(torch.float)
        outputs = model(reshaped_packets)

        # Convert tensors to numpy arrays for plotting
        original_image = reshaped_packets[0].squeeze().cpu().numpy()
        reconstructed_image = outputs[0].squeeze().detach().cpu().numpy()

        # Create subplots for original and reconstructed images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(args.traindata_file.split(".")[0], fontsize=14)
        axes[0].set_title("Original Image")
        axes[0].imshow(original_image, cmap="gray")
        axes[0].axis("off")

        axes[1].set_title("Reconstructed Image")
        axes[1].imshow(reconstructed_image, cmap="gray")
        axes[1].axis("off")

        # Show the images side by side
        plt.tight_layout()
        plt.show()
        break


def main(args):
    if args.eval:
        # TODO: Remove this
        saved = False

        if saved:
            anomaly_scores = load(
                "../artifacts/objects/anomaly_detectors/autoencoder/anomaly_scores"
            )["anomaly_scores"]
            y_true = load("../artifacts/objects/anomaly_detectors/autoencoder/y_true")[
                "y_true"
            ]
            y_pred = load("../artifacts/objects/anomaly_detectors/autoencoder/y_pred")[
                "y_pred"
            ]

        else:
            y_true, y_pred, anomaly_scores = infer(args)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1_score)

        # ROC and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        auc_score = roc_auc_score(y_true, y_pred)
        plt.text(0.5, 0.5, f"AUC score: {auc_score}", ha="center", va="center")

        plt.show()

    else:
        print("Training the model!!!")
        trainer(args)

    if not args.model_name == "AutoencoderRaw":
        plot_recon(args)
        print("Plotting the original and reconstructed images!!!")

    print("Done!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Model training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
