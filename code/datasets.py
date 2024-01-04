from preprocessing import extract_bits_from_packets

from scapy.utils import PcapReader
from torch.utils.data import Dataset
from preprocessing import FeatureRepresentation

class PcapDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing packets from a PCAP file.

    This class iterates over packets in a PCAP file and extracts features from each packet.
    It supports applying optional transformations to the extracted features.

    Args:
        pcap_file (str): Path to the PCAP file.
        max_iterations (int): Maximum number of iterations (packets) to process. If None, the entire dataset is processed.
        transform (callable, optional): A function/transform that takes in a packet tensor and returns a transformed tensor.

    Attributes:
        packets (PcapReader): A PcapReader object for reading packets from the PCAP file.
        prev_packets (PcapReader): A PcapReader object for reading previous packets from the PCAP file.
        transform (callable): The optional transform function.
        max_iterations (int): The maximum number of iterations (packets) to process.
    """

    def __init__(self, pcap_file, max_iterations, transform=None):
        self.packets = PcapReader(pcap_file)
        self.prev_packets = PcapReader(pcap_file)
        self.transform = transform
        self.max_iterations = max_iterations

    def __len__(self):
        """
        Returns the length of the dataset.

        If max_iterations is None, raises NotImplementedError.

        Returns:
            int: The length of the dataset (number of packets to process).
        """
        if self.max_iterations is not None:
            return self.max_iterations
        else:
            raise NotImplementedError("Length of StreamingDataset is not defined.")

    def __getitem__(self, index):
        """
        Retrieves the item at the specified index.

        Extracts features from the current and previous packets and applies the optional transform function.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: The transformed packet tensor.
        """
        if index == 0:
            packet = next(self.packets)
            packet_tensor = FeatureRepresentation().get_int_representation(packet, packet)
        else:
            packet = next(self.packets)
            prev_packet = next(self.prev_packets)
            packet_tensor = FeatureRepresentation().get_int_representation(packet, prev_packet)

        if self.transform:
            packet_tensor = self.transform(packet_tensor)

        return packet_tensor
