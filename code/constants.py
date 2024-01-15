# consts
import os

PCAP_PATH = "../data/benign/weekday.pcap"
MAX_BITS_PORT = 16
MAX_BITS_SIZE = 11
PACKET_LENGTH = 235

benign_data = [
    "../data/benign/weekday_100k.pcap",
    # "../data/benign/weekday_06.pcap",
]

malicious_data = [
    "../data/malicious/" + pcap for pcap in os.listdir("../data/malicious/")
]

merged_data = benign_data + malicious_data

# merged_data = benign_data
