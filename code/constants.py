# consts

PCAP_PATH = "../data/benign/weekday.pcap"
MAX_BITS_PORT = 16
MAX_BITS_SIZE = 11
PACKET_LENGTH = 235

benign_data = [
    # "../data/benign/weekday_100k.pcap",
    # "../data/benign/weekday_06.pcap",
]

malicious_data = [
    # "../data/malicious/UDP_Flooding_Lenovo_Bulb_1.pcap",
    "../data/malicious/Port_Scanning_SmartTV_Filtered.pcap",
    # "../data/malicious/" + pcap for pcap in os.listdir("../data/malicious/")
]

merged_data = benign_data + malicious_data
# merged_data = benign_data

clusters = [
    [0, 1],
    [23, 30],
    [37],
    [66, 65, 62, 59, 53, 56, 63, 60, 54, 57],
    [73, 80, 87, 94, 101, 86, 93, 100, 72, 79],
    [44, 51, 50, 43, 36, 22, 29],
    [13, 40, 16, 47],
    [69, 76, 83, 90, 97],
    [10, 33, 4, 19, 7, 26],
    [41, 48, 84, 91, 98, 70, 77, 34, 20, 27],
    [14, 45],
    [11, 38],
    [64, 95],
    [61, 88],
    [8, 31],
    [58, 81, 52, 67, 55, 74, 2, 17, 5, 24],
    [68, 75, 82, 89, 96],
    [12, 39, 15, 46, 9, 32, 3, 18, 6, 25],
    [92, 99, 85, 71, 78, 42, 49, 35, 21, 28],
]

kitsune_clusters = [
    [21, 28],
    [35],
    [64, 63, 60, 57, 51, 54, 61, 58, 52, 55],
    [71, 78, 85, 92, 99, 84, 91, 98, 70, 77],
    [42, 49, 48, 41, 34, 20, 27],
    [11, 38, 14, 45],
    [67, 74, 81, 88, 95],
    [8, 31, 2, 17, 5, 24],
    [39, 46, 82, 89, 96, 68, 75, 32, 18, 25],
    [12, 43],
    [9, 36],
    [62, 93],
    [59, 86],
    [6, 29],
    [56, 79, 50, 65, 53, 72, 0, 15, 3, 22],
    [66, 73, 80, 87, 94],
    [10, 37, 13, 44, 7, 30, 1, 16, 4, 23],
    [90, 97, 83, 69, 76, 40, 47, 33, 19, 26],
]
