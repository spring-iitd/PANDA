# Import the Tarda library
from tardigrade import AwesomeIDS

# Create an instance of the AwesomeIDS class
model = AwesomeIDS()

# Example: Parsing data from a pcap file
model.parse(
    "../data/benign/weekday_100k.pcap",
    "../data/benign/weekday_100k.csv",
    save_netstat="../data/benign/weekday_100k_netstat.pkl",
)
