import torch
import struct
import scapy.all as scapy

from constants import MAX_BITS_PORT, MAX_BITS_SIZE

def extract_bits_from_packets(packet, prev_packet):
    """
    Extracts binary representations of various packet attributes from a given packet.

    Args:
        packet (scapy.Packet)     : The packet object to extract attributes from.
        prev_packet (scapy.Packet): The packet object to extract attributes from.

    Returns:
        torch.Tensor or None: A tensor containing the binary representations of extracted attributes, or None if extraction fails.

    This function extracts the following attributes from the packet:
    - Timediff: Time difference between two consecutive packets, both integer
      and fractional parts.
    - Source and Destination MAC Addresses.
    - Source and Destination IP Addresses (if available).
    - Source and Destination Ports (if available).
    - Packet Size.

    The extracted attributes are concatenated into a binary string representation and then converted to a tensor.

    Example:
    >>> packet = scapy.IP()/scapy.TCP()  # Create a sample packet
    >>> packet_tensor = extract_bits_from_packets(packet)
    """
    def extract_timestamp(packet):
        # Convert the integer part to binary
        integer_part = int(packet.time)
        binary_integer = bin(integer_part)[2:]

        # Convert the fractional part to binary
        fractional_part = packet.time - integer_part
        binary_fractional = bin(struct.unpack('!I', struct.pack('!f', fractional_part))[0])[2:]

        # Combine the integer and fractional binary parts
        timestamp_bits = binary_integer + binary_fractional
        if len(timestamp_bits) < 61:
            # print(len(timestamp_bits))
            diff = 61 - len(timestamp_bits)
            timestamp_bits = timestamp_bits + "0" * diff

        return timestamp_bits

    try:
        #### Extract timestamp ####
        # timestamp_bits = extract_timestamp(packet)

        current_time = float(packet.time)
        prev_time    = float(prev_packet.time)
        diff         = (current_time - prev_time) * 1000000
        diff         = int(max(diff, 0))

        timestamp_bits = bin(diff)[2:]
        
        if len(timestamp_bits) < 32:
            diff = 32 - len(timestamp_bits)
            timestamp_bits = timestamp_bits + "0" * diff

        #### Extract Src and Dst MAC Address ####
        if packet.haslayer(scapy.Ether):
            # Ethernet header
            src_mac = packet[scapy.Ether].src.split(":")
            src_mac_bits = ''.join(format(int(digit, 16), '08b') for digit in src_mac)
            dst_mac = packet[scapy.Ether].dst.split(":")
            dst_mac_bits = ''.join(format(int(digit, 16), '08b') for digit in dst_mac)
            mac_bits = src_mac_bits + dst_mac_bits
        else:
            mac_bits = "0" * 96

        #### Extract Src and Dst IP Address ####
        # IP and TCP header
        if packet.haslayer(scapy.IP):
            src_ip = packet[scapy.IP].src.split(".")
            src_ip_bits = ''.join(format(int(octet), '08b') for octet in src_ip)
            dst_ip = packet[scapy.IP].dst.split(".")
            dst_ip_bits = ''.join(format(int(octet), '08b') for octet in dst_ip)
            ip_bits = src_ip_bits + dst_ip_bits

            #### Extract Src and Dst Port ####
            sport = None
            dport = None
            if packet.haslayer(scapy.TCP):
                sport = packet[scapy.TCP].sport
                dport = packet[scapy.TCP].dport
            elif packet.haslayer(scapy.UDP):
                sport = packet[scapy.UDP].sport
                dport = packet[scapy.UDP].dport
            else:
                sport = 1055
                dport = 1055
            
            sport_bits = format(sport, f'0{MAX_BITS_PORT}b')
            dport_bits = format(dport, f'0{MAX_BITS_PORT}b')
            port_bits = sport_bits + dport_bits
        
            # ipip_bits = ip_bits + port_bits
            # print(len(ip_bits), len(port_bits))
            # print(ip_bits)
            # print(type(ip_bits))
        else:
            ip_bits = "0" * 64
            port_bits = "0" * 32

        #### Extract Packet Size ####
        packet_size = len(packet)
        packet_size_bits = format(packet_size, f'0{MAX_BITS_SIZE}b')

        #### Combine Each Bits ####
        packet_bits = timestamp_bits + mac_bits + ip_bits + port_bits + packet_size_bits
        # print(len(timestamp_bits), len(mac_bits), len(ip_bits), len(port_bits), len(packet_size_bits), len(packet_bits))
        packet_tensor = torch.tensor([int(bit) for bit in packet_bits])

    except Exception as e:
        print(e)
        # return None
    
    return packet_tensor