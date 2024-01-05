import torch
import struct
import scapy.all as scapy

from constants import MAX_BITS_PORT, MAX_BITS_SIZE

class FeatureRepresentation:
    def __init__(self):
        self.packet = None
        self.prev_packet = None

    def _extract_timestamp(self, get_integer=False):
        current_time = float(self.packet.time)
        prev_time    = float(self.prev_packet.time)
        int_diff     = current_time - prev_time

        if get_integer:
            timestamp_tensor = torch.tensor([int_diff])
        else:
            diff = int(max(int_diff * 1000000, 0))
            timestamp_bits = bin(diff)[2:]

            if len(timestamp_bits) < 32:
                padding = 32 - len(timestamp_bits)
                timestamp_bits = timestamp_bits + "0" * padding
                timestamp_tensor = torch.tensor([int(bit) for bit in timestamp_bits])

        return timestamp_tensor

    def _extract_mac_address(self):
        if self.packet.haslayer(scapy.Ether):
            src_mac = self.packet[scapy.Ether].src.split(":")
            src_mac_bits = ''.join(format(int(digit, 16), '08b') for digit in src_mac)
            dst_mac = self.packet[scapy.Ether].dst.split(":")
            dst_mac_bits = ''.join(format(int(digit, 16), '08b') for digit in dst_mac)
            mac_bits = src_mac_bits + dst_mac_bits
            mac_tensor = torch.tensor([int(bit) for bit in mac_bits])
        else:
            mac_tensor = torch.full((96,), -1)

        return mac_tensor

    def _extract_ip_address(self):
        if self.packet.haslayer(scapy.IP):
            src_ip = self.packet[scapy.IP].src.split(".")
            src_ip_bits = ''.join(format(int(octet), '08b') for octet in src_ip)
            dst_ip = self.packet[scapy.IP].dst.split(".")
            dst_ip_bits = ''.join(format(int(octet), '08b') for octet in dst_ip)
            ip_bits = src_ip_bits + dst_ip_bits
            ip_tensor = torch.tensor([int(bit) for bit in ip_bits])
        else:
            ip_tensor = torch.full((64,), -1)

        return ip_tensor

    def _extract_port(self):
        if self.packet.haslayer(scapy.IP):
            sport = None
            dport = None
            if self.packet.haslayer(scapy.TCP):
                sport = self.packet[scapy.TCP].sport
                dport = self.packet[scapy.TCP].dport
            elif self.packet.haslayer(scapy.UDP):
                sport = self.packet[scapy.UDP].sport
                dport = self.packet[scapy.UDP].dport
            else:
                sport = 1055
                dport = 1055
            
            sport_bits = format(sport, f'0{MAX_BITS_PORT}b')
            dport_bits = format(dport, f'0{MAX_BITS_PORT}b')
            port_bits = sport_bits + dport_bits
            port_tensor = torch.tensor([int(bit) for bit in port_bits])
        else:
            port_tensor = torch.full((32,), -1)

        return port_tensor

    def _extract_packet_size(self, get_integer=False):
        packet_size = len(self.packet)
        if get_integer:
            min_size = 64  # minimum frame size
            max_size = 1518  # maximum frame size

            normalized_size = (packet_size - min_size) / (max_size - min_size)
            packet_size_tensor = torch.tensor([normalized_size])
        else:
            packet_size_bits = format(packet_size, f'0{MAX_BITS_SIZE}b')
            packet_size_tensor = torch.tensor([int(bit) for bit in packet_size_bits])

        return packet_size_tensor

    def get_bit_representation(self, packet, prev_packet):
        """
        Extracts binary representations of various packet attributes from a given packet.
        Args:
            packet (scapy.Packet)     : The packet object to extract attributes from.
            prev_packet (scapy.Packet): The packet object to extract attributes from.
        Returns:
            torch.Tensor or None: A tensor containing the binary representations of extracted attributes, or None if extraction fails.
        """
        self.packet = packet
        self.prev_packet = prev_packet
        try:
            timestamp_tensor = self._extract_timestamp()
            mac_tensor = self._extract_mac_address()
            ip_tensor = self._extract_ip_address()
            port_tensor = self._extract_port()
            packet_size_tensor = self._extract_packet_size()

            packet_tensor = torch.cat((timestamp_tensor, mac_tensor, ip_tensor, port_tensor, packet_size_tensor))

            return packet_tensor

        except Exception as e:
            print(f"Exception occured: {e}")
            return None
    
    def get_int_representation(self, packet, prev_packet):
        """
        Extracts integer representations of various packet attributes from a given packet.
        Args:
            packet (scapy.Packet)     : The packet object to extract attributes from.
            prev_packet (scapy.Packet): The packet object to extract attributes from.
        Returns:
            torch.Tensor or None: A tensor containing the integer representations of extracted attributes, or None if extraction fails.
        """
        self.packet = packet
        self.prev_packet = prev_packet
        try:
            timestamp_tensor = self._extract_timestamp(get_integer=True)
            mac_tensor = self._extract_mac_address()
            ip_tensor = self._extract_ip_address()
            port_tensor = self._extract_port()
            packet_size_tensor = self._extract_packet_size(get_integer=True)

            packet_tensor = torch.cat((timestamp_tensor, mac_tensor, ip_tensor, port_tensor, packet_size_tensor))

            return packet_tensor

        except Exception as e:
            print(f"Exception occured: {e}")
            return None

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