import numpy as np
import scapy.all as scapy
import torch
from constants import MAX_BITS_PORT
from constants import MAX_BITS_SIZE


class FeatureRepresentation:
    def __init__(self):
        self.packet = None
        self.prev_packet = None

    def _extract_iat(self, get_integer=False):
        current_time = float(self.packet.time)
        prev_time = float(self.prev_packet.time)
        int_diff = current_time - prev_time

        # Define your min and max IAT values
        min_iat = 0.00001
        max_iat = 1.0  # Adjust this value based on your specific context

        if get_integer:
            iat_tensor = torch.tensor([int_diff])
            iat_tensor = (iat_tensor - min_iat) / (max_iat - min_iat)
        else:
            diff = int(max(int_diff * 1000000, 0))
            iat_bits = bin(diff)[2:]

            if len(iat_bits) < 32:
                padding = 32 - len(iat_bits)
                iat_bits = iat_bits + "0" * padding
                iat_tensor = torch.tensor([int(bit) for bit in iat_bits])

        return iat_tensor

    def _extract_mac_address(self):
        if self.packet.haslayer(scapy.Ether):
            src_mac = self.packet[scapy.Ether].src.split(":")
            src_mac_bits = "".join(format(int(digit, 16), "08b") for digit in src_mac)
            dst_mac = self.packet[scapy.Ether].dst.split(":")
            dst_mac_bits = "".join(format(int(digit, 16), "08b") for digit in dst_mac)
            mac_bits = src_mac_bits + dst_mac_bits
            mac_tensor = torch.tensor([int(bit) for bit in mac_bits])
        else:
            mac_tensor = torch.full((96,), 0)

        return mac_tensor

    def _extract_ip_address(self):
        if self.packet.haslayer(scapy.IP):
            src_ip = self.packet[scapy.IP].src.split(".")
            src_ip_bits = "".join(format(int(octet), "08b") for octet in src_ip)
            dst_ip = self.packet[scapy.IP].dst.split(".")
            dst_ip_bits = "".join(format(int(octet), "08b") for octet in dst_ip)
            ip_bits = src_ip_bits + dst_ip_bits
            ip_tensor = torch.tensor([int(bit) for bit in ip_bits])
        else:
            ip_tensor = torch.full((64,), 0)

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

            sport_bits = format(sport, f"0{MAX_BITS_PORT}b")
            dport_bits = format(dport, f"0{MAX_BITS_PORT}b")
            port_bits = sport_bits + dport_bits
            port_tensor = torch.tensor([int(bit) for bit in port_bits])
        else:
            port_tensor = torch.full((32,), 0)

        return port_tensor

    def _extract_packet_size(self, get_integer=False):
        packet_size = len(self.packet)
        if get_integer:
            min_size = 64  # minimum frame size
            max_size = 1518  # maximum frame size

            normalized_size = (packet_size - min_size) / (max_size - min_size)
            packet_size_tensor = torch.tensor([normalized_size])
        else:
            packet_size_bits = format(packet_size, f"0{MAX_BITS_SIZE}b")
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
            iat_tensor = self._extract_iat()
            mac_tensor = self._extract_mac_address()
            ip_tensor = self._extract_ip_address()
            port_tensor = self._extract_port()
            packet_size_tensor = self._extract_packet_size()

            packet_tensor = torch.cat(
                (
                    iat_tensor,
                    mac_tensor,
                    ip_tensor,
                    port_tensor,
                    packet_size_tensor,
                )
            )

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
            iat_tensor = self._extract_iat(get_integer=True)
            self._extract_mac_address()
            self._extract_ip_address()
            self._extract_port()
            packet_size_tensor = self._extract_packet_size(get_integer=True)

            packet_tensor = torch.cat(
                (
                    iat_tensor,
                    # mac_tensor,
                    # ip_tensor,
                    # port_tensor,
                    packet_size_tensor,
                )
            )

            return packet_tensor

        except Exception as e:
            print(f"Exception occured: {e}")
            return None


def packet_parser(packet):
    IPtype = np.nan
    timestamp = packet.time
    framelen = len(packet)
    if packet.haslayer(scapy.IP):  # IPv4
        srcIP = packet[scapy.IP].src
        dstIP = packet[scapy.IP].dst
        IPtype = 0
    elif packet.haslayer(scapy.IPv6):  # ipv6
        srcIP = packet[scapy.IPv6].src
        dstIP = packet[scapy.IPv6].dst
        IPtype = 1
    else:
        srcIP = ""
        dstIP = ""

    if packet.haslayer(scapy.TCP):
        srcproto = str(packet[scapy.TCP].sport)
        dstproto = str(packet[scapy.TCP].dport)
    elif packet.haslayer(scapy.UDP):
        srcproto = str(packet[scapy.UDP].sport)
        dstproto = str(packet[scapy.UDP].dport)
    else:
        srcproto = ""
        dstproto = ""

    srcMAC = packet.src
    dstMAC = packet.dst
    if srcproto == "":  # it's a L2/L1 level protocol
        if packet.haslayer(scapy.ARP):  # is ARP
            srcproto = "arp"
            dstproto = "arp"
            srcIP = packet[scapy.ARP].psrc  # src IP (ARP)
            dstIP = packet[scapy.ARP].pdst  # dst IP (ARP)
            IPtype = 0
        elif packet.haslayer(scapy.ICMP):  # is ICMP
            srcproto = "icmp"
            dstproto = "icmp"
            IPtype = 0
        elif srcIP + srcproto + dstIP + dstproto == "":  # some other protocol
            srcIP = packet.src  # src MAC
            dstIP = packet.dst  # dst MAC

    return IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, framelen, timestamp
