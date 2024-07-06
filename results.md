Threshold for the Anomaly Detector: 0.2661924958229065!!!\
Average reconstruction error for weekday_100k_05_10 is: 0.3605623245239258\
Average reconstruction error for weekday_06_10 is: 0.36467239260673523\
Average reconstruction error for ACK_Flooding_Smart_Clock_1 is: 0.10133973509073257\
Average reconstruction error for ARP_Spoofing_Google-Nest-Mini_1 is: 0.26101186871528625\
Average reconstruction error for Port_Scanning_SmartTV is: 0.19003543257713318\
Average reconstruction error for Service_Detection_Smartphone_1 is: 0.17429673671722412\
Average reconstruction error for SYN_Flooding_SmartTV is: 0.1412145048379898\
Average reconstruction error for UDP_Flooding_Lenovo_Bulb_1 is: 0.1187320351600647\
Precision: 0.9958076448828607\
Recall: 0.9955621301775148\
F1 score: 0.9956848723955123

## After modifying the threshold and everything to -ve
Threshold for the Anomaly Detector: -0.2661924958229065!!!\
Average anomaly score for weekday_100k.pcap is: -0.3605623245239258\
Average anomaly score for weekday_06.pcap is: -0.36467239260673523\
Average anomaly score for ACK_Flooding_Smart_Clock_1.pcap is: -0.10133973509073257\
Average anomaly score for ARP_Spoofing_Google-Nest-Mini_1.pcap is: -0.26101186871528625\
Average anomaly score for Port_Scanning_SmartTV.pcap is: -0.19003543257713318\
Average anomaly score for Service_Detection_Smartphone_1.pcap is: -0.17429673671722412\
Average anomaly score for SYN_Flooding_SmartTV.pcap is: -0.1412145048379898\
Average anomaly score for UDP_Flooding_Lenovo_Bulb_1.pcap is: -0.1187320351600647\
Precision: 0.9958076448828607\
Recall: 0.9955621301775148\
F1 score: 0.9956848723955123

## Adversarial attack
Total time: 18065314.904064, Adv: 1981418.758143999
Pcap file: Port_Scanning_SmartTV
Mean RE for malicious packets: 0.19003543257713318
Mean RE for adversarial malicious packets: 0.28416725993156433
Evasion Rate: 0.88

Total time: 13382684.299775993, Adv: 1103122.1027839999
Pcap file: Service_Detection_Smartphone_1
Mean RE for malicious packets: 0.17429673671722412
Mean RE for adversarial malicious packets: 0.2689577341079712
Evasion Rate: 0.6111111111111112

Total time: 12921467.416576, Adv: 1445283.7867520007
Pcap file: ARP_Spoofing_Google-Nest-Mini_1
Mean RE for malicious packets: 0.26101186871528625
Mean RE for adversarial malicious packets: 0.3394051790237427
Evasion Rate: 0.9444444444444444

##### Setting: All Real Numbers Image:

All the fields, namely size, time, ips, macs, ports are converted to integer numbers, then to a range between 0 to 1.

##### Setting: Loopback PGD Training
```python main.py --model-name AutoencoderRaw --num-epochs 10 --loss MSELoss --batch-size 1024 --eval --threshold 7765.834472```

Loaded the model in eval mode!!!
Threshold for the Anomaly Detector: 7765.8345!!! \

Processing ../data/benign/weekday_06.pcap!!! \
Average anomaly score for weekday_06.pcap is: 1275953792.0 \
Time taken: 1.6390 seconds

Processing ../data/malicious/Service_Detection_Smartphone_1.pcap!!! \
Average anomaly score for Service_Detection_Smartphone_1.pcap is: 109706.375 \
Time taken: 4.3141 seconds

Processing ../data/malicious/UDP_Flooding_Lenovo_Bulb_1.pcap!!! \
Average anomaly score for UDP_Flooding_Lenovo_Bulb_1.pcap is: 299097760.0 \
Time taken: 0.9191 seconds

Processing ../data/malicious/ARP_Spoofing_Google-Nest-Mini_1.pcap!!! \
Average anomaly score for ARP_Spoofing_Google-Nest-Mini_1.pcap is: 12888678400.0 \
Time taken: 0.4677 seconds

Processing ../data/malicious/Port_Scanning_SmartTV.pcap!!! \
Average anomaly score for Port_Scanning_SmartTV.pcap is: 2478600192.0 \
Time taken: 3.4180 seconds

Precision: 0.9409937888198758 \
Recall: 1.0 \
F1 score: 0.9696

Total time: 34.343587160110474, Adv: 370.64359180629253 \
Pcap file: Port_Scanning_SmartTV \
Mean RE for malicious packets: 14755971072.0 \
Mean RE for adversarial malicious packets: 585.290771484375 \
Evasion Rate: 1.0

Total time: 149.5606439113617, Adv: 421.5606459900737 \
Pcap file: ARP_Spoofing_Google-Nest-Mini_1 \
Mean RE for malicious packets: 22593273856.0 \
Mean RE for adversarial malicious packets: 1146.8372802734375 \
Evasion Rate: 1.0

Total time: 30.596760988235474, Adv: 235.99676393717527 \
Pcap file: Service_Detection_Smartphone_1 \
Mean RE for malicious packets: 358917.125 \
Mean RE for adversarial malicious packets: 746.3072509765625 \
Evasion Rate: 1.0

Total time: 137.42234086990356, Adv: 24078.422697141767 \
Pcap file: UDP_Flooding_Lenovo_Bulb_1 \
Mean RE for malicious packets: 10504731.0 \
Mean RE for adversarial malicious packets: 373.85540771484375 \
Evasion Rate: 1.0

Total time: 177.8861949443817, Adv: 47252.18689412624 \
Pcap file: ACK_Flooding_Smart_Clock_1 \
Mean RE for malicious packets: 360871008.0 \
Mean RE for adversarial malicious packets: 1417.90966796875 \
Evasion Rate: 1.0

### Analysis:
#### Why image based method didn't work?
If target model and surrogate model make decision by looking at different set of features, then evading surrogate model doesn't guarantee evding the target model. Hence, the image based method didn't work where the surrogate model was making decision based on the structural pattern of different types of image while the target model (Kitsune) makes results by looking at the temporal pattern of different packet.

#### What to do?
The goal of the surrogate model should be:\
1. Trained with features that are temporal
2. Give similar performance to the target model
3. Easily to perturb

Hence: Loopback PGD

#### Why perturbing size is difficult?
If size is modified, we need to modify:
1. Checksum of the packet
2. Sequence number of the packet
3. Sequence number of all the packets in the same connection.

DOUBT: Does attack traffic start with a connection like a normal TCP three-way handshake or it's just bunch of random packets? Does this requires updating sequence numbers in all the subsequent packets?

#### Why perturbing other fields except the ones used by Kitsune won't provide any additional advantages?
Since, Kitsune's feature vector depends on size and time only grouped by MACs, IPs, and ports, perturbing these entities might bring a change in the decision of Kitsune. Perturbing random feilds, e.g. TTL, even it's independednt of all the fields and easier to perturb, won't yield good results.

#### Why we shouldn't take Loopback PGD and make an image version out of it?
Even though it looks fancy, it's the same thing as doing it individually. Even in the images, the perturbation can't be applied at a time since the feature of the current packet depends on the previous ones.

#### What next?
It might be interesting to accomodate features in the feature extraction process which are harder to perturb. If the target model depends on the fields that are harder to perturb or impossible to perturb, then the system will be secure.

OR

Find fields whose perturbation affects the attack functionality of the network tarffic. If so, then even if the attacker able to perturb the fields, it won't be able to replay that in the real life.

Questions and Pointers:
1. Do I reply the response packets as well or just the request packets?
2. Size changes in UDP connection doesn't requries change in sequence numbers of subsequent packets. Same with ACK, UDP, and SYN flooding.
3. For a TCP connection, it's crucial to change the sequence numbers of all the subsequent packets, if you change the sequence number of one packet.
4. For PS, SD, and ARP Spoofing, it's crucial to investigate which technique is used to get the results, then the sequence number change will depend on that. Obtain the technique used from this link.
https://nmap.org/book/man-port-scanning-techniques.html