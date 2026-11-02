"""Per-class row counts published by the dataset authors, used to validate the
phase 3 downloads (PLAN.md, phase 3). Copied from the source named on each table;
a download is accepted only if its own counts equal these."""

# Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks
# in IoT Environment", Sensors 23(13):5941 (2023), Table 3 "Number of rows for each
# attack and category", https://www.mdpi.com/1424-8220/23/13/5941 (read 2026-11-02).
CICIOT2023 = {
    "DDoS-ICMP_Flood": 7_200_504, "DoS-TCP_Flood": 2_671_445,
    "DDoS-UDP_Flood": 5_412_287, "DoS-SYN_Flood": 2_028_834,
    "DDoS-TCP_Flood": 4_497_667, "BenignTraffic": 1_098_195,
    "DDoS-PSHACK_Flood": 4_094_755, "Mirai-greeth_flood": 991_866,
    "DDoS-SYN_Flood": 4_059_190, "Mirai-udpplain": 890_576,
    "DDoS-RSTFINFlood": 4_045_285, "Mirai-greip_flood": 751_682,
    "DDoS-SynonymousIP_Flood": 3_598_138, "DDoS-ICMP_Fragmentation": 452_489,
    "DoS-UDP_Flood": 3_318_595, "MITM-ArpSpoofing": 307_593,
    "Recon-PingSweep": 2_262, "Uploading_Attack": 1_252,
    "DDoS-UDP_Fragmentation": 286_925, "DDoS-HTTP_Flood": 28_790,
    "DDoS-ACK_Fragmentation": 285_104, "DDoS-SlowLoris": 23_426,
    "DNS_Spoofing": 178_911, "DictionaryBruteForce": 13_064,
    "Recon-HostDiscovery": 134_378, "BrowserHijacking": 5_859,
    "Recon-OSScan": 98_259, "CommandInjection": 5_409,
    "Recon-PortScan": 82_284, "SqlInjection": 5_245,
    "DoS-HTTP_Flood": 71_864, "XSS": 3_846,
    "VulnerabilityScan": 37_382, "Backdoor_Malware": 3_218,
}

# CSE-CIC-IDS2018, "Processed Traffic Data for ML Algorithms" (all ten days). CIC
# publishes no per-class counts; this is the distribution tabulated in Sensors
# 22(19):7409 (2022), Table 3 "Distribution of stream records in CICIDS2018 dataset",
# https://pmc.ncbi.nlm.nih.gov/articles/PMC9572777/table/sensors-22-07409-t003 (read
# 2026-11-02). Keys use the spelling in the CSVs ("Infilteration", "Brute Force -Web");
# the table writes "Infiltration" and "Brute Force-Web". Repeated header rows inside
# the files are not records.
CSECICIDS2018 = {
    "Benign": 13_484_708, "DDOS attack-HOIC": 686_012, "DDoS attacks-LOIC-HTTP": 576_191,
    "DoS attacks-Hulk": 461_912, "Bot": 286_191, "FTP-BruteForce": 193_360, "SSH-Bruteforce": 187_589,
    "Infilteration": 161_934, "DoS attacks-SlowHTTPTest": 139_890, "DoS attacks-GoldenEye": 41_508,
    "DoS attacks-Slowloris": 10_990, "DDOS attack-LOIC-UDP": 1_730, "Brute Force -Web": 611,
    "Brute Force -XSS": 230, "SQL Injection": 87,
}
