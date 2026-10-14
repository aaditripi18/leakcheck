"""E0-src (week 4): check each week-4 dataset against an independent reference.

  python -m experiments.e0_sources_week4 [unsw_nb15 cicids2017 uci_phishing phiusiil sms_spam ember2018]

References (all retrieved 2026-10-14/15):
  UNSW-NB15    totals: official UNSW page (175,341 / 82,332); per category:
               Moustafa & Slay 2016 as reproduced in J. Big Data 2024
               (doi:10.1186/s40537-024-00994-7, Table 2) and Springer
               Cybersecurity 2025 (doi:10.1186/s42400-025-00364-7, Table 1).
  CIC-IDS2017  per-day CSV SHA-256 of the independent upload
               c01dsnap/CIC-IDS2017 @ 4e9edfa; label counts: Ahmim et al. 2018,
               arXiv:1812.09059, Table II ("Total" column).
  UCI sets     instance counts from the UCI API; SMS ham/spam from the readme
               shipped in the official zip.
  EMBER 2018   tarball SHA-256 from the elastic/ember README (enforced by fetch);
               sample counts from the EMBER paper/README: 800k train
               (300k malicious, 300k benign, 200k unlabelled), 200k test (100k/100k).
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile

from experiments import data
from experiments.ledger import record

UNSW = {
    "train": {"Normal": 56000, "Generic": 40000, "Exploits": 33393, "Fuzzers": 18184, "DoS": 12264,
              "Reconnaissance": 10491, "Analysis": 2000, "Backdoor": 1746, "Shellcode": 1133, "Worms": 130},
    "test": {"Normal": 37000, "Generic": 18871, "Exploits": 11132, "Fuzzers": 6062, "DoS": 4089,
             "Reconnaissance": 3496, "Analysis": 677, "Backdoor": 583, "Shellcode": 378, "Worms": 44},
}
CIC_C01DSNAP = {
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000",
    "Monday-WorkingHours.pcap_ISCX.csv": "852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851",
    "Wednesday-workingHours.pcap_ISCX.csv": "893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a",
}
# Ahmim et al. 2018 Table II writes "Web Attack- Brute Force"; the CSVs have a
# non-ASCII dash there, which load_cicids2017 turns into a space.
CIC_LABELS = {
    "BENIGN": 2273097, "DDoS": 128027, "DoS slowloris": 5796, "DoS Slowhttptest": 5499, "DoS Hulk": 231073,
    "DoS GoldenEye": 10293, "Heartbleed": 11, "PortScan": 158930, "Bot": 1966, "FTP-Patator": 7938,
    "SSH-Patator": 5897, "Web Attack Brute Force": 1507, "Web Attack XSS": 652, "Web Attack Sql Injection": 21,
    "Infiltration": 36,
}
UCI_COUNTS = {"uci_phishing": 11055, "phiusiil": 235795, "sms_spam": 5574}
SMS_CLASSES = {"ham": 4827, "spam": 747}
EMBER = {"train": {"1": 300000, "0": 300000, "-1": 200000}, "test": {"1": 100000, "0": 100000}}


def cmp(ours: dict, ref: dict) -> dict:
    keys = sorted(set(ours) | set(ref))
    diff = {k: ours.get(k, 0) - ref.get(k, 0) for k in keys if ours.get(k, 0) != ref.get(k, 0)}
    return {"match": not diff, "diff": diff}


def check_unsw() -> dict:
    return {p: cmp(data.load_unsw(p)["attack_cat"].value_counts().to_dict(), UNSW[p]) for p in ("train", "test")}


def check_cic() -> dict:
    (path,) = data.verify_local("cicids2017")
    hashes = {}
    with zipfile.ZipFile(path) as zf:
        for f, m in data.cic_members(zf).items():
            h = hashlib.sha256()
            with zf.open(m) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            hashes[f] = h.hexdigest() == CIC_C01DSNAP[f]
    labels = data.load_cicids2017()["Label"]  # the loader normalises the Web Attack dash
    return {"csv_equal_to_independent_upload": hashes, "all_csv_equal": all(hashes.values()),
            "rows": int(len(labels)), "labels": cmp(labels.value_counts().to_dict(), CIC_LABELS)}


def check_uci(name: str) -> dict:
    df = {"uci_phishing": data.load_uci_phishing, "phiusiil": data.load_phiusiil, "sms_spam": data.load_sms_spam}[name]()
    out = {"rows": len(df), "rows_match_uci": len(df) == UCI_COUNTS[name]}
    if name == "sms_spam":
        out["classes"] = cmp(df["label"].value_counts().to_dict(), SMS_CLASSES)
    return out


def check_ember() -> dict:
    from experiments.ember import load_ember

    df = load_ember()
    out = {}
    for split, ref in EMBER.items():
        counts = df.loc[df["split"] == split, "label"].astype(str).value_counts().to_dict()
        out[split] = cmp(counts, ref)
    out["appeared_range"] = {s: [str(df.loc[df["split"] == s, "appeared"].min()), str(df.loc[df["split"] == s, "appeared"].max())]
                             for s in ("train", "test")}
    out["unique_sha256"] = int(df["sha256"].nunique())
    return out


CHECKS = {"unsw_nb15": check_unsw, "cicids2017": check_cic, "uci_phishing": lambda: check_uci("uci_phishing"),
          "phiusiil": lambda: check_uci("phiusiil"), "sms_spam": lambda: check_uci("sms_spam"), "ember2018": check_ember}


def main(names: list[str]) -> None:
    for name in names or list(CHECKS):
        res = CHECKS[name]()
        record("e0_sources_week4:" + name, {}, res, {name: data.dataset_hashes(name)})
        print(name, json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
