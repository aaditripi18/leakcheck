# Data

Raw files live in `data/raw/` (git-ignored). `data/registry.json` lists every
file with its URL and pinned SHA-256; `scripts/fetch.py` refuses any download
whose hash differs. `data/manifest.json` records what was fetched and when.

```
python scripts/fetch.py            # all datasets (resumes interrupted downloads)
python -m experiments.e0_sources          # KDD99 / NSL-KDD copies vs verified originals
python -m experiments.e0_sources_week4    # the other six datasets vs independent references
```

## Provenance

| Dataset | Where the bytes come from | Why we trust them |
|---|---|---|
| `kdd99` (kddcup.data.gz, 4,898,431 rows) | scikit-learn's figshare mirror | SHA-256 equals the hash that **both** scikit-learn and TensorFlow Datasets pin for the original UCI file |
| `kdd99_test` (labelled "corrected" test, 311,029 rows) | Hugging Face `akahana/kdd-cup-1999` @ 44542fc, parquet | Not byte-verifiable (converted). The same upload's training split equals the verified `kdd99` row for row, in order (E0-src) |
| `nslkdd` (KDDTrain+ 125,973, KDDTest+ 22,544) | original NSL-KDD zip via GitHub mirror `Jehuty4949/NSL_KDD` @ 27bbbdf | Every row occurs in verified KDD99 (E0-src) |
| `unsw_nb15` (training-set 175,341, testing-set 82,332) | Hugging Face `Mireu-Lab/UNSW-NB15` @ 3eb02c4 (official files are in a UNSW SharePoint folder) | The 175,341-row file is byte-identical to an independent upload under the official name (`Mouwiya/UNSW-NB15`); Mireu-Lab had swapped the names, so the files are stored under the official names. All 20 per-category counts match the published table |
| `cicids2017` (MachineLearningCSV.zip, 2,830,743 flows) | Hugging Face `bencorn/CICIDS2017` @ 811c007 (official download is behind a registration form; no official checksum exists) | All 8 CSVs inside are byte-identical to the independent upload `c01dsnap/CIC-IDS2017` @ 4e9edfa; all 15 label counts match Ahmim et al. 2018, Table II |
| `ember2018` (1M PE files, raw JSON features) | official ember.elastic.co | SHA-256 published in the elastic/ember README |
| `uci_phishing`, `phiusiil`, `sms_spam` | official UCI downloads | Row counts match the UCI API; SMS ham/spam counts match the shipped readme |

Not used, and why:
- UCI dataset page 130 zip: contains link stubs, not data. The old
  `kdd.ics.uci.edu` host returns 403 for data files.
- CIC's official NSL-KDD download sits behind a registration form.
- IEEE DataPort KDD-CUP99: needs a login.

Downloads resume after dropped connections (two CDNs truncated large files
silently during setup; the hash check caught both). Archives are never extracted to disk; loaders read named members directly.

## Sources checked and rejected (2026-10-16)

Looked for the raw UNSW-NB15 flow records (UNSW-NB15_1..4.csv, 2,540,044 rows
with capture timestamps), for a time-ordered split in which every attack
category occurs before and after the cut. Neither mirror below is the raw data,
so neither is used:

- `rdpahalavan/UNSW-NB15`, Network-Flows/UNSW_Flow.parquet (commit 25ecbc46f36e,
  SHA-256 8129db1c...): 2,059,415 flows, not 2,540,044; Generic 25,378, not the
  published 215,481. Its README says the flows were re-extracted and matched to
  the PCAP files, so it is a derivative.
- `bastyje/UNSW-NB15` train.csv / test.csv: every feature min-max scaled to
  [0, 1] and addresses encoded; not the published records.

The raw files are distributed from the UNSW project page through SharePoint,
which needs a person to download them.
