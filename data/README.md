# Data

Raw files live in `data/raw/` (git-ignored). `data/registry.json` lists every
file with its URL and pinned SHA-256; `scripts/fetch.py` refuses any download
whose hash differs. `data/manifest.json` records what was fetched and when.

```
python scripts/fetch.py kdd99 kdd99_test kdd99_hf_train nslkdd
python -m experiments.e0_sources   # checks third-party copies against verified originals
```

## Provenance

| Dataset | Where the bytes come from | Why we trust them |
|---|---|---|
| `kdd99` (kddcup.data.gz, 4,898,431 rows) | scikit-learn's figshare mirror | SHA-256 equals the hash that **both** scikit-learn and TensorFlow Datasets pin for the original UCI file |
| `kdd99_test` (labelled "corrected" test, 311,029 rows) | Hugging Face `akahana/kdd-cup-1999` @ 44542fc, parquet | Not byte-verifiable (converted). The same upload's training split equals the verified `kdd99` row for row, in order (E0-src) |
| `nslkdd` (KDDTrain+ 125,973, KDDTest+ 22,544) | original NSL-KDD zip via GitHub mirror `Jehuty4949/NSL_KDD` @ 27bbbdf | Every row occurs in verified KDD99 (E0-src) |

Not used, and why:
- UCI dataset page 130 zip: contains link stubs, not data. The old
  `kdd.ics.uci.edu` host returns 403 for data files.
- CIC's official NSL-KDD download sits behind a registration form.
- IEEE DataPort KDD-CUP99: needs a login.

Archives are never extracted to disk; loaders read named members directly.
