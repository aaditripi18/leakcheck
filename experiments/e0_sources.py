"""E0-src: check the third-party copies against verified originals.

1. The Hugging Face KDD99 upload's training split must equal the original
   kddcup.data.gz (hash-verified via scikit-learn and TFDS) row for row, as a
   multiset. If the conversion is exact for 4.9M training rows, we accept the
   same upload's test split.
2. Every NSL-KDD row (features + label) must occur in KDD99: KDDTrain+ in the
   full training set, KDDTest+ in the labelled test set, since Tavallaee et al.
   built NSL-KDD by sampling distinct KDD records.

  python -m experiments.e0_sources
"""

from __future__ import annotations

import numpy as np

from experiments.data import KDD_FEATURES, dataset_hashes, load_kdd99, load_nslkdd
from experiments.ledger import record
from leakcheck.split import canonicalize, row_hashes

RECORD = KDD_FEATURES + ["label"]


def record_hashes(df) -> np.ndarray:
    return row_hashes(canonicalize(df[RECORD]))


def main() -> dict:
    full = record_hashes(load_kdd99("full"))
    hf = record_hashes(load_kdd99("hf_train"))
    same_multiset = len(full) == len(hf) and bool(np.array_equal(np.sort(full), np.sort(hf)))
    same_order = len(full) == len(hf) and bool(np.array_equal(full, hf))

    test = record_hashes(load_kdd99("test"))
    nsl_tr = record_hashes(load_nslkdd("train"))
    nsl_te = record_hashes(load_nslkdd("test"))
    tr_in = np.isin(nsl_tr, full)
    te_in = np.isin(nsl_te, test)
    te_in_any = np.isin(nsl_te, np.concatenate([full, test]))

    results = {
        "hf_train_rows": len(hf),
        "official_train_rows": len(full),
        "hf_train_equals_official_multiset": same_multiset,
        "hf_train_equals_official_order": same_order,
        "nsl_train_rows": len(nsl_tr),
        "nsl_train_in_kdd_train": int(tr_in.sum()),
        "nsl_test_rows": len(nsl_te),
        "nsl_test_in_kdd_test": int(te_in.sum()),
        "nsl_test_in_kdd_train_or_test": int(te_in_any.sum()),
    }
    datasets = {n: dataset_hashes(n) for n in ("kdd99", "kdd99_test", "kdd99_hf_train", "nslkdd")}
    record("e0_sources", {}, results, datasets)
    for k, v in results.items():
        print(f"{k:40} {v}")
    return results


if __name__ == "__main__":
    main()
