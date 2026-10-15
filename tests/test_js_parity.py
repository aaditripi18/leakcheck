"""The browser engine (site/engine, JavaScript) must report what the Python package reports.

Each case writes a train/test pair to CSV, scans it with Python's leakcheck
(from the files, as the CLI does) and with `node site/engine/cli.mjs`, and
compares the findings. Counts must match exactly; single-feature scores (a
port of scikit-learn's tree) within 0.005. Random baselines and sampling are
excluded: both engines check every test row here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leakcheck import Split, scan
from leakcheck.synth import make_split

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "site" / "engine" / "cli.mjs"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

EXACT = {  # code -> detail keys that must match exactly
    "L1.within_train": ["n_rows", "n_unique"], "L1.within_test": ["n_rows", "n_unique"],
    "L1.cross": ["n_hit"], "L2.near": ["n_near", "n_checked"], "L2.near_text": ["n_near", "n_checked"],
    "L3.conflict": ["n_groups", "n_rows"], "L3.cross_contradict": ["n_rows"],
    "L6.group": ["n_hit", "shared_groups", "test_groups"],
}


def run_js(train: Path, test: Path, label: str, **kw) -> dict:
    cmd = [NODE, str(CLI), "scan", "--train", str(train), "--test", str(test), "--label", label, "--sample", "1000000000"]
    for k in ("time", "group", "text"):
        if kw.get(k):
            cmd += [f"--{k}", kw[k]]
    if kw.get("ignore"):
        cmd += ["--ignore", ",".join(kw["ignore"])]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def run_py(train: Path, test: Path, label: str, **kw) -> dict:
    split = Split.from_files(train, test, label, time=kw.get("time"), group=kw.get("group"),
                             text=kw.get("text"), ignore=kw.get("ignore", []))
    return json.loads(scan(split, sample=10**9).to_json())


def compare(py: dict, js: dict) -> list[str]:
    bad = []
    pf = {f["code"]: f for f in py["findings"]}
    jf = {f["code"]: f for f in js["findings"]}
    if set(pf) != set(jf):
        bad.append(f"finding codes differ: python {sorted(pf)} vs js {sorted(jf)}")
    if sorted(py["meta"]["skipped"]) != sorted(js["meta"]["skipped"]):
        bad.append(f"skipped differ: {py['meta']['skipped']} vs {js['meta']['skipped']}")
    for code in set(pf) & set(jf):
        p, j = pf[code], jf[code]
        tol = 0.005 if code in ("L5.shortcut", "L4.time_shortcut") else 1e-9
        if abs(p["value"] - j["value"]) > tol:
            bad.append(f"{code}: value {p['value']} vs {j['value']}")
        for k in EXACT.get(code, []):
            if p["details"].get(k) != j["details"].get(k):
                bad.append(f"{code}.{k}: {p['details'].get(k)} vs {j['details'].get(k)}")
        if code == "L1.cross":
            pc = {k: round(v["rate"], 12) for k, v in p["details"]["per_class"].items()}
            jc = {k: round(v["rate"], 12) for k, v in j["details"]["per_class"].items()}
            if pc != jc:
                bad.append(f"L1.cross per_class {pc} vs {jc}")
            if p["examples"] != j["examples"]:
                bad.append(f"L1.cross examples {p['examples']} vs {j['examples']}")
        if code == "L2.near":
            for e, v in p["details"]["curve"].items():
                if abs(v - j["details"]["curve"].get(e, -1)) > 1e-9:
                    bad.append(f"L2.near curve[{e}] {v} vs {j['details']['curve'].get(e)}")
        if code == "L2.near_text" and abs(p["details"]["identical_rate"] - j["details"]["identical_rate"]) > 1e-9:
            bad.append("L2.near_text identical_rate differs")
        if code == "L5.shortcut":
            ps = {t["feature"]: t["balanced_acc"] for t in p["details"]["top"]}
            js_ = {t["feature"]: t["balanced_acc"] for t in j["details"]["top"]}
            for feat in set(ps) & set(js_):
                if abs(ps[feat] - js_[feat]) > tol:
                    bad.append(f"L5.shortcut {feat}: {ps[feat]} vs {js_[feat]}")
            if set(p["details"]["flagged"]) != set(j["details"]["flagged"]):
                bad.append(f"L5 flagged {p['details']['flagged']} vs {j['details']['flagged']}")
        if code == "L5.identifier" and p["details"]["columns"] != j["details"]["columns"]:
            bad.append(f"L5.identifier {p['details']['columns']} vs {j['details']['columns']}")
        if code == "L4.after_test_start":
            for k in ("after_test_median", "after_test_end"):
                if abs(p["details"][k] - j["details"][k]) > 1e-9:
                    bad.append(f"L4 {k} differs")
        if code not in ("L5.shortcut", "L4.time_shortcut") and p["severity"] != j["severity"]:
            bad.append(f"{code}: severity {p['severity']} vs {j['severity']}")
    return bad


def check(tmp_path, train: pd.DataFrame, test: pd.DataFrame, label: str, **kw):
    tr, te = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    bad = compare(run_py(tr, te, label, **kw), run_js(tr, te, label, **kw))
    assert not bad, "\n".join(bad)


# --- synthetic -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_leaks(tmp_path, seed):
    train, test, _ = make_split(cross_dup=0.1, near_dup=0.05, train_dup=0.05, conflict=0.02,
                                shortcut=True, n_classes=2 + seed % 2, seed=seed)
    check(tmp_path, train, test, "label")


def test_clean_split(tmp_path):
    train, test, _ = make_split(seed=9)
    check(tmp_path, train, test, "label")


def _mixed(n, rng, offset=0):
    proto = rng.choice(["tcp", "udp", "icmp"], n)
    x = rng.normal(size=n).round(4)
    df = pd.DataFrame({
        "Flow ID": [f"10.0.0.{i % 97}-{i + offset}" for i in range(n)],
        "proto": proto,
        "bytes": rng.integers(0, 5000, n),
        "rate": np.where(rng.random(n) < 0.05, np.nan, x),
        "ratio": np.where(rng.random(n) < 0.01, np.inf, rng.random(n).round(3)),
        "flag": rng.random(n) < 0.3,
        "msg": [f"user {i % 40} sent message number {i % 13} to host {i % 7}" for i in range(n)],
        "ts": pd.date_range("2024-01-01", periods=n, freq="37min").strftime("%Y-%m-%d %H:%M") if offset == 0
        else pd.date_range("2024-01-20", periods=n, freq="41min").strftime("%Y-%m-%d %H:%M"),
        "host": [f"h{i % 23}" for i in range(n)],
    })
    df["label"] = np.where((df["proto"] == "udp") & (df["bytes"] > 2500), "attack", "benign")
    return df


def test_mixed_types_text_time_group(tmp_path):
    rng = np.random.default_rng(5)
    train = _mixed(1500, rng)
    test = _mixed(600, rng, offset=10_000)
    test.iloc[:60, 1:] = train.iloc[:60, 1:].to_numpy()  # copies, but keep a unique Flow ID
    check(tmp_path, train, test, "label", time="ts", group="host", text="msg")


def test_ignore_columns(tmp_path):
    rng = np.random.default_rng(6)
    train, test = _mixed(800, rng), _mixed(300, rng, offset=5000)
    check(tmp_path, train, test, "label", ignore=["Flow ID", "msg"])


# --- real datasets (skipped when the data is not downloaded) -------------------------


def _real(loader):
    try:
        return loader()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"dataset not available: {e}")


def test_uci_phishing(tmp_path):
    from experiments.e1_audit import prep_uci_phishing
    (_, s, _), = _real(prep_uci_phishing).splits
    check(tmp_path, s.train, s.test, "Result")


def test_sms_spam_text(tmp_path):
    from experiments.e1_audit import prep_sms
    (_, s, _), = _real(prep_sms).splits
    check(tmp_path, s.train, s.test, "label", text="message")


def test_unsw_sample(tmp_path):
    from experiments.e1_audit import prep_unsw
    (_, s, _), = _real(prep_unsw).splits
    check(tmp_path, s.train.iloc[::12], s.test.iloc[::12], "label", ignore=["id", "attack_cat"])


def test_nslkdd_sample(tmp_path):
    from experiments.e1_audit import prep_nslkdd
    (_, s, _), = _real(prep_nslkdd).splits
    check(tmp_path, s.train.iloc[::8], s.test.iloc[::3], "is_attack", ignore=["difficulty"])


def test_clean_output_verified(tmp_path):
    train, test, t = make_split(cross_dup=0.2, seed=11)
    tr, te, out = tmp_path / "tr.csv", tmp_path / "te.csv", tmp_path / "clean.csv"
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    res = subprocess.run([NODE, str(CLI), "clean", "--train", str(tr), "--test", str(te), "--label", "label",
                          "--out", str(out)], capture_output=True, text=True, check=True)
    r = json.loads(res.stdout)
    assert r["removed"] == t["cross_dup_rows"] and r["verified"]
    py = scan(Split.from_files(tr, out, "label"), checks=("L1",))
    assert py.get("L1.cross").value == 0  # Python agrees the cleaned file has no copies left
