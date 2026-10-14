"""Generate docs/RESULTS.md from results/ledger.jsonl. Never edit RESULTS.md by hand.

Uses the latest ledger line of each experiment, and prints the commit it came
from next to every table.

  python scripts/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "ledger.jsonl"
OUT = ROOT / "docs" / "RESULTS.md"


def latest() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        runs[d["experiment"]] = d
    return runs


def src(d: dict) -> str:
    g = d["git"]
    return f"_ledger: `{d['experiment']}` @ `{g['commit']}`{' (dirty)' if g['dirty'] else ''}, {d['time']}_"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def e0_sources(d: dict) -> list[str]:
    r = d["results"]
    return [
        "## E0-src: third-party copies checked against verified originals", "", src(d), "",
        "| Check | Result |", "|---|---|",
        f"| HF KDD99 train equals official kddcup.data.gz (multiset) | {r['hf_train_equals_official_multiset']} ({r['hf_train_rows']:,} rows) |",
        f"| ... and in the same order | {r['hf_train_equals_official_order']} |",
        f"| NSL-KDD train rows found in KDD99 train | {r['nsl_train_in_kdd_train']:,} / {r['nsl_train_rows']:,} |",
        f"| NSL-KDD test rows found in KDD99 test | {r['nsl_test_in_kdd_test']:,} / {r['nsl_test_rows']:,} |",
        "",
    ]


def e0_synthetic(d: dict) -> list[str]:
    r = d["results"]
    er, ci, mh = r["exact_recovery"], r["interval_coverage"], r["minhash"]
    out = ["## E0a: recovery of planted leakage", "", src(d), "",
           f"Exact recovery over {er['cells']} grid cells (5 seeds): maximum absolute error per check.", "",
           "| Check | Max abs error |", "|---|---|"]
    out += [f"| {k.removeprefix('err_')} | {v} |" for k, v in er["max_abs_error"].items()]
    out += ["", f"Sampled L2 interval coverage: {pct(ci['coverage'])} of {ci['runs']} runs "
            f"(sample {ci['sample']}, nominal 95%).", "",
            f"MinHash L2 (tau = {mh['tau']}): recall by true Jaccard; false positives {mh['false_positives']}, "
            f"every reported pair verified: {mh['all_reported_verified']}.", "",
            "| True Jaccard | Pairs | Recall |", "|---|---|---|"]
    out += [f"| {k} | {v['n']} | {pct(v['recall']) if v['recall'] is not None else 'n/a'} |"
            for k, v in mh["recall_by_jaccard"].items()]
    return out + [""]


def e0_kdd(d: dict) -> list[str]:
    r = d["results"]
    out = ["## E0b: KDD99 redundancy vs Tavallaee et al. 2009 (Tables I and II)", "", src(d), "",
           "| Set | Class | Records (ours / paper) | Distinct (ours / paper) | Difference |", "|---|---|---|---|---|"]
    for s, t in r["tavallaee_comparison"].items():
        for c, v in t.items():
            (o1, o2), (p1, p2) = v["ours"], v["paper"]
            diff = "exact" if v["diff"] == [0, 0] else f"{v['diff'][0]:+,} / {v['diff'][1]:+,}"
            out.append(f"| {s} | {c} | {o1:,} / {p1:,} | {o2:,} / {p2:,} | {diff} |")
    out += ["", "leakcheck findings on the binary attack/normal task (features only, label column excluded):", "",
            "| Finding | KDD99 (full train vs corrected test) | NSL-KDD (KDDTrain+ vs KDDTest+) |", "|---|---|---|"]
    k, n = r["kdd99_leakcheck"], r["nslkdd_leakcheck"]
    for code in ["L1.within_train", "L1.within_test", "L1.cross", "L2.near", "L3.conflict", "L3.ceiling",
                 "L3.cross_contradict", "L5.shortcut"]:
        def cell(x):
            if code not in x:
                return "not run"
            f = x[code]
            extra = ""
            if "n_hit" in f:
                extra = f" ({f['n_hit']:,} rows)"
            elif "n_rows" in f and code.startswith("L3"):
                extra = f" ({f['n_rows']:,} rows)"
            elif code == "L2.near" and f.get("vs_train_baseline") is not None:
                extra = f" (x{f['vs_train_baseline']:.2f} of train baseline)"
            return f"{pct(f['value'])}{extra} [{f['severity']}]"
        out.append(f"| {code} | {cell(k)} | {cell(n)} |")
    pc = k["L1.cross"].get("per_class", {})
    if pc:
        out += ["", "KDD99 L1.cross by class: " + ", ".join(
            f"{'attack' if c == '1' else 'normal'} {pct(v['rate'])} of {v['n']:,}" for c, v in pc.items())]
    return out + [""]


E1_ORDER = ["kdd99", "nslkdd", "unsw_nb15", "cicids2017", "ember2018", "uci_phishing", "phiusiil", "sms_spam"]


def _merged(d: dict) -> tuple[dict, dict]:
    f = {}
    for v in d["results"]["views"].values():
        f.update(v["findings"])
    return f, next(iter(d["results"]["views"].values()))["meta"]


def e1(runs: dict) -> list[str]:
    done = [n for n in E1_ORDER if f"e1_audit:{n}" in runs]
    if not done:
        return []

    def v(f, code, key="value", fmt=pct):
        if code not in f or f[code].get(key) is None:
            return "n/a"
        x = f[code][key]
        return fmt(x) if fmt else x

    out = ["## E1: leakage audit", "",
           "Binary task per dataset; official split where one exists, otherwise stratified random 80/20 (seed 0). "
           "`n/a` = not applicable (no timestamp / text column / group column).", "",
           "| Dataset | Split | Test rows | Exact copies of train | Near-copies (x train baseline) | Text near-copies "
           "| Accuracy ceiling | Contradicted | Train after test start | Best single feature | Group overlap |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    h1, h5 = [], []
    for n in done:
        d = runs[f"e1_audit:{n}"]
        f, meta = _merged(d)
        near = v(f, "L2.near")
        ratio = f.get("L2.near", {}).get("vs_train_baseline")
        if near != "n/a" and ratio is not None:
            near += f" (x{ratio:.2f})"
        top = f.get("L5.shortcut", {}).get("top", [{}])
        best = f"{top[0].get('feature', '')} {pct(f['L5.shortcut']['value'])}" if "L5.shortcut" in f else "n/a"
        split = "official" if d["results"]["split_policy"].startswith("official") else "random 80/20"
        out.append(f"| {n} | {split} | {meta['n_test']:,} | {v(f, 'L1.cross')} | {near} | {v(f, 'L2.near_text')} "
                   f"| {v(f, 'L3.ceiling')} | {v(f, 'L3.cross_contradict')} | {v(f, 'L4.after_test_start')} "
                   f"| {best} | {v(f, 'L6.group')} |")
        h1.append(f["L1.cross"]["value"] >= 0.05)
        if "L5.shortcut" in f:
            h5.append((n, f["L5.shortcut"]["value"]))
    out += ["", "Ledger lines: " + ", ".join(f"`e1_audit:{n}` @ `{runs['e1_audit:' + n]['git']['commit']}`" for n in done), ""]
    k = sum(h1)
    out += ["### Hypotheses decided by E1 alone", "",
            f"- **H1** (at least half of the audited benchmarks have >= 5% of test rows exactly duplicating a training "
            f"row): {k} of {len(h1)} -> **{'supported' if 2 * k >= len(h1) else 'not supported'}**"
            + ("" if len(h1) == len(E1_ORDER) else f" (provisional: {len(h1)} of {len(E1_ORDER)} datasets audited)") + ".",
            f"- **H5** (at least one benchmark has a single feature reaching >= 0.95 balanced accuracy): "
            + (", ".join(f"{n} {pct(x)}" for n, x in h5 if x >= 0.95) or "none")
            + f" -> **{'supported' if any(x >= 0.95 for _, x in h5) else 'not supported'}**.", ""]
    return out


def main() -> None:
    runs = latest()
    lines = ["# Results", "",
             "Generated by `scripts/report.py` from `results/ledger.jsonl`. Do not edit by hand.", ""]
    for name, fn in (("e0_sources", e0_sources), ("e0_synthetic", e0_synthetic), ("e0_kdd", e0_kdd)):
        if name in runs:
            lines += fn(runs[name])
    lines += e1(runs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
