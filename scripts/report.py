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
SITE_DATA = ROOT / "site" / "data.json"


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
    out += ["", "Ledger lines: " + ", ".join(f"`e1_audit:{n}` @ `{runs['e1_audit:' + n]['git']['commit']}`" for n in done), ""]
    return out


MODELS = ("lr", "rf", "hgb", "mlp")
H2_DATASETS = ("kdd99", "uci_phishing")  # the two highest E1 exact cross-split rates (PLAN.md)


def pt(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:+.2f}"


def e2(runs: dict) -> list[str]:
    done = [n for n in E1_ORDER if f"e2_retrain:{n}" in runs]
    if not done:
        return []
    out = ["## E2: retraining the baselines", "",
           "Macro-F1, mean over 5 seeds (± sd). Each model is trained once and scored on the full test set "
           "(original), on test rows with no exact training copy (no_exact), on rows with neither an exact nor a near "
           "copy (no_near), and on the removed rows alone (leaked). `time` = retrained on a chronological split. "
           "Gap = original minus cleaned, in points.", ""]
    for n in done:
        d = runs[f"e2_retrain:{n}"]
        r = d["results"]
        bm = r["by_model"]
        has_time = any("time" in bm[m] for m in MODELS)
        out += [f"### {n}", "", src(d), "",
                f"Training rows used: {r['n_train']:,}; test rows: {r['n_test_mean']:,.0f}; "
                f"leaked test rows: exact {pct(r['leak_rate_mean']['exact'])}, exact or near {pct(r['leak_rate_mean']['near'])}.", "",
                "| Model | original | no_exact | no_near | leaked | gap no_exact | gap no_near |"
                + (" time | gap time |" if has_time else ""),
                "|---|---|---|---|---|---|---|" + ("---|---|" if has_time else "")]
        for m in MODELS:
            v = bm[m]

            def cell(c):
                if c not in v or v[c]["macro_f1_mean"] != v[c]["macro_f1_mean"]:
                    return "n/a"
                sd = v[c].get("macro_f1_std", 0.0)
                return f"{v[c]['macro_f1_mean']:.4f} ± {sd:.4f}"
            g = r["macro_f1_gap"][m]
            row = (f"| {m} | {cell('original')} | {cell('no_exact')} | {cell('no_near')} | {cell('leaked')} "
                   f"| {pt(g['orig_minus_no_exact'])} | {pt(g['orig_minus_no_near'])} |")
            if has_time:
                row += f" {cell('time')} | {pt(g.get('orig_minus_time'))} |"
            out.append(row)
        out.append("")

    return out


def e2_stats(runs: dict) -> list[str]:
    if "e2_stats" not in runs:
        return []
    d = runs["e2_stats"]
    r = d["results"]
    out = ["## E2 uncertainty (post hoc)", "", src(d), "",
           "Added after the E2 verdicts, so it changes none of them; it says which gaps the data can tell from zero. "
           "`seeds`: macro-F1 gap original - no_near, mean and 95% t interval over the 5 seeds (official splits keep "
           "the test set fixed, so this is model and subsample variation only). `rows`: seed 0, accuracy on leaked "
           "minus clean (no_near) test rows, 95% Newcombe interval, with the two-proportion p-value; it treats test rows "
           "as independent, which near-copy clusters are not, so it is optimistic. Points.", "",
           "| Dataset | Model | seeds: macro-F1 gap [95%] | rows: leaked n / clean n | rows: accuracy leaked - clean [95%] | p |",
           "|---|---|---|---|---|---|"]
    for n in E1_ORDER:
        if n not in r:
            continue
        for m in MODELS:
            v = r[n]["models"][m]
            rows_ci = (f"{pt(v['acc_diff'])} [{pt(v['acc_diff_lo'])}, {pt(v['acc_diff_hi'])}]" if "acc_diff" in v else "n/a")
            p = v.get("p_two_prop")
            ptxt = "n/a" if p is None else ("< 1e-6" if p < 1e-6 else f"{p:.3f}")
            out.append(f"| {n} | {m} | {pt(v['f1_gap_mean'])} [{pt(v['f1_gap_lo'])}, {pt(v['f1_gap_hi'])}] | "
                       f"{v['n_leaked']:,} / {v['n_clean']:,} | {rows_ci} | {ptxt} |")
    out += ["", "H4, paired by seed: LR gap minus HistGB gap (macro-F1 points), 95% t interval.", "",
            "| Dataset | LR - HistGB [95%] | Excludes 0 |", "|---|---|---|"]
    for n in E1_ORDER:
        h = r.get(n, {}).get("h4_paired")
        if h:
            excl = ("n/a (no gap)" if abs(h["lr_minus_hgb"]) < 1e-4
                    else "yes" if h["lo"] > 0 or h["hi"] < 0 else "no")
            out.append(f"| {n} | {pt(h['lr_minus_hgb'])} [{pt(h['lo'])}, {pt(h['hi'])}] | {excl} |")
    return out + [""]


def h4_paired(runs: dict, eligible: list[str]) -> list[str]:
    """Eligible H4 datasets whose paired LR - HistGB interval lies above zero (post hoc)."""
    r = runs.get("e2_stats", {}).get("results", {})
    return [n for n in eligible if n in r and r[n].get("h4_paired", {}).get("lo", 0) > 0]


def _verdict(ok: bool | None) -> str:
    return "pending" if ok is None else ("supported" if ok else "not supported")


def hypotheses(runs: dict) -> list[dict]:
    """H1-H5 under the rules in PLAN.md. Single source for RESULTS.md and site/data.json."""
    e1 = {n: _merged(runs[f"e1_audit:{n}"])[0] for n in E1_ORDER if f"e1_audit:{n}" in runs}
    gaps = {n: runs[f"e2_retrain:{n}"]["results"]["macro_f1_gap"] for n in E1_ORDER if f"e2_retrain:{n}" in runs}
    complete_e1, complete_e2 = len(e1) == len(E1_ORDER), len(gaps) == len(E1_ORDER)
    out = []

    hit = [n for n, f in e1.items() if f["L1.cross"]["value"] >= 0.05]
    out.append({"id": "H1", "statement": "At least half of the audited benchmarks have >= 5% of test rows exactly "
                "duplicating a training row.",
                "verdict": _verdict(2 * len(hit) >= len(e1) if complete_e1 else None),
                "evidence": f"{len(hit)} of {len(e1)} datasets: " + ", ".join(
                    f"{n} {pct(e1[n]['L1.cross']['value'])}" for n in sorted(hit, key=lambda n: -e1[n]['L1.cross']['value']))})

    h2 = None
    ev2 = "pending"
    if all(n in gaps for n in H2_DATASETS):
        per = {n: sum(gaps[n][m]["orig_minus_no_near"] >= 0.05 for m in MODELS) for n in H2_DATASETS}
        h2 = all(k >= 3 for k in per.values())
        ev2 = "; ".join(f"{n}: {k}/4 models lose >= 5 macro-F1 points (gaps " + ", ".join(
            pt(gaps[n][m]["orig_minus_no_near"]) for m in MODELS) + ")" for n, k in per.items())
    out.append({"id": "H2", "statement": "On the most affected benchmarks, removing cross-split duplicates and "
                "near-duplicates lowers macro-F1 of the standard baselines by >= 5 points.",
                "verdict": _verdict(h2), "evidence": ev2})

    h3_per = {}
    for n, g in gaps.items():
        if all("orig_minus_time" in g[m] for m in MODELS):
            h3_per[n] = sum(g[m]["orig_minus_time"] > g[m]["orig_minus_no_near"] for m in MODELS)
    h3 = (h3_per["ember2018"] >= 3) if "ember2018" in h3_per else None
    out.append({"id": "H3", "statement": "A time-aware split costs more accuracy than deduplication (decided on EMBER; "
                "CIC-IDS2017's chronological split is confounded with unseen attack types).",
                "verdict": _verdict(h3),
                "evidence": "; ".join(f"{n}: time gap > no_near gap for {k}/4 models" for n, k in h3_per.items()) or "pending"})

    eligible = [n for n, g in gaps.items() if max(abs(g["lr"]["orig_minus_no_near"]), abs(g["hgb"]["orig_minus_no_near"])) >= 0.01]
    wins = [n for n in eligible if gaps[n]["lr"]["orig_minus_no_near"] > gaps[n]["hgb"]["orig_minus_no_near"]]
    out.append({"id": "H4", "statement": "The accuracy lost to leakage shrinks as model strength grows: the gap for "
                "logistic regression exceeds the gap for gradient boosting.",
                "verdict": _verdict((2 * len(wins) > len(eligible)) if eligible and complete_e2 else None),
                "evidence": f"LR gap > HistGB gap on {len(wins)} of {len(eligible)} eligible datasets: " + ", ".join(
                    f"{n} {pt(gaps[n]['lr']['orig_minus_no_near'])} vs {pt(gaps[n]['hgb']['orig_minus_no_near'])}" for n in eligible)
                + (f". Post hoc (E2 uncertainty): the paired 95% interval lies above zero on {len(h4_paired(runs, wins))}"
                   f" of {len(wins)} ({', '.join(h4_paired(runs, wins))})" if "e2_stats" in runs else ""),
                "paired_above_zero": h4_paired(runs, wins) if "e2_stats" in runs else None,
                "wins": wins})

    flag = [(n, f["L5.shortcut"]["value"], f["L5.shortcut"]["top"][0]["feature"]) for n, f in e1.items()
            if "L5.shortcut" in f and f["L5.shortcut"]["value"] >= 0.95]
    out.append({"id": "H5", "statement": "At least one widely used benchmark has a single feature reaching >= 0.95 "
                "balanced accuracy on its own.",
                "verdict": _verdict(bool(flag) if complete_e1 else None),
                "evidence": ", ".join(f"{n}: {feat} {pct(x)}" for n, x, feat in flag) or "none"})
    return out


# --- phase 2 (PLAN.md): E5 predictions, E6 sweeps and bootstrap -----------------------

PHASE2_LARGE = ("kdd99", "nslkdd", "unsw_nb15", "ember2018")


def _cells(runs: dict) -> dict:
    return runs.get("e6_analysis", {}).get("results", {}).get("_cells", {})


def phase2_hypotheses(runs: dict) -> list[dict]:
    """H6-H8 under the rules in PLAN.md, phase 2. Single source for RESULTS.md and the site."""
    if "e6_analysis" not in runs:
        return []
    r = runs["e6_analysis"]["results"]
    cells = _cells(runs)
    out = []

    def cell(d, m, sweep="eps", value=0.01):
        return cells.get(f"{d}|{sweep}|{value}|{m}")

    ev6, ok6 = [], None
    if all(cell(d, "hgb_tuned") for d in ("kdd99", "nslkdd")):
        ok6 = True
        for d in ("kdd99", "nslkdd"):
            c = cell(d, "hgb_tuned")
            above = sum(lo > 0.05 for lo in c["lo"])
            ok6 &= c["gap_mean"] >= 0.05 and above >= 2
            ev6.append(f"{d}: tuned HistGB loses {pt(c['gap_mean'])} (interval above 5 points in {above}/3 seeds)")
    out.append({"id": "H6", "statement": "Tuning and full training data do not remove the leakage gap (tuned HistGB "
                "loses >= 5 macro-F1 points on KDD99 and NSL-KDD).", "verdict": _verdict(ok6),
                "evidence": "; ".join(ev6) or "pending"})

    ev7, wins = [], 0
    done = [d for d in PHASE2_LARGE if d in r]
    for d in done:
        seeds = r[d]["seeds"].values()
        k = sum(s["lr_minus_hgb_tuned"]["lo"] > 0 for s in seeds)
        wins += k >= 2
        diffs = [s["lr_minus_hgb_tuned"]["diff"] for s in seeds]
        ev7.append(f"{d}: LR - tuned HistGB {pt(sum(diffs) / len(diffs))}, interval above zero in {k}/3 seeds")
    ok7 = (wins >= 3) if len(done) == len(PHASE2_LARGE) else None
    out.append({"id": "H7", "statement": "LR's leakage gap exceeds tuned HistGB's, with a bootstrap interval above zero, "
                "on at least 3 of the 4 large benchmarks.", "verdict": _verdict(ok7),
                "evidence": "; ".join(ev7) or "pending"})

    total = same = 0
    flips = []
    for key, c in cells.items():
        d, sweep, value, m = key.split("|")
        if sweep != "eps" or float(value) != 0.01:
            continue
        states = [cells.get(f"{d}|eps|{v}|{m}") for v in (0.003, 0.01, 0.03)]
        if any(x is None for x in states):
            continue
        total += 1
        big = {x["gap_mean"] >= 0.01 for x in states}
        if len(big) == 1:
            same += 1
        else:
            flips.append(f"{d}/{m}")
    ok8 = (same >= 0.9 * total) if total else None
    out.append({"id": "H8", "statement": "Conclusions do not hinge on the near-copy threshold: whether a gap is >= 1 "
                "point is the same at eps 0.003, 0.01 and 0.03 in >= 90% of dataset x model cells.",
                "verdict": _verdict(ok8),
                "evidence": f"same in {same} of {total} cells" + (f"; changes: {', '.join(flips)}" if flips else "")})
    return out


def e6(runs: dict) -> list[str]:
    if "e6_analysis" not in runs:
        return []
    d = runs["e6_analysis"]
    cells = _cells(runs)
    models = d["config"]["models"]
    out = ["## E5/E6: uncapped and tuned models, threshold sweeps, bootstrap intervals", "", src(d), "",
           "Full training sets (no cap), 3 seeds, plus `hgb_tuned` (randomised search, PLAN.md phase 2). "
           "Macro-F1 gap original - clean in points, mean over seeds; the bracket is the range of the seeds' "
           "95% bootstrap intervals (1,000 resamples over test rows).", "",
           "### At the default threshold (eps 0.01, Jaccard 0.8)", "",
           "| Dataset | Leaked share | " + " | ".join(models) + " |", "|---|--:|" + "--:|" * len(models)]
    for n in E1_ORDER:
        row = [cells.get(f"{n}|eps|0.01|{m}") for m in models]
        if not all(row):
            continue
        out.append(f"| {n} | {pct(row[0]['leaked_share_mean'])} | " + " | ".join(
            f"{pt(c['gap_mean'])} [{pt(min(c['lo']))}, {pt(max(c['hi']))}]" for c in row) + " |")
    out += ["", "### Gap against the near-copy threshold (tuned HistGB, then LR)", "",
            "| Dataset | " + " | ".join(f"eps {v}" for v in (0.001, 0.003, 0.01, 0.03, 0.1)) + " |",
            "|---|" + "--:|" * 5]
    for n in E1_ORDER:
        for m in ("hgb_tuned", "lr"):
            cs = [cells.get(f"{n}|eps|{v}|{m}") for v in (0.001, 0.003, 0.01, 0.03, 0.1)]
            if all(cs):
                out.append(f"| {n} ({m}) | " + " | ".join(
                    f"{pt(c['gap_mean'])} ({pct(c['leaked_share_mean'])} leaked)" for c in cs) + " |")
    hs = phase2_hypotheses(runs)
    out += ["", "### Phase 2 verdicts", "", "| | Hypothesis | Verdict | Evidence |", "|---|---|---|---|"]
    out += [f"| {h['id']} | {h['statement']} | **{h['verdict']}** | {h['evidence']} |" for h in hs]
    return out + [""]



def hypotheses_md(runs: dict) -> list[str]:
    hs = hypotheses(runs)
    out = ["## Hypothesis verdicts", "", "Decision rules were committed in PLAN.md before the runs they govern.", "",
           "| | Hypothesis | Verdict | Evidence |", "|---|---|---|---|"]
    out += [f"| {h['id']} | {h['statement']} | **{h['verdict']}** | {h['evidence']} |" for h in hs]
    return out + [""]


def e4(runs: dict) -> list[str]:
    done = [n for n in E1_ORDER if f"e4_ablation:{n}" in runs]
    if not done:
        return []
    out = ["## E4: shortcut ablation", "",
           "Retrain without the features E1 flagged (drop sets read from the E1 ledger). Macro-F1 on the full test set, "
           "mean over 5 seeds. A shortcut carries the score if removing it costs >= 5 points for at least 3 of 4 models "
           "(PLAN.md, week 6).", ""]
    for n in done:
        d = runs[f"e4_ablation:{n}"]
        r = d["results"]
        out += [f"### {n}", "", src(d), "",
                "| Condition | Dropped | " + " | ".join(MODELS) + " | Best single feature afterwards |",
                "|---|---|" + "---|" * len(MODELS) + "---|"]
        for c, v in r["conditions"].items():
            cells = " | ".join(f"{v['models'][m]['macro_f1_mean']:.4f}" + (
                f" ({pt(r['macro_f1_drop_vs_full'][c][m])})" if c in r["macro_f1_drop_vs_full"] else "")
                for m in MODELS)
            after = v.get("best_single_feature_after") or []
            best = f"{after[0]['feature']} {pct(after[0]['balanced_acc'])}" if after else "n/a"
            out.append(f"| {c} | {', '.join(v['dropped']) or '(none)'} | {cells} | {best} |")
        carried = {c: sum(x >= 0.05 for x in dr.values()) for c, dr in r["macro_f1_drop_vs_full"].items()}
        out += ["", "Shortcut carries the score: " + "; ".join(
            f"{c}: {k}/4 models drop >= 5 pt -> **{'yes' if k >= 3 else 'no'}**" for c, k in carried.items()), ""]
    return out


DISPLAY = {
    "kdd99": ("KDD Cup 99", "Network intrusion"), "nslkdd": ("NSL-KDD", "Network intrusion"),
    "unsw_nb15": ("UNSW-NB15", "Network intrusion"), "cicids2017": ("CIC-IDS2017", "Network intrusion"),
    "ember2018": ("EMBER 2018", "Malware (PE files)"), "uci_phishing": ("UCI Phishing Websites", "Phishing"),
    "phiusiil": ("PhiUSIIL", "Phishing URLs"), "sms_spam": ("SMS Spam Collection", "Spam"),
}
CHECKS = [
    {"id": "L1", "name": "Exact copies", "what": "Test rows identical to a training row, after canonical hashing."},
    {"id": "L2", "name": "Near copies", "what": "Test rows within a tiny distance of a training row (tabular), or "
     "Jaccard >= 0.8 (text, MinHash + exact check), compared with how close training rows sit to each other."},
    {"id": "L3", "name": "Label conflicts", "what": "Identical features with different labels, and the accuracy "
     "ceiling they impose."},
    {"id": "L4", "name": "Time leakage", "what": "Training rows newer than test rows, and the timestamp as a "
     "one-feature classifier."},
    {"id": "L5", "name": "Shortcuts", "what": "A single feature that predicts the label on its own, and "
     "identifier-like columns."},
    {"id": "L6", "name": "Group overlap", "what": "The same host, family or session on both sides of the split."},
]


def _num(x):
    return None if x is None or x != x else round(float(x), 6)


def site_data(runs: dict) -> dict:
    """Every number the project page shows, straight from the ledger."""
    used = []

    def ref(key):
        used.append(f"{key} @ {runs[key]['git']['commit']}")
        return {"ledger": key, "commit": runs[key]["git"]["commit"]}

    audit, retrain, ablation = [], [], []
    for n in E1_ORDER:
        name, domain = DISPLAY[n]
        if f"e1_audit:{n}" in runs:
            d = runs[f"e1_audit:{n}"]
            f, meta = _merged(d)
            g = lambda c, k="value": _num(f[c].get(k)) if c in f else None  # noqa: E731
            top = f.get("L5.shortcut", {}).get("top", [{}])[0]
            audit.append({
                "id": n, "name": name, "domain": domain,
                "split": "official" if d["results"]["split_policy"].startswith("official") else "random 80/20",
                "n_train": meta["n_train"], "n_test": meta["n_test"],
                "exact_copies": g("L1.cross"), "near_copies": g("L2.near"), "near_vs_baseline": g("L2.near", "vs_train_baseline"),
                "text_near_copies": g("L2.near_text"), "accuracy_ceiling": g("L3.ceiling"),
                "contradicted": g("L3.cross_contradict"), "train_after_test_start": g("L4.after_test_start"),
                "best_single_feature": top.get("feature"), "best_single_feature_bacc": g("L5.shortcut"),
                "group_overlap": g("L6.group"), "source": ref(f"e1_audit:{n}"),
            })
        if f"e2_retrain:{n}" in runs:
            d = runs[f"e2_retrain:{n}"]
            r = d["results"]
            models = {}
            for m in MODELS:
                bm, gp = r["by_model"][m], r["macro_f1_gap"][m]
                models[m] = {
                    "original": _num(bm["original"]["macro_f1_mean"]), "no_exact": _num(bm["no_exact"]["macro_f1_mean"]),
                    "no_near": _num(bm["no_near"]["macro_f1_mean"]),
                    "leaked": _num(bm["leaked"]["macro_f1_mean"]) if "leaked" in bm else None,
                    "gap_no_exact": _num(gp["orig_minus_no_exact"]), "gap_no_near": _num(gp["orig_minus_no_near"]),
                    "time": _num(bm["time"]["macro_f1_mean"]) if "time" in bm else None,
                    "gap_time": _num(gp.get("orig_minus_time")),
                }
            st = runs.get("e2_stats", {}).get("results", {}).get(n)
            if st:
                for m in MODELS:
                    v = st["models"][m]
                    models[m]["gap_no_near_ci"] = [_num(v["f1_gap_lo"]), _num(v["f1_gap_hi"])]
            retrain.append({"id": n, "name": name, "leaked_exact": _num(r["leak_rate_mean"]["exact"]),
                            "leaked_near": _num(r["leak_rate_mean"]["near"]), "n_train_used": r["n_train"],
                            "models": models, "source": ref(f"e2_retrain:{n}"),
                            **({"h4_paired": {k: _num(x) for k, x in st["h4_paired"].items()},
                                "stats_source": ref("e2_stats")} if st and "h4_paired" in st else {})})
        if f"e4_ablation:{n}" in runs:
            d = runs[f"e4_ablation:{n}"]
            r = d["results"]
            for c, v in r["conditions"].items():
                after = (v.get("best_single_feature_after") or [{}])[0]
                ablation.append({"id": n, "name": name, "condition": c, "dropped": v["dropped"],
                                 "macro_f1": {m: _num(v["models"][m]["macro_f1_mean"]) for m in MODELS},
                                 "drop_vs_full": {m: _num(r["macro_f1_drop_vs_full"].get(c, {}).get(m)) for m in MODELS},
                                 "best_single_feature_after": after.get("feature"),
                                 "best_single_feature_after_bacc": _num(after.get("balanced_acc")),
                                 "source": ref(f"e4_ablation:{n}")})
    validation = {}
    if "e0_kdd" in runs:
        validation["tavallaee"] = {"rows": runs["e0_kdd"]["results"]["tavallaee_comparison"], "source": ref("e0_kdd")}
    if "e0_synthetic" in runs:
        r = runs["e0_synthetic"]["results"]
        validation["synthetic"] = {"configurations": r["exact_recovery"]["cells"],
                                   "max_abs_error": r["exact_recovery"]["max_abs_error"],
                                   "interval_coverage": r["interval_coverage"]["coverage"],
                                   "minhash_recall": {k: v["recall"] for k, v in r["minhash"]["recall_by_jaccard"].items()},
                                   "source": ref("e0_synthetic")}
    rows = sum(a["n_train"] + a["n_test"] for a in audit)
    return {
        "generated_from": "results/ledger.jsonl via scripts/report.py",
        "ledger_lines": sorted(set(used)),
        "summary": {"datasets": len(audit), "rows_audited": rows,
                    "datasets_with_5pct_exact_copies": sum((a["exact_copies"] or 0) >= 0.05 for a in audit),
                    "models": list(MODELS), "seeds": 5},
        "checks": CHECKS, "hypotheses": hypotheses(runs), "audit": audit, "retrain": retrain,
        "ablation": ablation, "validation": validation,
    }


def main() -> None:
    runs = latest()
    lines = ["# Results", "",
             "Generated by `scripts/report.py` from `results/ledger.jsonl`. Do not edit by hand.", ""]
    lines += hypotheses_md(runs)
    for name, fn in (("e0_sources", e0_sources), ("e0_synthetic", e0_synthetic), ("e0_kdd", e0_kdd)):
        if name in runs:
            lines += fn(runs[name])
    lines += e1(runs)
    lines += e2(runs)
    lines += e2_stats(runs)
    lines += e6(runs)
    lines += e4(runs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(json.dumps(site_data(runs), indent=1), encoding="utf-8")
    print(f"wrote {SITE_DATA}")


if __name__ == "__main__":
    main()
