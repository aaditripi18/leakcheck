"""Build the static site in site/ from the ledger.

  python scripts/report.py          # refreshes docs/RESULTS.md and site/data.json
  python scripts/build_site.py      # site/index.html, site/scan.html, site/examples/*.js, site/assets/site.css

Every number on the pages comes from site/data.json (generated from
results/ledger.jsonl) or is measured here by leakcheck itself (example
datasets). Design: Stitch; content and wiring: this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SITE = ROOT / "site"
REPO_URL = "https://github.com/aaditripi18/leakcheck"  # links to code, findings and provenance

CSP = ("default-src 'none'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'none'; "
       "worker-src 'self' blob:; form-action 'none'; base-uri 'none'; manifest-src 'self'")


def e(x) -> str:
    return escape(str(x), quote=True)


def fpct(x: float | None) -> str:
    if x is None:
        return "—"
    v = x * 100
    if v == 0:
        return "0.0%"
    if abs(v) < 0.01:
        return f"{v:.3f}%"
    if abs(v) < 1:
        return f"{v:.2f}%"
    return f"{v:.1f}%"


def fpt(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


# --- icons (inline SVG, stroke = currentColor) -------------------------------------

_ICONS = {
    "north_east": '<path d="M7 17 17 7M9 7h8v8"/>',
    "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "lock": '<rect x="4" y="11" width="16" height="10" rx="1"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "check": '<path d="m5 12 5 5 9-10"/>',
    "upload": '<path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v4h16v-4"/>',
    "play": '<path d="M7 5v14l12-7z"/>',
    "download": '<path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16"/>',
    "chevron": '<path d="m6 9 6 6 6-6"/>',
    "terminal": '<path d="m5 8 4 4-4 4M12 16h7"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "file": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="1"/><path d="M16 8V4H4v12h4"/>',
}


def icon(name: str, cls: str = "") -> str:
    return (f'<svg class="icon {cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


# --- shared chrome ---------------------------------------------------------------------


def head(title: str, description: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{CSP}">
<meta name="referrer" content="no-referrer">
<title>{e(title)}</title>
<meta name="description" content="{e(description)}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%23111110'/%3E%3Crect x='3' y='3' width='4' height='10' fill='%23d97706'/%3E%3Crect x='9' y='3' width='4' height='10' fill='%23f9f9f6'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<link rel="stylesheet" href="assets/site.css">
<script src="assets/theme.js"></script>
</head>
"""


def nav(active: str) -> str:
    links = [("Results", "index.html#audit", "results"), ("Hypotheses", "index.html#hypotheses", "hypotheses"),
             ("Scanner", "scan.html", "scanner"), ("Reproduce", "index.html#reproduce", "reproduce")]
    items = []
    for label, href, key in links:
        cur = key == active
        cls = ("text-on-surface font-medium underline underline-offset-8 decoration-primary" if cur
               else "text-on-surface-variant hover:text-on-surface")
        items.append(f'<a class="font-mono-data-sm text-mono-data-sm transition-colors {cls}" href="{href}"'
                     + (' aria-current="page"' if cur else "") + f">{label}</a>")
    items.append(f'<a class="font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface '
                 f'inline-flex items-center gap-0.5" href="{REPO_URL}" rel="noopener noreferrer">GitHub'
                 f'{icon("north_east", "text-[12px]")}</a>')
    return f"""<body>
<a href="#main" class="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 bg-surface p-2 z-[60]">Skip to content</a>
<header class="fixed top-0 w-full z-50 bg-background/90 backdrop-blur-md border-b border-surface-container-high">
  <div class="h-14 max-w-[1100px] mx-auto px-margin-mobile md:px-gutter flex items-center justify-between gap-space-md">
    <div class="flex items-center gap-space-sm">
      <a class="font-mono-data-lg text-mono-data-lg text-on-surface tracking-tight font-medium hover:text-primary transition-colors" href="index.html">leakcheck</a>
      <span class="hidden sm:inline-flex items-center border border-outline-variant px-space-xs py-0.5 bg-surface-container-low sec-label text-on-surface-variant">v0.1</span>
    </div>
    <nav class="hidden md:flex items-center gap-space-lg" aria-label="Main">{"".join(items)}</nav>
    <div class="flex items-center gap-space-sm">
      <a class="md:hidden inline-block py-2 px-1 font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface" href="{"index.html" if active == "scanner" else "scan.html"}">{"Report" if active == "scanner" else "Scanner"}</a>
      <button type="button" data-theme-toggle aria-label="Toggle dark mode" class="w-8 h-8 flex items-center justify-center text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors">
        <span class="dark:hidden">{icon("moon", "text-[18px]")}</span><span class="hidden dark:inline">{icon("sun", "text-[18px]")}</span>
      </button>
    </div>
  </div>
</header>
<main id="main" class="w-full pt-14">
"""


def footer(generated: str) -> str:
    return f"""</main>
<footer class="w-full bg-surface-container-low border-t border-surface-container-high py-space-xl mt-space-2xl">
  <div class="max-w-[1100px] mx-auto px-margin-mobile md:px-gutter flex flex-col md:flex-row justify-between items-start md:items-center gap-space-lg">
    <div class="flex flex-col gap-space-xs">
      <span class="font-mono-data-md text-mono-data-md font-medium text-on-surface">leakcheck</span>
      <p class="font-mono-data-sm text-mono-data-sm text-on-surface-variant">Built with a ledger, not a slide deck. Code under the <a class="underline underline-offset-2 hover:text-on-surface" href="{REPO_URL}/blob/main/LICENSE" rel="noopener noreferrer">MIT licence</a>.</p>
    </div>
    <nav class="flex flex-wrap items-center gap-space-md" aria-label="Footer">
      <a class="inline-block py-2 font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface" href="index.html">Report</a>
      <a class="inline-block py-2 font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface" href="scan.html">Scanner</a>
      <a class="inline-block py-2 font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface" href="{REPO_URL}/blob/main/docs/FINDINGS.md" rel="noopener noreferrer">Findings</a>
      <a class="inline-block py-2 font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface" href="{REPO_URL}/blob/main/data/README.md" rel="noopener noreferrer">Data provenance</a>
      <a class="inline-block py-2 font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface" href="{REPO_URL}" rel="noopener noreferrer">GitHub</a>
    </nav>
  </div>
  <div class="max-w-[1100px] mx-auto px-margin-mobile md:px-gutter mt-space-lg pt-space-md border-t border-surface-container-highest flex flex-col sm:flex-row gap-space-xs justify-between text-on-surface-variant font-mono-data-sm text-mono-data-sm">
    <span>Every number on this site is generated from <code>results/ledger.jsonl</code>.</span>
    <span>Generated {e(generated)}</span>
  </div>
</footer>
<script src="assets/site.js"></script>
"""


def sec_head(num: str, kicker: str, title: str, anchor: str | None = None, level: str = "h2") -> str:
    a = f' id="{anchor}"' if anchor else ""
    return f"""<div{a} class="scroll-mt-20">
  <div class="flex items-center gap-space-xs"><span class="sec-label text-primary">{num}</span><span class="w-1.5 h-1.5 bg-outline-variant"></span><span class="sec-label text-on-surface-variant">{e(kicker)}</span></div>
  <{level} class="font-headline-md text-headline-md text-on-surface mt-1">{e(title)}</{level}>
</div>"""


# --- example datasets for the scanner --------------------------------------------------


def build_examples(force: bool) -> dict:
    """Write site/examples/*.js and return their leakcheck-measured summaries."""
    import numpy as np  # noqa: F401  (pandas needs it; keeps import errors obvious)

    from experiments.data import load_kdd99
    from experiments.e1_audit import prep_uci_phishing
    from leakcheck import Split, scan
    from leakcheck.synth import make_split

    out_dir = SITE / "examples"
    out_dir.mkdir(exist_ok=True)
    meta_path = out_dir / "examples.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    examples = {}

    def emit(key, name, train, test, label, ignore, source, licence, extra=None):
        split = Split(train, test, label, ignore=ignore)
        rep = scan(split, checks=("L1",))
        cross = rep.get("L1.cross")
        payload = {"name": name, "train": train.to_csv(index=False), "test": test.to_csv(index=False),
                   "label": label, "ignore": ignore, "source": source, "licence": licence,
                   "files": {"train": f"{key}_train.csv", "test": f"{key}_test.csv"}}
        js = ("// Example split for the leakcheck scanner. Loaded as a script, so the page needs no network access.\n"
              f"// Source: {source}. Licence: {licence}.\n"
              f"(window.LEAKCHECK_EXAMPLES = window.LEAKCHECK_EXAMPLES || {{}})[{json.dumps(key)}] = "
              f"{json.dumps(payload)};\n")
        (out_dir / f"{key}.js").write_text(js, encoding="utf-8")
        examples[key] = {"name": name, "exact_copies": cross.value, "n_train": len(train), "n_test": len(test),
                         **(extra or {})}

    # KDD Cup 99: a random sample of the official training set and labelled test set
    full, test = load_kdd99("full"), load_kdd99("test")
    tr = full.sample(n=12_000, random_state=0).reset_index(drop=True)
    te = test.sample(n=4_000, random_state=0).reset_index(drop=True)
    pairs = []
    split = Split(tr, te, "is_attack", ignore=["label"])
    hits = np.flatnonzero(np.isin(split.test_hash, split.train_hash))
    first_train = {}
    for i, h in enumerate(split.train_hash):
        first_train.setdefault(int(h), i)
    for ti in hits[:4]:
        tj = first_train[int(split.test_hash[ti])]
        pairs.append({"test": int(ti) + 1, "train": int(tj) + 1, "label": str(te.loc[ti, "label"]),
                      "proto": f'{te.loc[ti, "protocol_type"]}/{te.loc[ti, "service"]}/{te.loc[ti, "flag"]}',
                      "src_bytes": int(te.loc[ti, "src_bytes"])})
    others_te = [i for i in range(len(te)) if i not in set(hits)][:2]
    others_tr = [j for j in range(len(tr)) if j not in {p["train"] - 1 for p in pairs}][:2]
    context = {"test": [{"row": i + 1, "label": str(te.loc[i, "label"]),
                         "proto": f'{te.loc[i, "protocol_type"]}/{te.loc[i, "service"]}/{te.loc[i, "flag"]}'} for i in others_te],
               "train": [{"row": j + 1, "label": str(tr.loc[j, "label"]),
                          "proto": f'{tr.loc[j, "protocol_type"]}/{tr.loc[j, "service"]}/{tr.loc[j, "flag"]}'} for j in others_tr]}
    emit("kdd99", "KDD Cup 99 sample", tr, te, "is_attack", ["label"],
         "UCI KDD Cup 1999 (kddcup.data, corrected), random 12,000 / 4,000 rows, seed 0", "CC BY 4.0",
         {"pairs": pairs, "context": context})

    # UCI Phishing Websites: the E1 split (stratified 80/20, seed 0)
    (_, s, _), = prep_uci_phishing().splits
    emit("uci_phishing", "UCI Phishing Websites", s.train, s.test, "Result", [],
         "UCI Phishing Websites (dataset 327), stratified 80/20 split, seed 0", "CC BY 4.0")

    # a clean control: synthetic data with nothing planted
    tr, te, _ = make_split(n_train=3000, n_test=1000, n_features=8, seed=42)
    emit("control", "Clean control split", tr, te, "label", [], "leakcheck.synth.make_split, seed 42, no leakage planted",
         "MIT (generated)")

    meta_path.write_text(json.dumps(examples, indent=1), encoding="utf-8")
    return examples


# --- index.html ------------------------------------------------------------------------

SHORT_H = {
    "H1": "At least half of the benchmarks leak ≥ 5% of their test set into training.",
    "H2": "Removing the copies costs ≥ 5 macro-F1 points on the two most-copied benchmarks.",
    "H3": "A time-based split hurts more than removing copies.",
    "H4": "Weaker models lean harder on leaked rows.",
    "H5": "Some benchmark is solved by a single feature.",
}


def short_evidence(h: dict, d: dict) -> str:
    rt = {r["id"]: r for r in d["retrain"]}
    au = {a["id"]: a for a in d["audit"]}
    if h["id"] == "H1":
        n = d["summary"]["datasets_with_5pct_exact_copies"]
        return f"{n} of {d['summary']['datasets']} datasets."
    if h["id"] == "H2":
        k = [rt["kdd99"]["models"][m]["gap_no_near"] for m in ("lr", "rf", "hgb", "mlp")]
        u = [rt["uci_phishing"]["models"][m]["gap_no_near"] for m in ("lr", "rf", "hgb", "mlp")]
        return f"KDD99 loses {fpt(min(k))}–{fpt(max(k))} points; UCI Phishing loses at most {fpt(max(u))}."
    if h["id"] == "H3":
        em = rt["ember2018"]["models"]
        t = [em[m]["gap_time"] for m in em]
        nn = [em[m]["gap_no_near"] for m in em]
        diffs = [em[m]["gap_time"] - em[m]["gap_no_near"] for m in em]
        return (f"EMBER: the time split costs {fpt(min(t))}–{fpt(max(t))} points vs {fpt(min(nn))}–{fpt(max(nn))} for "
                f"removing near-copies; a margin of {fpt(min(diffs))}–{fpt(max(diffs))}.")
    if h["id"] == "H4":
        return h["evidence"].split(":")[0] + "."
    if h["id"] == "H5":
        a = max(d["audit"], key=lambda a: a["best_single_feature_bacc"] or 0)
        return f"{a['name']}: {a['best_single_feature']} alone reaches {fpct(a['best_single_feature_bacc'])} balanced accuracy."
    return h["evidence"]


def build_index(d: dict, ex: dict, generated: str) -> str:
    au = {a["id"]: a for a in d["audit"]}
    rt = {r["id"]: r for r in d["retrain"]}
    kdd, uci = au["kdd99"], au["uci_phishing"]
    kdd_gap = rt["kdd99"]["models"]["lr"]["gap_no_near"]
    kl = rt["kdd99"]["models"]["lr"]
    near_share = 1 - kl["gap_no_exact"] / kl["gap_no_near"]  # part of the LR gap that exact copies do not explain
    uci_gap = rt["uci_phishing"]["models"]["lr"]["gap_no_near"]
    syn = d["validation"]["synthetic"]
    syn_err = sum(syn["max_abs_error"].values())

    # hero illustration from real KDD sample pairs
    kx = ex["kdd99"]
    tr_rows = [f'<div class="p-1 bg-surface-container-lowest text-secondary truncate">#{c["row"]} · {e(c["label"])} · {e(c["proto"])}</div>'
               for c in kx["context"]["train"][:1]]
    te_rows = [f'<div class="p-1 bg-surface-container-lowest text-secondary truncate">#{c["row"]} · {e(c["label"])} · {e(c["proto"])}</div>'
               for c in kx["context"]["test"][:1]]
    for p in kx["pairs"][:3]:
        tr_rows.append(f'<div class="p-1 bg-primary/10 text-primary font-medium truncate flex items-center justify-between gap-2">'
                       f'<span class="truncate">#{p["train"]} · {e(p["label"])} · {e(p["proto"])}</span><span class="text-[9px] shrink-0">copied</span></div>')
        te_rows.append(f'<div class="p-1 bg-primary/10 text-primary font-medium truncate flex items-center justify-between gap-2 border-l-2 border-primary">'
                       f'<span class="truncate">#{p["test"]} · {e(p["label"])} · {e(p["proto"])}</span><span class="shrink-0">= #{p["train"]}</span></div>')
    tr_rows += [f'<div class="p-1 bg-surface-container-lowest text-secondary truncate">#{c["row"]} · {e(c["label"])} · {e(c["proto"])}</div>'
                for c in kx["context"]["train"][1:2]]
    te_rows += [f'<div class="p-1 bg-surface-container-lowest text-secondary truncate">#{c["row"]} · {e(c["label"])} · {e(c["proto"])}</div>'
                for c in kx["context"]["test"][1:2]]

    checks = "".join(
        f"""<div class="py-space-sm grid grid-cols-1 md:grid-cols-12 gap-space-xs items-baseline">
  <span class="md:col-span-3 font-mono-data-md text-mono-data-md {'text-primary' if i % 2 == 0 else 'text-on-surface'} font-bold">{e(c['id'])} · {e(c['name']).upper()}</span>
  <span class="md:col-span-9 font-body-sm text-body-sm text-on-surface-variant">{e(c['what'])}</span>
</div>""" for i, c in enumerate(d["checks"]))

    rows = []
    for a in d["audit"]:
        exact = fpct(a["exact_copies"])
        exact_cell = f'<span class="pill-leak">{exact}</span>' if (a["exact_copies"] or 0) >= 0.05 else f'<span class="text-secondary">{exact}</span>'
        near = "—" if a["near_copies"] is None else fpct(a["near_copies"]) + (
            f' <span class="text-secondary">(×{a["near_vs_baseline"]:.2f})</span>' if a["near_vs_baseline"] else "")
        best = "—"
        if a["best_single_feature"] and a["id"] != "sms_spam":
            txt = f'{e(a["best_single_feature"])} <span class="text-secondary">({fpct(a["best_single_feature_bacc"])})</span>'
            best = (f'<span class="pill-leak">{e(a["best_single_feature"])} {fpct(a["best_single_feature_bacc"])}</span>'
                    if a["best_single_feature_bacc"] >= 0.95 else txt)
        rows.append(f"""<tr class="hover:bg-surface-container-low transition-colors">
  <th scope="row" class="py-space-sm px-space-md font-medium text-on-surface text-left font-mono-data-sm">{e(a['name'])}</th>
  <td class="py-space-sm px-space-md text-secondary">{e(a['domain'])}</td>
  <td class="py-space-sm px-space-md text-secondary">{e(a['split'])}</td>
  <td class="py-space-sm px-space-md text-right tabular">{a['n_test']:,}</td>
  <td class="py-space-sm px-space-md text-right tabular">{exact_cell}</td>
  <td class="py-space-sm px-space-md text-right tabular">{near}</td>
  <td class="py-space-sm px-space-md text-right">{best}</td>
</tr>""")

    bars = []
    order = sorted(d["retrain"], key=lambda r: -r["models"]["lr"]["gap_no_near"])
    for r in order:
        lr, hgb = r["models"]["lr"]["gap_no_near"], r["models"]["hgb"]["gap_no_near"]
        w = lambda x: max(0.0, min(100.0, x * 100 / 20 * 100))  # noqa: E731  0-20 points axis
        label = r["name"] + (" (random split)" if r["id"] == "ember2018" else "")
        bars.append(f"""<div class="grid grid-cols-12 items-center gap-2">
  <span class="col-span-4 sm:col-span-3 text-on-surface font-medium truncate" title="{e(label)}">{e(label)}</span>
  <div class="col-span-6 sm:col-span-7 flex flex-col gap-1" role="img" aria-label="{e(r['name'])}: logistic regression loses {fpt(lr)} points, gradient boosting {fpt(hgb)}">
    <div class="w-full bg-surface-container-high h-2.5"><div class="h-2.5 bg-primary w-0" data-w="{w(lr):.1f}"></div></div>
    <div class="w-full bg-surface-container-high h-2.5"><div class="h-2.5 bg-bar w-0" data-w="{w(hgb):.1f}"></div></div>
  </div>
  <span class="col-span-2 text-right tabular leading-tight"><span class="text-primary font-bold block">{fpt(lr)}</span><span class="text-on-surface-variant block">{fpt(hgb)}</span></span>
</div>""")

    hyps = []
    for h in d["hypotheses"]:
        ok = h["verdict"] == "supported"
        pill = ('<span class="inline-flex items-center px-2.5 py-0.5 font-mono-data-sm text-mono-data-sm font-semibold '
                + ('bg-success/10 text-success border border-success/30">SUPPORTED' if ok
                   else 'bg-surface-container-high text-secondary border border-outline-variant">NOT SUPPORTED') + "</span>")
        hyps.append(f"""<div class="p-space-md flex flex-col md:flex-row md:items-center justify-between gap-space-sm hover:bg-surface-container-low transition-colors">
  <div class="flex items-start gap-space-md max-w-3xl">
    <span class="font-mono-data-md text-mono-data-md font-bold text-on-surface bg-surface-container px-2 py-0.5">{e(h['id'])}</span>
    <div class="flex flex-col">
      <span class="font-body-md text-body-md font-medium text-on-surface">{e(SHORT_H[h['id']])}</span>
      <span class="font-mono-data-sm text-mono-data-sm text-secondary mt-0.5">{e(short_evidence(h, d))}</span>
    </div>
  </div>
  <div class="self-start md:self-center shrink-0">{pill}</div>
</div>""")

    # findings: every number from data
    em = rt["ember2018"]
    ph = [a for a in d["ablation"] if a["id"] == "phiusiil"]
    ph_min = min(min(a["macro_f1"].values()) for a in ph)
    cic = rt["cicids2017"]["models"]
    cic_orig = [cic[m]["original"] for m in cic]
    cic_time = [cic[m]["time"] for m in cic]
    ember_audit = au["ember2018"]
    findings = [
        ("Copying is not the same as inflation.",
         f"UCI Phishing copies about as much as KDD Cup 99 ({fpct(uci['exact_copies'])} vs {fpct(kdd['exact_copies'])} of test rows) "
         f"but loses only {fpt(uci_gap)} macro-F1 points when the copies go, against {fpt(kdd_gap)} for KDD Cup 99. "
         "With 30 features of 2–3 values each, a UCI Phishing copy is just a common pattern a model gets right anyway; "
         f"KDD Cup 99's copies are memorised exceptions, and {near_share * 100:.0f}% of its gap comes from near-copies rather than exact ones.",
         f"Logistic regression: KDD99 −{fpt(kdd_gap)} vs UCI −{fpt(uci_gap)}"),
        ("The control comes back clean.",
         f"EMBER 2018's official split is by time. leakcheck finds {fpct(ember_audit['exact_copies'])} exact copies, "
         f"{fpct(ember_audit['train_after_test_start'])} of training files newer than the test period, and test rows further "
         f"from training than training rows are from each other (×{ember_audit['near_vs_baseline']:.2f}). The audit does not flag everything.",
         f"EMBER 2018: {fpct(ember_audit['exact_copies'])} copies · no future data"),
        ("Too easy to rank models.",
         f"Every baseline scores at least {min(rt['phiusiil']['models'][m]['original'] for m in rt['phiusiil']['models']) * 100:.2f}% "
         f"macro-F1 on PhiUSIIL. Removing the four features that each reach 95% on their own still leaves at least {ph_min * 100:.2f}%. "
         "A test that easy cannot tell models apart.",
         f"Best single feature: {au['phiusiil']['best_single_feature']} {fpct(au['phiusiil']['best_single_feature_bacc'])}"),
        ("Random splits can hide a collapse.",
         f"On CIC-IDS2017 the baselines score {min(cic_orig) * 100:.1f}–{max(cic_orig) * 100:.1f}% macro-F1 on a random split, "
         f"and {min(cic_time) * 100:.1f}–{max(cic_time) * 100:.1f}% when trained on earlier capture days and tested on the last ones, "
         "whose attack types (PortScan, DDoS) never appear in training.",
         f"Gradient boosting: random split {cic['hgb']['original'] * 100:.1f}% → later days {cic['hgb']['time'] * 100:.1f}%"),
    ]
    finding_cards = "".join(f"""<div class="p-space-lg bg-surface-container-low border border-surface-container-high flex flex-col justify-between">
  <div class="flex flex-col gap-space-xs">
    <span class="font-mono-data-sm text-mono-data-sm text-primary font-semibold">FINDING {i + 1:02d}</span>
    <h3 class="font-headline-sm text-headline-sm text-on-surface">{e(t)}</h3>
    <p class="font-body-md text-body-md text-on-surface-variant mt-2">{e(b)}</p>
  </div>
  <div class="mt-space-md pt-space-xs border-t border-surface-container-high font-mono-data-sm text-mono-data-sm text-secondary">{e(f)}</div>
</div>""" for i, (t, b, f) in enumerate(findings))

    tav_rows = []
    for split_name, classes in d["validation"]["tavallaee"]["rows"].items():
        for cls in ("total", "attack", "normal"):
            v = classes[cls]
            exact = v["diff"] == [0, 0]
            status = ('<span class="inline-flex items-center gap-1 text-success">' + icon("check") + " exact</span>" if exact
                      else f'<span class="inline-block px-1.5 py-0.5 bg-primary/10 text-primary text-[10px]">{v["diff"][0]:+,} records</span>')
            tav_rows.append(f"""<tr>
  <th scope="row" class="py-2 px-3 font-medium text-left">{e(split_name.capitalize())} · {e(cls)}</th>
  <td class="py-2 px-3 text-right tabular">{v['paper'][0]:,} / {v['paper'][1]:,}</td>
  <td class="py-2 px-3 text-right tabular">{v['ours'][0]:,} / {v['ours'][1]:,}</td>
  <td class="py-2 px-3 text-right">{status}</td>
</tr>""")

    recall = min(v for v in syn["minhash_recall"].values() if v is not None)
    limitations = [
        "A copy is not always a bug: real traffic repeats. Both the original and the cleaned score are reported.",
        "The cleaned test set is harder by construction; the gap measures how much of a score was earned on copies.",
        "Near-copy thresholds (0.01 RMS, Jaccard 0.8) are fixed choices; a curve and a training-set baseline are reported alongside.",
        "Baselines are untuned standard models, and training sets are capped at 300,000 rows.",
        "EMBER is scanned on leakcheck's own numeric view, not the official 2,381-dimension feature vector.",
        "CIC-IDS2017's time split is confounded with unseen attack types, so H3 was decided on EMBER.",
    ]
    lim = "".join(f"<li>{e(x)}</li>" for x in limitations)

    commands = [
        "python -m venv .venv",
        "# then activate it: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate",
        "python -m pip install -e \".[dev]\"",
        "python scripts/fetch.py",
        "python -m experiments.e0_sources && python -m experiments.e0_sources_week4",
        "python -m experiments.e0_synthetic && python -m experiments.e0_kdd",
        "python -m experiments.e1_audit",
        "python -m experiments.e2_retrain kdd99 nslkdd unsw_nb15 cicids2017 ember2018 uci_phishing phiusiil sms_spam",
        "python -m experiments.e4_ablation phiusiil cicids2017",
        "python scripts/report.py",
    ]
    shown = ["# 1. Install, then fetch every dataset (SHA-256 verified)", *commands[:4],
             "# 2. Validate, audit, retrain, ablate", *commands[4:9],
             "# 3. Regenerate the report and this page's numbers", commands[9]]
    code_lines = "".join(
        (f'<span class="text-[#8c8c88]">{e(c)}</span>\n' if c.startswith("#")
         else f'<span class="text-[#f59e0b] font-bold">&gt;</span> {e(c)}\n') for c in shown)
    commands = "\n".join(commands)

    return head("leakcheck: how much of a 99% score is the test set copying the training set?",
                "An audit of eight security ML benchmarks for train/test leakage, with an in-browser scanner.") + nav("report") + f"""
<article class="max-w-[1100px] mx-auto px-margin-mobile md:px-gutter py-space-xl flex flex-col gap-space-2xl">

<header class="grid grid-cols-1 lg:grid-cols-12 gap-space-xl items-start pt-space-md">
  <div class="lg:col-span-7 flex flex-col gap-space-lg">
    <div class="inline-flex items-center gap-space-xs self-start"><span class="w-2 h-2 rounded-full bg-primary"></span><span class="sec-label text-on-surface-variant tracking-widest">Security ML · dataset audit</span></div>
    <h1 class="font-headline-lg text-[34px] leading-[1.15] md:text-headline-lg text-on-surface tracking-tight">How much of a 99% score is the test set copying the training set?</h1>
    <p class="font-body-lg text-body-lg text-on-surface-variant max-w-xl">leakcheck finds duplicated rows, contradictory labels, future data and one-column shortcuts across train/test splits, then retrains the standard models without them.</p>
    <div class="flex flex-wrap items-center gap-space-md pt-space-xs">
      <a class="inline-flex items-center justify-center h-10 px-space-lg bg-ink text-on-ink font-mono-data-sm text-mono-data-sm hover:opacity-90 transition-opacity" href="#audit">Read the findings</a>
      <a class="inline-flex items-center justify-center h-10 px-space-lg bg-surface border border-outline-variant text-on-surface font-mono-data-sm text-mono-data-sm hover:bg-surface-container-low transition-colors" href="scan.html">{icon("upload", "mr-1.5 text-primary")} Scan your own split</a>
      <a class="inline-flex items-center justify-center h-10 px-space-md text-on-surface-variant font-mono-data-sm text-mono-data-sm hover:text-on-surface" href="#reproduce"><span class="mr-1.5 text-primary">&gt;_</span>Reproduce it</a>
    </div>
  </div>
  <figure class="lg:col-span-5 flex flex-col bg-surface-container-low p-space-md border border-surface-container-high">
    <figcaption class="flex items-center justify-between pb-space-xs border-b border-surface-container-high mb-space-sm gap-2">
      <span class="sec-label text-on-surface-variant">KDD Cup 99 · real rows</span>
      <span class="font-mono-data-sm text-mono-data-sm text-primary font-medium">same features, both sides</span>
    </figcaption>
    <div class="grid grid-cols-2 gap-space-md font-mono-data-sm text-[10px] leading-tight text-on-surface-variant py-1">
      <div class="flex flex-col gap-1.5 min-w-0"><div class="sec-label pb-1 text-on-surface border-b border-surface-container-high">TRAIN · sample of {kx['n_train']:,}</div>{"".join(tr_rows)}</div>
      <div class="flex flex-col gap-1.5 min-w-0"><div class="sec-label pb-1 text-on-surface border-b border-surface-container-high">TEST · sample of {kx['n_test']:,}</div>{"".join(te_rows)}</div>
    </div>
    <p class="font-body-sm text-body-sm text-on-surface-variant mt-space-xs">Row numbers from the KDD sample you can load in the <a class="underline underline-offset-2 hover:text-on-surface" href="scan.html">scanner</a>.</p>
    <div class="mt-space-md pt-space-sm border-t border-surface-container-high flex flex-col bg-surface-container-lowest p-space-md">
      <div class="flex items-baseline justify-between gap-2"><span class="font-mono-data-lg text-[44px] leading-none text-primary font-bold tracking-tight tabular">{fpct(kdd['exact_copies'])}</span><span class="sec-label bg-primary/10 text-primary px-2 py-0.5">exact copies</span></div>
      <p class="font-body-sm text-body-sm text-on-surface-variant mt-2">of KDD Cup 99's test rows have exactly the same features as a training row. Removing leaked rows costs up to {fpt(kdd_gap)} macro-F1 points.</p>
    </div>
  </figure>
</header>

<section aria-label="Summary" class="grid grid-cols-1 md:grid-cols-3 bg-surface-container-low border-y border-surface-container-high py-space-sm divide-y md:divide-y-0 md:divide-x divide-surface-container-high">
  <div class="py-space-xs md:py-0 px-space-md flex flex-col justify-center"><span class="font-mono-data-md text-mono-data-md text-on-surface font-medium">{d['summary']['datasets']} benchmarks audited</span><span class="font-mono-data-sm text-mono-data-sm text-secondary tabular">{d['summary']['rows_audited']:,} rows</span></div>
  <div class="py-space-xs md:py-0 px-space-md flex flex-col justify-center"><span class="font-mono-data-md text-mono-data-md text-primary font-semibold">{d['summary']['datasets_with_5pct_exact_copies']} of {d['summary']['datasets']} leak ≥ 5% of their test set</span><span class="font-mono-data-sm text-mono-data-sm text-secondary">exact copies of training rows</span></div>
  <div class="py-space-xs md:py-0 px-space-md flex flex-col justify-center"><span class="font-mono-data-md text-mono-data-md text-on-surface font-medium">{syn_err} errors on {syn['configurations']} planted-leak tests</span><span class="font-mono-data-sm text-mono-data-sm text-secondary">synthetic validation</span></div>
</section>

<section class="grid grid-cols-1 lg:grid-cols-12 gap-space-xl">
  <div class="lg:col-span-5 flex flex-col gap-space-md">
    {sec_head("01", "The problem", "A high score can be a lookup, not a detector")}
    <div class="font-body-md text-body-md text-on-surface-variant flex flex-col gap-space-sm">
      <p>Security benchmarks make high scores easy: on a random split of CIC-IDS2017, untuned baseline models reach {min(cic_orig) * 100:.1f}–{max(cic_orig) * 100:.1f}% macro-F1. Many benchmarks put identical or near-identical rows on both sides of the train/test split, mix future data into training, or contain a single column that gives the label away.</p>
      <p>A model can then score by recognising rows it has already seen. That says little about how it handles traffic or files it has not seen, which is the thing a deployment needs.</p>
    </div>
    <div class="p-space-md bg-surface-container-low border-l-2 border-primary">
      <div class="font-mono-data-sm text-mono-data-sm text-on-surface font-semibold">What leakcheck measures</div>
      <div class="font-body-sm text-body-sm text-on-surface-variant mt-1">How much of a benchmark's score is earned on rows the model has already seen, and how much remains once they are removed.</div>
    </div>
  </div>
  <div class="lg:col-span-7 flex flex-col gap-space-md">
    {sec_head("02", "What it checks", "Six checks on every split")}
    <div class="flex flex-col divide-y divide-surface-container-high border-y border-surface-container-high">{checks}</div>
    <p class="font-body-sm text-body-sm text-on-surface-variant"><code class="font-mono-data-sm text-mono-data-sm text-on-surface">leakcheck clean</code> removes the leakage, then re-reads its own output and re-checks every row. If anything remains, it fails.</p>
  </div>
</section>

<section class="flex flex-col gap-space-md">
  <div class="flex flex-col md:flex-row md:items-end justify-between border-b border-surface-container-high pb-space-sm gap-2">
    {sec_head("03", "The audit", "Eight benchmarks, one audit", "audit")}
    <div class="font-mono-data-sm text-mono-data-sm text-secondary">Highlighted: exact copies ≥ 5% · single feature ≥ 95%</div>
  </div>
  <div class="w-full overflow-x-auto" tabindex="0" role="region" aria-label="Audit table, scrolls sideways">
    <table class="w-full text-left border-collapse border border-surface-container-high min-w-[860px] bg-surface font-mono-data-sm text-mono-data-sm text-on-surface">
      <caption class="sr-only">leakcheck audit of eight security benchmarks</caption>
      <thead><tr class="bg-surface-container-low border-b border-on-surface sec-label text-secondary">
        <th scope="col" class="py-space-sm px-space-md">Dataset</th><th scope="col" class="py-space-sm px-space-md">Domain</th><th scope="col" class="py-space-sm px-space-md">Split</th>
        <th scope="col" class="py-space-sm px-space-md text-right">Test rows</th><th scope="col" class="py-space-sm px-space-md text-right">Exact copies</th>
        <th scope="col" class="py-space-sm px-space-md text-right">Near copies (× train baseline)</th><th scope="col" class="py-space-sm px-space-md text-right">Best single feature</th>
      </tr></thead>
      <tbody class="divide-y divide-surface-container-high">{"".join(rows)}</tbody>
    </table>
  </div>
  <p class="font-body-sm text-body-sm text-on-surface-variant">× train baseline above 1 means test rows sit closer to training rows than training rows sit to each other. Best single feature is the balanced accuracy of a small tree on that one column.</p>
</section>

<section class="flex flex-col gap-space-md">
  {sec_head("04", "What the leakage is worth", "Same copying, very different damage")}
  <div class="p-space-md bg-surface-container border border-surface-container-high">
    <p class="font-body-lg text-body-lg text-on-surface font-medium">KDD Cup 99 loses {fpt(kdd_gap)} macro-F1 points when its leaked test rows are removed. UCI Phishing, with as much copying, loses {fpt(uci_gap)}.</p>
  </div>
  <div class="bg-surface-container-low p-space-md md:p-space-lg border border-surface-container-high flex flex-col gap-space-md">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-space-xs border-b border-surface-container-high font-mono-data-sm text-mono-data-sm">
      <span class="text-secondary sec-label">Macro-F1 points lost</span>
      <div class="flex items-center gap-space-md"><span class="flex items-center gap-1.5"><span class="w-3 h-3 bg-primary"></span>Logistic regression</span><span class="flex items-center gap-1.5"><span class="w-3 h-3 bg-bar"></span>Gradient boosting</span></div>
    </div>
    <div class="flex flex-col gap-3 font-mono-data-sm text-mono-data-sm">{"".join(bars)}</div>
    <div class="grid grid-cols-12 gap-2 pt-space-xs border-t border-surface-container-high font-mono-data-sm text-[10px] text-secondary">
      <span class="col-span-4 sm:col-span-3"></span><div class="col-span-6 sm:col-span-7 flex justify-between"><span>0</span><span>5</span><span>10</span><span>15</span><span>20 pts</span></div><span class="col-span-2"></span>
    </div>
  </div>
  <p class="font-body-sm text-body-sm text-on-surface-variant">Four untuned baselines × 5 seeds; this chart shows two of them. All four models, with seed standard deviations, are in <a class="underline underline-offset-2 hover:text-on-surface" href="{REPO_URL}/blob/main/docs/RESULTS.md" rel="noopener noreferrer">RESULTS.md</a>.</p>
</section>

<section class="flex flex-col gap-space-md">
  {sec_head("05", "Pre-registered", "Five hypotheses, decided by rules fixed before the runs", "hypotheses")}
  <div class="flex flex-col border border-surface-container-high divide-y divide-surface-container-high bg-surface">{"".join(hyps)}</div>
  <p class="font-mono-data-sm text-mono-data-sm text-secondary">Each decision rule was committed to <a class="underline underline-offset-2 hover:text-on-surface" href="{REPO_URL}/blob/main/PLAN.md" rel="noopener noreferrer">PLAN.md</a> before the experiment it judges was run.</p>
</section>

<section class="flex flex-col gap-space-md">
  {sec_head("06", "What we learned", "Four findings")}
  <div class="grid grid-cols-1 md:grid-cols-2 gap-space-lg">{finding_cards}</div>
</section>

<section class="flex flex-col gap-space-md">
  {sec_head("07", "Validation", "Checked against the literature")}
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-space-xl items-start">
    <div class="lg:col-span-7 flex flex-col gap-space-xs">
      <p class="font-body-sm text-body-sm text-on-surface-variant">KDD Cup 99 records / distinct records, against Tables I and II of Tavallaee et al. (2009), the paper that introduced NSL-KDD:</p>
      <div class="border border-surface-container-high bg-surface overflow-x-auto">
        <table class="w-full text-left border-collapse font-mono-data-sm text-mono-data-sm text-on-surface min-w-[520px]">
          <thead><tr class="bg-surface-container-low border-b border-on-surface sec-label text-secondary"><th scope="col" class="py-2 px-3">Slice</th><th scope="col" class="py-2 px-3 text-right">Paper</th><th scope="col" class="py-2 px-3 text-right">leakcheck</th><th scope="col" class="py-2 px-3 text-right">Status</th></tr></thead>
          <tbody class="divide-y divide-surface-container-high">{"".join(tav_rows)}</tbody>
        </table>
      </div>
      <p class="font-body-sm text-body-sm text-on-surface-variant">The two-record difference is in the paper's test-set normal count (60,591); the published file has 60,593.</p>
    </div>
    <div class="lg:col-span-5 flex flex-col gap-space-sm bg-surface-container-low p-space-md border border-surface-container-high">
      <span class="sec-label text-on-surface-variant pb-space-xs border-b border-surface-container-high">Synthetic validation</span>
      <div class="flex flex-col py-space-xs"><span class="font-mono-data-lg text-[32px] leading-tight text-on-surface font-bold tabular">{syn['configurations'] if syn_err == 0 else '—'} / {syn['configurations']}</span><span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant">planted-leak configurations recovered with zero counting error</span></div>
      <div class="flex flex-col py-space-xs border-t border-surface-container-high"><span class="font-mono-data-lg text-[32px] leading-tight text-on-surface font-bold tabular">{syn['interval_coverage'] * 100:.0f}%</span><span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant">of 95% intervals cover the true near-copy rate</span></div>
      <div class="flex flex-col py-space-xs border-t border-surface-container-high"><span class="font-mono-data-lg text-[32px] leading-tight text-primary font-bold tabular">{recall * 100:.0f}%</span><span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant">text near-copy recall at Jaccard ≥ 0.8, zero false positives</span></div>
    </div>
  </div>
</section>

<section class="flex flex-col gap-space-md" aria-labelledby="reproduce-title">
  <div id="reproduce" class="scroll-mt-20">
    <div class="flex items-center gap-space-xs"><span class="sec-label text-primary">08</span><span class="w-1.5 h-1.5 bg-outline-variant"></span><span class="sec-label text-on-surface-variant">Reproduce</span></div>
    <h2 id="reproduce-title" class="font-headline-md text-headline-md text-on-surface mt-1">Reproduce every number</h2>
  </div>
  <div class="bg-[#111110] text-[#f4f4f1] p-space-md md:p-space-lg border border-[#2f312f]">
    <div class="flex items-center justify-between pb-space-sm border-b border-[#2f312f] mb-space-md">
      <span class="font-mono-data-sm text-mono-data-sm text-[#8c8c88]">bash · laptop CPU, no GPU</span>
      <button type="button" data-copy="#reproduce-code" class="font-mono-data-sm text-mono-data-sm text-[#dadad7] hover:text-white px-2 py-1 bg-[#2f312f] border border-[#474745] inline-flex items-center gap-1">{icon("copy")}<span>Copy</span></button>
    </div>
    <pre class="font-mono-data-sm text-mono-data-sm text-[#e2e3e0] overflow-x-auto leading-relaxed"><code>{code_lines}</code></pre>
    <pre id="reproduce-code" hidden>{e(commands)}</pre>
    <div class="mt-space-md pt-space-sm border-t border-[#2f312f] flex flex-col md:flex-row md:items-center justify-between gap-1 font-mono-data-sm text-mono-data-sm text-[#8c8c88]">
      <span>Every run appends a line to an append-only ledger; this page is generated from it.</span>
      <span class="text-white font-medium">{len(d['ledger_lines'])} ledger lines behind this page</span>
    </div>
  </div>
</section>

<section class="flex flex-col gap-space-md pt-space-md border-t border-surface-container-high">
  {sec_head("09", "Scope", "Limitations", level="h2")}
  <ul class="grid grid-cols-1 md:grid-cols-2 gap-x-space-xl gap-y-space-xs font-body-sm text-body-sm text-on-surface-variant list-disc pl-4">{lim}</ul>
</section>

</article>
""" + footer(generated)


# --- scan.html ---------------------------------------------------------------------------


def build_scan(ex: dict, generated: str) -> str:
    chips = []
    for key in ("kdd99", "uci_phishing", "control"):
        x = ex[key]
        leak = x["exact_copies"]
        dot = "bg-primary" if leak >= 0.05 else "bg-secondary"
        tag = f'<span class="{"text-primary font-semibold" if leak >= 0.05 else "text-on-surface-variant"}">({fpct(leak)} copies)</span>'
        chips.append(f'<button type="button" data-example="{key}" class="px-space-sm py-1 bg-surface-container border border-outline-variant '
                     f'hover:border-outline text-on-surface flex items-center gap-1.5 transition-colors"><span class="w-1.5 h-1.5 {dot}"></span>'
                     f'<span>{e(x["name"])}</span>{tag}</button>')

    def dropzone(which: str) -> str:
        name = which.upper()
        return f"""<div class="dropzone bg-surface-container-lowest border border-dashed border-outline-variant p-space-lg flex flex-col justify-between gap-space-md transition-colors hover:border-outline min-h-[168px]" data-drop="{which}">
  <div class="flex items-start justify-between gap-2">
    <span class="sec-label px-1.5 py-0.5 bg-surface-container-high text-on-surface tracking-wider">{name} SPLIT</span>
    <span class="hidden items-center text-success font-mono-data-sm text-mono-data-sm gap-1" data-loaded-badge>{icon("check")} Loaded</span>
  </div>
  <label class="flex flex-col items-center justify-center text-center gap-1 cursor-pointer py-space-sm" data-empty>
    <span class="text-on-surface-variant text-[22px]">{icon("upload")}</span>
    <span class="font-mono-data-sm text-mono-data-sm text-on-surface">Drop a CSV here or <span class="underline underline-offset-2">choose a file</span></span>
    <input type="file" accept=".csv,text/csv" class="sr-only" data-input="{which}" aria-label="Choose the {which} CSV">
  </label>
  <div class="hidden flex-col gap-space-sm" data-filled>
    <div class="flex items-center gap-space-md">
      <div class="w-10 h-10 bg-surface-container flex items-center justify-center text-on-surface shrink-0 border border-outline-variant text-[20px]">{icon("file")}</div>
      <div class="flex flex-col min-w-0">
        <span class="font-mono-data-md text-mono-data-md font-medium text-on-surface truncate" data-field="name"></span>
        <span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant tabular" data-field="shape"></span>
      </div>
    </div>
    <div class="flex items-center justify-between gap-2 text-on-surface-variant font-mono-data-sm text-mono-data-sm border-t border-surface-container pt-space-xs">
      <span class="truncate" data-field="sha"></span>
      <button type="button" class="underline underline-offset-4 hover:text-on-surface shrink-0" data-replace="{which}">replace</button>
    </div>
  </div>
</div>"""

    def select(field: str, label: str, none: bool) -> str:
        return f"""<div class="flex flex-col gap-1">
  <label for="opt-{field}" class="sec-label text-on-surface-variant">{label}</label>
  <select id="opt-{field}" data-opt="{field}" class="bg-surface border border-outline-variant px-space-sm py-1.5 font-mono-data-sm text-mono-data-sm text-on-surface" disabled>
    {'<option value="">None</option>' if none else '<option value="">Choose…</option>'}
  </select>
</div>"""

    return head("Scan your own split · leakcheck", "Check a train/test split for leakage in your browser. Files never leave the page.") + nav("scanner") + f"""
<div class="w-full max-w-[1100px] mx-auto px-margin-mobile md:px-gutter py-space-xl flex flex-col gap-space-2xl">

<section class="flex flex-col gap-space-md">
  <div class="flex flex-wrap items-center justify-between gap-space-sm">
    <div class="flex items-center gap-space-xs"><span class="sec-label text-primary tracking-wider">Scanner</span><span class="text-outline-variant font-mono-data-sm">/</span><span class="sec-label text-on-surface-variant tracking-wider">runs in this tab</span></div>
    <div class="flex items-center gap-1.5 px-space-sm py-1 bg-surface-container border border-outline-variant font-mono-data-sm text-mono-data-sm text-on-surface"><span class="w-2 h-2 rounded-full bg-success"></span><span>JavaScript engine · checks L1–L6</span></div>
  </div>
  <div class="flex flex-col md:flex-row md:items-baseline justify-between gap-space-md">
    <h1 class="font-display-editorial text-[40px] leading-tight md:text-display-editorial text-on-surface">Scan your own split</h1>
    <p class="font-mono-data-sm text-mono-data-sm text-on-surface-variant max-w-sm md:text-right">Exact and near copies, label conflicts, time leakage, one-column shortcuts and group overlap between a training and a test file.</p>
  </div>
  <div class="bg-surface-container-low border border-outline-variant p-space-md flex items-start gap-space-md">
    <div class="p-1.5 bg-surface-container-high border border-outline-variant text-on-surface shrink-0 text-[18px]">{icon("lock")}</div>
    <div class="flex flex-col gap-1">
      <span class="font-mono-data-sm text-mono-data-sm font-medium text-on-surface">Your files never leave this page.</span>
      <p class="font-body-md text-body-md text-on-surface-variant">The page's Content-Security-Policy sets <code class="font-mono-data-sm bg-surface-container px-1 text-on-surface">connect-src 'none'</code>, so it cannot make network requests, and the scan runs in a worker that inherits that policy. Check it yourself: open your browser's network panel and run a scan.</p>
    </div>
  </div>
</section>

<section class="flex flex-col gap-space-md" aria-labelledby="step1">
  <div class="flex flex-wrap items-center justify-between gap-2">
    <div class="flex items-center gap-space-sm"><span class="w-5 h-5 bg-ink text-on-ink flex items-center justify-center font-mono-data-sm text-[11px] font-bold">1</span><h2 id="step1" class="sec-label text-on-surface font-medium">Load the two files</h2></div>
    <span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant">CSV with a header row · same columns in both</span>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-space-md">{dropzone("train")}{dropzone("test")}</div>
  <div class="flex flex-wrap items-center gap-space-sm pt-space-xs font-mono-data-sm text-mono-data-sm">
    <span class="font-body-sm text-body-sm text-on-surface-variant">Or try an example:</span>{"".join(chips)}
  </div>
  <p class="font-body-sm text-body-sm text-on-surface-variant" data-example-note hidden></p>
</section>

<section class="flex flex-col gap-space-sm" aria-labelledby="step2">
  <div class="flex flex-wrap items-center justify-between gap-2">
    <div class="flex items-center gap-space-sm"><span class="w-5 h-5 bg-ink text-on-ink flex items-center justify-center font-mono-data-sm text-[11px] font-bold">2</span><h2 id="step2" class="sec-label text-on-surface font-medium">Tell it about the columns</h2></div>
    <span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant" data-schema>Load both files to continue</span>
  </div>
  <div class="bg-surface-container-low border border-outline-variant p-space-md flex flex-col gap-space-md">
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-space-md">
      {select("label", "Label column", False)}{select("time", "Time column (optional)", True)}{select("group", "Group column (optional)", True)}{select("text", "Text column (optional)", True)}
    </div>
    <div class="border-t border-surface-container-high pt-space-sm flex flex-col gap-space-sm">
      <span class="sec-label text-on-surface-variant">Leave out of the features (IDs, the attack name, anything that encodes the label)</span>
      <div class="flex flex-wrap gap-1.5" data-ignore-list><span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant">—</span></div>
    </div>
    <div class="border-t border-surface-container-high pt-space-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-space-md">
      <label class="flex items-center gap-space-sm font-mono-data-sm text-mono-data-sm text-on-surface-variant">Near-copy check on
        <select data-opt="sample" class="bg-surface border border-outline-variant px-1.5 py-1 text-on-surface">
          <option value="1000">1,000</option><option value="2000" selected>2,000</option><option value="5000">5,000</option><option value="1000000000">all</option>
        </select> test rows</label>
      <div class="flex items-center gap-space-sm w-full md:w-auto justify-end">
        <span class="hidden sm:inline font-mono-data-sm text-mono-data-sm text-on-surface-variant"><kbd class="px-1.5 py-0.5 bg-surface border border-outline-variant text-[10px]">Ctrl</kbd> + <kbd class="px-1.5 py-0.5 bg-surface border border-outline-variant text-[10px]">Enter</kbd></span>
        <button type="button" data-run disabled class="w-full sm:w-auto bg-ink text-on-ink px-space-lg py-2 font-mono-data-sm text-mono-data-sm font-medium flex items-center justify-center gap-space-sm border border-ink disabled:opacity-40 disabled:cursor-not-allowed">{icon("play")}<span>Run scan</span></button>
      </div>
    </div>
  </div>
</section>

<section class="bg-surface-container-low border border-outline-variant p-space-md flex-col gap-space-sm hidden" data-progress aria-live="polite">
  <div class="flex items-center justify-between font-mono-data-sm text-mono-data-sm"><span class="text-on-surface font-medium" data-progress-title>Scanning</span><span class="text-primary font-semibold tabular" data-progress-pct>0%</span></div>
  <div class="w-full bg-surface-container-high h-1.5 overflow-hidden"><div class="bg-primary h-full w-0 transition-all duration-200" data-progress-bar></div></div>
  <div class="flex items-center gap-space-sm font-mono-data-sm text-mono-data-sm text-on-surface-variant">{icon("terminal", "text-primary")}<span class="truncate" data-progress-log></span></div>
</section>

<section class="border-l-2 border-error bg-error/5 p-space-md font-mono-data-sm text-mono-data-sm text-on-surface hidden" data-error role="alert"></section>

<section class="flex-col gap-space-lg border-t border-outline-variant pt-space-xl hidden" data-results aria-live="polite">
  <div class="flex flex-wrap items-center justify-between gap-2">
    <div class="flex items-center gap-space-sm"><span class="w-2.5 h-2.5 bg-primary"></span><h2 class="sec-label text-on-surface tracking-wider">Findings</h2></div>
    <span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant" data-results-meta></span>
  </div>
  <div class="border-2 p-space-lg flex flex-col gap-space-md" data-banner>
    <div class="flex flex-wrap items-center justify-between gap-space-sm"><span class="px-2 py-0.5 sec-label tracking-wider" data-banner-pill></span><span class="font-mono-data-sm text-mono-data-sm text-on-surface-variant" data-banner-meta></span></div>
    <h3 class="font-headline-md text-headline-md text-on-surface" data-banner-title></h3>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-space-md border-t border-outline-variant/40 pt-space-md" data-scoreboard></div>
  </div>
  <div class="flex flex-col gap-space-sm">
    <div class="flex items-center justify-between pb-space-xs border-b border-on-surface"><span class="sec-label text-on-surface-variant">Every check</span><span class="sec-label text-on-surface-variant" data-count></span></div>
    <div class="flex flex-col gap-space-sm" data-cards></div>
  </div>
  <div class="border-t border-outline-variant pt-space-lg flex flex-col gap-space-sm">
    <div class="flex flex-wrap items-center gap-space-sm">
      <button type="button" data-download-report class="bg-surface-container hover:bg-surface-container-high text-on-surface px-space-md py-2 font-mono-data-sm text-mono-data-sm font-medium flex items-center gap-2 border border-outline-variant">{icon("download")}<span>Download report (JSON)</span></button>
      <button type="button" data-download-clean class="bg-ink text-on-ink px-space-md py-2 font-mono-data-sm text-mono-data-sm font-medium flex items-center gap-2 border border-ink">{icon("download")}<span>Download test set without exact copies (CSV)</span></button>
    </div>
    <p class="font-body-sm text-body-sm text-on-surface-variant" data-clean-note>The cleaned file is re-read and re-checked before you get it. To also remove near-copies across every row, use <code class="font-mono-data-sm">leakcheck clean</code> from the Python package.</p>
  </div>
</section>

</div>
<script type="module" src="assets/scan.js"></script>
""" + footer(generated)


HEADER_CSP = CSP + "; frame-ancestors 'none'"  # frame-ancestors only works as an HTTP header
HEADERS = {"Content-Security-Policy": HEADER_CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


def write_host_headers() -> None:
    """Serve the same policy as an HTTP header on Vercel, Netlify or Cloudflare Pages."""
    rules = [{"key": k, "value": v} for k, v in HEADERS.items()]
    # the site is prebuilt: Vercel serves the folder as is, with no install or build step
    config = {"framework": None, "installCommand": "", "buildCommand": "", "outputDirectory": ".",
              "headers": [{"source": "/(.*)", "headers": rules}]}
    (SITE / "vercel.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (SITE / "_headers").write_text("/*\n" + "".join(f"  {k}: {v}\n" for k, v in HEADERS.items()), encoding="utf-8")
    ignore = ["node_modules", "src", "package.json", "package-lock.json", "tailwind.config.cjs", "engine/cli.mjs",
              "examples/examples.json", ".env*", ".vercel"]
    (SITE / ".vercelignore").write_text("\n".join(ignore) + "\n", encoding="utf-8")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true", help="rebuild the example datasets")
    ap.add_argument("--skip-css", action="store_true")
    a = ap.parse_args()
    data = json.loads((SITE / "data.json").read_text(encoding="utf-8"))
    ex = build_examples(force=a.examples)
    generated = _ledger_date()
    (SITE / "index.html").write_text(build_index(data, ex, generated), encoding="utf-8")
    (SITE / "scan.html").write_text(build_scan(ex, generated), encoding="utf-8")
    write_host_headers()
    print("wrote site/index.html, site/scan.html, site/vercel.json, site/_headers")
    if not a.skip_css:
        npx = "npx.cmd" if sys.platform == "win32" else "npx"
        subprocess.run([npx, "tailwindcss", "-c", "tailwind.config.cjs", "-i", "src/input.css", "-o", "assets/site.css",
                        "--minify"], cwd=SITE, check=True)
        print("wrote site/assets/site.css")


def _ledger_date() -> str:
    last = ""
    for line in open(ROOT / "results" / "ledger.jsonl", encoding="utf-8"):
        last = max(last, json.loads(line)["time"])
    return last[:10]


if __name__ == "__main__":
    main()
