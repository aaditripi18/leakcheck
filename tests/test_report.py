import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_report():
    spec = importlib.util.spec_from_file_location("report", ROOT / "scripts" / "report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_and_site_data_from_ledger(tmp_path, monkeypatch):
    report = _load_report()
    ledger = tmp_path / "ledger.jsonl"
    shutil.copy(ROOT / "results" / "ledger.jsonl", ledger)
    monkeypatch.setattr(report, "LEDGER", ledger)
    monkeypatch.setattr(report, "OUT", tmp_path / "RESULTS.md")
    monkeypatch.setattr(report, "SITE_DATA", tmp_path / "site" / "data.json")
    report.main()

    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert md.index("## Hypothesis verdicts") < md.index("## E1")
    data = json.loads((tmp_path / "site" / "data.json").read_text(encoding="utf-8"))
    verdicts = {h["id"]: h["verdict"] for h in data["hypotheses"]}
    assert verdicts == {"H1": "supported", "H2": "not supported", "H3": "supported", "H4": "supported", "H5": "supported",
                        "H6": "supported", "H7": "supported", "H8": "not supported", "H9": "not supported"}
    for section in ("audit", "retrain", "ablation"):
        assert data[section] and all(len(row["source"]["commit"]) >= 7 for row in data[section])
    # the page's numbers equal the generated report's
    kdd = next(r for r in data["retrain"] if r["id"] == "kdd99")
    assert f"{kdd['models']['lr']['gap_no_near'] * 100:+.2f}" in md


def test_later_verdicts_from_ledger():
    """Phases 4-6 as the committed ledger gives them (PLAN.md amendments 2026-11-09 to 2026-11-20)."""
    report = _load_report()
    runs = report.latest()
    got = {h["id"]: h["verdict"] for h in report.phase4_hypotheses(runs)}
    assert got == {"H19": "not judged", "H20": "not judged", "H19b": "not supported", "H21": "supported",
                   "H22": "not judged", "H23": "not judged", "H24": "not judged"}
    h23 = report.h23(runs)
    assert "right on 2/2" in h23["evidence"] and "Not run" not in h23["evidence"]
    assert "andmal2020:no_identifiers S -0.114 predicted memorised, was memorised" in h23["evidence"]
    assert "50,000-row sample per seed" in h23["evidence"]


def test_kdd99_table_has_no_unmeasured_cell():
    """E0b ran L1 and L3 on KDD99; its L2.near and L5.shortcut cells come from E1 on the same split."""
    report = _load_report()
    runs = report.latest()
    md = "\n".join(report.e0_kdd(runs["e0_kdd"], runs.get("e1_audit:kdd99")))
    assert "not run" not in md
    e1 = runs["e1_audit:kdd99"]["results"]["views"]["all"]["findings"]
    assert runs["e0_kdd"]["results"]["kdd99_leakcheck"]["L1.cross"]["n_hit"] == e1["L1.cross"]["n_hit"]  # same task
    assert "| L2.near | 14.80% (x1.37 of train baseline) [high] (E1) |" in md
