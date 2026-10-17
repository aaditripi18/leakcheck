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
