"""The generated pages keep the promises they make."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGES = ["index.html", "scan.html"]


@pytest.fixture(scope="module")
def built():
    if not (SITE / "examples" / "examples.json").exists():
        pytest.skip("examples not built (python scripts/build_site.py --examples)")
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys
    argv = sys.argv
    sys.argv = ["build_site.py", "--skip-css"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    return mod, {p: (SITE / p).read_text(encoding="utf-8") for p in PAGES}


@pytest.mark.parametrize("page", PAGES)
def test_csp_blocks_network(built, page):
    _, html = built
    m = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html[page])
    assert m, "no CSP meta tag"
    csp = m.group(1)
    assert "connect-src 'none'" in csp and "default-src 'none'" in csp and "script-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp


@pytest.mark.parametrize("page", PAGES)
def test_no_inline_code_or_third_party_scripts(built, page):
    _, html = built
    h = html[page]
    assert 'style="' not in h, "inline style attributes are blocked by the CSP"
    assert not re.search(r"\son[a-z]+=", h), "inline event handlers are blocked by the CSP"
    for src in re.findall(r'<script[^>]*src="([^"]+)"', h):
        assert not src.startswith(("http:", "https:", "//")), src
    assert not re.search(r"<script(?![^>]*src=)[^>]*>\s*\S", h), "inline <script> blocked by the CSP"


def test_host_headers_match_meta(built):
    mod, _ = built
    v = json.loads((SITE / "vercel.json").read_text(encoding="utf-8"))
    vals = {h["key"]: h["value"] for h in v["headers"][0]["headers"]}
    assert vals["Content-Security-Policy"] == mod.CSP + "; frame-ancestors 'none'"
    assert mod.HEADER_CSP in (SITE / "_headers").read_text(encoding="utf-8")


def test_numbers_come_from_data(built):
    mod, html = built
    d = json.loads((SITE / "data.json").read_text(encoding="utf-8"))
    idx = html["index.html"]
    for a in d["audit"]:
        assert mod.fpct(a["exact_copies"]) in idx, a["id"]
    verdicts = [h["verdict"] for h in d["hypotheses"]]
    assert idx.count(">SUPPORTED<") == verdicts.count("supported")
    assert idx.count(">NOT SUPPORTED<") == verdicts.count("not supported")
    kdd = next(r for r in d["retrain"] if r["id"] == "kdd99")
    assert f"{kdd['models']['lr']['gap_no_near'] * 100:.1f}" in idx


@pytest.mark.parametrize("bad", ["Pyodide", "e1d4b", "Foundation", "ROC-AUC", "AUROC", "235,685", "181,856",
                                 "Report No.", "Hazard Index", "10⁻⁴"])
def test_no_invented_design_placeholders(built, bad):
    _, html = built
    for page, h in html.items():
        assert bad not in h, f"{bad!r} in {page}"
