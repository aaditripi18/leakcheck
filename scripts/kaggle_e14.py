"""E14's full runs on the fresh cases as Kaggle CPU notebooks, one per case.

    python scripts/kaggle_e14.py push-data              the fresh datasets as a private Kaggle dataset
    python scripts/kaggle_e14.py push-data3             round 3 of H22 (PLAN.md 2026-11-13), likewise
    python scripts/kaggle_e14.py run ton_network ...    code from HEAD, then one notebook per case
                                                       (started as Kaggle allows), pulled when done
    python scripts/kaggle_e14.py pull ton_network ...   new ledger lines into results/ledger.jsonl

Kaggle runs only committed code, and the code includes the committed ledger, so a case's H19
prediction must be committed before its notebook can run `full` (e14_explain refuses otherwise).
Every notebook installs this machine's numpy, pandas, scikit-learn and scipy versions, checks
each data file's SHA-256 against data/registry.json (the loaders do), and records the commit in
the ledger line (LEAKCHECK_COMMIT: there is no git on Kaggle).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "ledger.jsonl"
USER = os.environ.get("LEAKCHECK_KAGGLE_USER", "aadityae5545454")
CODE, DATA, DATA3 = "leakcheck-code", "leakcheck-fresh-data", "leakcheck-round3-data"
FRESH_DATA = ("ton_iot", "botiot", "xiiotid")
ROUND3_DATA = ("uci_bhp", "uci_bitcoin_heist", "uci_firewall", "openml_creditcard", "openml_timing_attack",
               "openml_padding_attack", "openml_ieee_fraud", "openml_phishing_tan")  # H22, PLAN.md 2026-11-13
PINS = ["numpy==2.5.4", "pandas==3.0.6", "scikit-learn==1.9.1", "scipy==1.18.1"]
WAIT = 900


def kaggle(*args: str, check: bool = True, timeout: int = 3600) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle"))
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONUTF8": "1"}, timeout=timeout)
    except subprocess.TimeoutExpired:
        r = subprocess.CompletedProcess([exe, *args], 1, "", f"kaggle call timed out after {timeout} s")
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}{r.stderr}")
    return r


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def kernel(case: str) -> str:
    return "leakcheck-e14-" + case.replace(":", "-").replace("_", "-")


def _upload(slug: str, folder: Path, message: str) -> None:
    ds = f"{USER}/{slug}"
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": ds, "licenses": [{"name": "other"}]}), encoding="utf-8")
    if kaggle("datasets", "status", ds, check=False).returncode == 0:
        kaggle("datasets", "version", "-p", str(folder), "-m", message, timeout=7200)
    else:
        kaggle("datasets", "create", "-p", str(folder), timeout=7200)  # private unless --public


def push_data(slug: str = DATA, names: tuple[str, ...] = FRESH_DATA) -> None:
    """The fresh cases' raw files, flat (their names are unique), checked against the registry."""
    sys.path.insert(0, str(ROOT))
    from experiments.data import verify_local
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name in names:
            for p in verify_local(name):
                os.link(p, d / p.name) if os.name != "nt" else shutil.copy2(p, d / p.name)
        _upload(slug, d, "fresh data")
    print(f"uploaded {USER}/{slug}")


def push_code() -> str:
    if git("status", "--porcelain", "--untracked-files=no", "--", "leakcheck", "experiments", "scripts",
           "data/registry.json", "results/ledger.jsonl"):
        sys.exit("commit first: Kaggle runs only committed code and the committed ledger")
    sha = git("rev-parse", "--short", "HEAD")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        archive = d / f"leakcheck-{sha}.zip"
        subprocess.run(["git", "archive", "--format=zip", "--prefix=leakcheck/", "-o", str(archive), "HEAD"],
                       cwd=ROOT, check=True)
        with zipfile.ZipFile(archive, "a") as z:
            z.writestr("leakcheck/COMMIT", sha + "\n")
        _upload(CODE, d, f"code {sha}")
    for _ in range(90):  # Kaggle can report "ready" before it serves the new version
        with tempfile.TemporaryDirectory() as d:
            r = kaggle("datasets", "download", f"{USER}/{CODE}", "-f", "leakcheck/COMMIT", "-p", d, check=False)
            got = [p for p in Path(d).rglob("*") if p.is_file()]
            if r.returncode == 0 and got:
                data = got[0].read_bytes()
                if data[:2] == b"PK":
                    with zipfile.ZipFile(got[0]) as z:
                        data = z.read(z.namelist()[0])
                if data.decode().strip() == sha:
                    print(f"Kaggle serves code {sha}", flush=True)
                    return sha
        time.sleep(10)
    sys.exit("Kaggle did not serve the new code within 15 minutes")


NOTEBOOK = r'''
import json, os, pathlib, shutil, subprocess, sys
inp = pathlib.Path("/kaggle/input")
src = next(p for p in inp.rglob("COMMIT") if p.parent.name == "leakcheck").parent
got = (src / "COMMIT").read_text().strip()
assert got == SHA, f"code dataset serves {got}, expected {SHA}"
work = pathlib.Path("/kaggle/working/leakcheck")
shutil.copytree(src, work)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(work)], check=True)
reg = json.loads((work / "data" / "registry.json").read_text(encoding="utf-8"))
for name in FRESH_DATA:
    for f in reg[name]["files"]:
        base = pathlib.PurePosixPath(f["path"]).name
        hit = [p for p in inp.rglob(base) if work.name not in p.parts and src not in p.parents]
        dst = work / "data" / "raw" / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(hit[0], dst)   # the loaders check the SHA-256
ledger = work / "results" / "ledger.jsonl"
before = len(ledger.read_text(encoding="utf-8").splitlines())
env = {**os.environ, "LEAKCHECK_COMMIT": SHA, "PYTHONUNBUFFERED": "1"}
r = subprocess.run([sys.executable, "-u", "-m", "experiments.e14_explain", "full", CASE], cwd=work, env=env)
lines = ledger.read_text(encoding="utf-8").splitlines()[before:]
out = pathlib.Path("/kaggle/working")
(out / "new_ledger_lines.jsonl").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
(out / "failed.txt").write_text("" if r.returncode == 0 else f"{CASE}: exit {r.returncode}\n", encoding="utf-8")
shutil.rmtree(work)
print("exit", r.returncode, CASE, len(lines), "ledger lines", flush=True)
'''


def notebook_source(case: str, sha: str) -> str:
    return (f"# leakcheck E14 full {case}, code {sha}\nCASE = {case!r}\nSHA = {sha!r}\nPINS = {PINS!r}\n"
            f"FRESH_DATA = {FRESH_DATA + ROUND3_DATA!r}\n" + NOTEBOOK)


def start(case: str, sha: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(notebook_source(case, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{kernel(case)}", "title": kernel(case), "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USER}/{CODE}", f"{USER}/{DATA}", f"{USER}/{DATA3}"],
            "competition_sources": [], "kernel_sources": []}), encoding="utf-8")
        r = kaggle("kernels", "push", "-p", str(d), check=False)
    out = (r.stdout + r.stderr).strip()
    if "successfully pushed" in out:
        print(f"{time.strftime('%H:%M')} started {case}", flush=True)
        return True
    if any(t in out for t in ("Maximum batch", "session count", "concurrent", "timed out", "Connection",
                              "Max retries", "503", "502", "getaddrinfo")):
        return False  # no free session, or the network: try again later
    sys.exit(f"{time.strftime('%H:%M')} FAILED to start {case}: {out}")


def status(case: str) -> str:
    out = kaggle("kernels", "status", f"{USER}/{kernel(case)}", check=False).stdout
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out:
            return s
    return "UNKNOWN"


def pull(case: str) -> int:
    dest = ROOT / "results" / "kaggle" / kernel(case)
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", f"{USER}/{kernel(case)}", "-p", d, "--file-pattern",
               r".*(\.jsonl|\.txt|\.log)$", check=False)
        for f in Path(d).rglob("*"):
            if f.is_file():
                (dest / f.relative_to(d)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest / f.relative_to(d))
    f = dest / "new_ledger_lines.jsonl"
    new = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()] if f.exists() else []
    have = set(LEDGER.read_text(encoding="utf-8").splitlines())
    add = [ln for ln in new if ln not in have]
    with open(LEDGER, "a", encoding="utf-8") as out:
        out.writelines(ln + "\n" for ln in add)
    failed = (dest / "failed.txt").read_text(encoding="utf-8").strip() if (dest / "failed.txt").exists() else ""
    print(f"{time.strftime('%H:%M')} {case}: appended {len(add)} ledger lines{'; FAILED ' + failed if failed else ''}",
          flush=True)
    return len(add)


def run(cases: list[str]) -> None:
    sha = push_code()
    pending, running = list(cases), []
    while pending or running:
        for c in list(pending):
            if start(c, sha):
                pending.remove(c)
                running.append(c)
            else:
                break
        time.sleep(WAIT)
        for c in list(running):
            s = status(c)
            if s in ("COMPLETE", "ERROR", "CANCEL"):
                running.remove(c)
                if s != "COMPLETE":
                    print(f"{time.strftime('%H:%M')} {c}: Kaggle status {s}", flush=True)
                pull(c)
    print(f"{time.strftime('%H:%M')} all done: {', '.join(cases)}", flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "push-data":
        push_data()
    elif cmd == "push-data3":
        push_data(DATA3, ROUND3_DATA)
    elif cmd == "run":
        run(args)
    elif cmd == "pull":
        for c in args:
            pull(c)
    else:
        sys.exit(__doc__)
