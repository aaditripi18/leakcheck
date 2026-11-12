"""Complete clean-split lists for the three largest benchmarks, on a Kaggle GPU.

    python scripts/kaggle_clean.py push-ember          EMBER's archive as a private Kaggle dataset
    python scripts/kaggle_clean.py run cseids2018 ...  code from HEAD, one GPU notebook per case, pulled when done

Each notebook runs `experiments.clean_splits make <case>` with LEAKCHECK_NEAR=torch (the GPU
search, validated against near.nearest: tests/test_near_torch.py), so every test row is
searched, not a sample. The data come from Kaggle copies already verified byte for byte
(CSE-CIC-IDS2018: solarmainframe/ids-intrusion-csv, identical to the official S3 objects;
CIC-IoT2023: madhavmalhotra/unb-cic-iot-dataset) and EMBER's own archive; the loaders check
every file's SHA-256 against data/registry.json before use.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("kaggle_e14", ROOT / "scripts" / "kaggle_e14.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

EMBER = "leakcheck-ember2018"
SOURCES = {"cseids2018": ["solarmainframe/ids-intrusion-csv"],
           "cseids2018:no_identifiers": ["solarmainframe/ids-intrusion-csv"],
           "ciciot2023": ["madhavmalhotra/unb-cic-iot-dataset"],
           "ember2018": [f"{k.USER}/{EMBER}"],
           "botiot": [f"{k.USER}/{k.DATA}"]}  # phase 4's private copy of the fresh data (hash-checked)
DATASET_OF = {"cseids2018": "cseids2018", "cseids2018:no_identifiers": "cseids2018", "ciciot2023": "ciciot2023",
              "ember2018": "ember2018", "botiot": "botiot"}

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
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
reg = json.loads((work / "data" / "registry.json").read_text(encoding="utf-8"))
names = {}
for p in inp.rglob("*"):
    if p.is_file() and work.name not in p.parts:
        names.setdefault(p.name, p)
for f in reg[DATASET]["files"]:
    base = pathlib.PurePosixPath(f["path"]).name
    dst = work / "data" / "raw" / f["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(names[base], dst)   # the loaders check the SHA-256
env = {**os.environ, "LEAKCHECK_NEAR": "torch", "PYTHONUNBUFFERED": "1"}
r = subprocess.run([sys.executable, "-u", "-m", "experiments.clean_splits", "make", CASE], cwd=work, env=env)
out = pathlib.Path("/kaggle/working")
slug = CASE.replace(":", "__")
for suffix in (".json", ".drop.txt.gz"):
    f = work / "splits" / (slug + suffix)
    if f.exists():
        shutil.copy2(f, out / f.name)
(out / "failed.txt").write_text("" if r.returncode == 0 else f"{CASE}: exit {r.returncode}\n", encoding="utf-8")
shutil.rmtree(work)
print("exit", r.returncode, CASE, flush=True)
'''


def kernel(case: str) -> str:
    return "leakcheck-clean-" + case.replace(":", "-").replace("_", "-")


def notebook_source(case: str, sha: str) -> str:
    return (f"# leakcheck clean split {case}, code {sha}\nCASE = {case!r}\nSHA = {sha!r}\nPINS = {k.PINS!r}\n"
            f"DATASET = {DATASET_OF[case]!r}\n" + NOTEBOOK)


def push_ember() -> None:
    sys.path.insert(0, str(ROOT))
    from experiments.data import verify_local
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for p in verify_local("ember2018"):
            shutil.copy2(p, d / p.name)
        k._upload(EMBER, d, "EMBER 2018 archive")
    print(f"uploaded {k.USER}/{EMBER}")


def start(case: str, sha: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(notebook_source(case, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{k.USER}/{kernel(case)}", "title": kernel(case), "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{k.USER}/{k.CODE}", *SOURCES[case]],
            "competition_sources": [], "kernel_sources": []}), encoding="utf-8")
        r = k.kaggle("kernels", "push", "-p", str(d), check=False)
    out = (r.stdout + r.stderr).strip()
    if "successfully pushed" in out:
        print(f"{time.strftime('%H:%M')} started {case}", flush=True)
        return True
    if "GPU quota" in out:
        print(f"{time.strftime('%H:%M')} {case}: weekly GPU quota used up; trying again in an hour", flush=True)
        return False
    if any(t in out for t in ("Maximum batch", "session count", "concurrent", "timed out", "Connection",
                              "Max retries", "503", "502", "getaddrinfo")):
        return False
    sys.exit(f"{time.strftime('%H:%M')} FAILED to start {case}: {out}")


def pull(case: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        k.kaggle("kernels", "output", f"{k.USER}/{kernel(case)}", "-p", d, "--file-pattern",
                 r".*(\.json|\.gz|\.txt|\.log)$", check=False)
        dest = ROOT / "results" / "kaggle" / kernel(case)
        dest.mkdir(parents=True, exist_ok=True)
        for f in Path(d).rglob("*"):
            if f.is_file():
                target = ROOT / "splits" / f.name if f.name.endswith((".drop.txt.gz",)) or (
                    f.suffix == ".json" and f.name != "kernel-metadata.json") else dest / f.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    failed = (dest / "failed.txt").read_text(encoding="utf-8").strip() if (dest / "failed.txt").exists() else "missing"
    print(f"{time.strftime('%H:%M')} {case}: pulled; {'FAILED ' + failed if failed else 'ok'}", flush=True)


def run(cases: list[str]) -> None:
    sha = k.push_code()
    pending, running = list(cases), []
    while pending or running:
        for c in list(pending):
            if start(c, sha):
                pending.remove(c)
                running.append(c)
            else:
                break
        time.sleep(3600 if pending and not running else k.WAIT)
        for c in list(running):
            if k.kaggle("kernels", "status", f"{k.USER}/{kernel(c)}", check=False).stdout.count("RUNNING") == 0:
                s = k.kaggle("kernels", "status", f"{k.USER}/{kernel(c)}", check=False).stdout
                if any(t in s for t in ("COMPLETE", "ERROR", "CANCEL")):
                    running.remove(c)
                    pull(c)
    print(f"{time.strftime('%H:%M')} all done: {', '.join(cases)}", flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"push-ember": lambda: push_ember(), "run": lambda: run(args),
     "pull": lambda: [pull(c) for c in args]}.get(cmd, lambda: sys.exit(__doc__))()
