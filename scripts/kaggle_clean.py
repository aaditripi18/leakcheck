"""Complete clean-split lists for the three largest benchmarks, on a Kaggle GPU.

    python scripts/kaggle_clean.py push-ember          EMBER's archive as a private Kaggle dataset
    python scripts/kaggle_clean.py run cseids2018 ...  code from HEAD, one GPU notebook per case, pulled when done
    python scripts/kaggle_clean.py handoff ciciot2023@8   the parts listed in OWNERS, as notebooks for friends
    python scripts/kaggle_clean.py run ciciot2023@6    the near search in 6 parts (one notebook each, at most
                                                       two at a time), then a final notebook that reads them

A case too large for one 12-hour session runs in parts (experiments/near_parts.py): part i of n
searches every n-th distinct test row and saves the distances; the final notebook mounts the
parts' outputs, so clean_split and its re-check read the distances and search only what is missing.

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
# parts run on friends' Kaggle accounts with this code (the notebook is shared back to USER, who
# needs to share the private code dataset with them); every other job runs on USER's account
OWNERS = {**{f"ciciot2023@{i}/16": "alien5060707" for i in (0, 1, 2)},
          **{f"ciciot2023@{i}/16": "moinuddinhasan8" for i in (3, 4, 5)},
          **{f"ciciot2023@{i}/16": "rohajraj" for i in (6, 7)},
          **{f"ciciot2023@{i}/16": "alubalub" for i in (8, 9)},
          "ciciot2023@10/16": "alien5060707", "ciciot2023@11/16": "moinuddinhasan8",
          **{f"cseids2018:no_identifiers@{i}/8": "rohajraj" for i in (0, 1, 2, 6)},
          **{f"cseids2018:no_identifiers@{i}/8": "alubalub" for i in (3, 4, 5, 7)}}
# assigned by GPU quota left on 2026-11-18 (alien5060707 12h56m, moinuddinhasan8 12h49m, rohajraj 11h03m,
# alubalub 10h48m) against the T4 benchmark (results/kaggle/leakcheck-gpu-bench): a ciciot2023 sixteenth
# is at most 3.3 h, a cseids2018:no_identifiers eighth at most 1.1 h; ciciot2023 parts 10-15 and the final
# notebooks run on USER's account after its weekly quota resets. 2026-11-20, with p0-p9 done (2.8-2.9 h each):
# alien5060707 (4h09m left) and moinuddinhasan8 (4h08m) take p10 and p11; rohajraj (3h00m) and alubalub (2h47m)
# have too little for a 2.9 h part
HANDOFF = ROOT.parent / "leakcheck_handoff"
PULL_RETRY = 60  # seconds between download attempts
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
        names.setdefault(p.name[:-4] if p.name.endswith(".bin") else p.name, p)  # <name>.bin: see push_ember
for f in reg[DATASET]["files"]:
    base = pathlib.PurePosixPath(f["path"]).name
    dst = work / "data" / "raw" / f["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(names[base], dst)   # the loaders check the SHA-256
env = {**os.environ, "LEAKCHECK_NEAR": "torch", "PYTHONUNBUFFERED": "1"}
cache = work / "near_cache"
if PART is not None:
    cache.mkdir()
    env["LEAKCHECK_NEAR_CACHE"] = str(cache)
    if PART == "final":
        for p in inp.rglob("*.npz"):
            shutil.copy2(p, cache / p.name)
        print("parts read:", sorted(p.name for p in cache.iterdir()), flush=True)
    else:
        env["LEAKCHECK_NEAR_PART"] = PART
r = subprocess.run([sys.executable, "-u", "-m", "experiments.clean_splits", "make", CASE], cwd=work, env=env)
out = pathlib.Path("/kaggle/working")
saved = []
if PART is not None and PART != "final":
    for p in cache.glob("*.npz"):
        shutil.copy2(p, out / p.name)
        saved.append(p.name)
slug = CASE.replace(":", "__")
for suffix in (".json", ".drop.txt.gz"):
    f = work / "splits" / (slug + suffix)
    if f.exists():
        shutil.copy2(f, out / f.name)
ok = r.returncode == 0 and (PART is None or PART == "final" or bool(saved))
(out / "failed.txt").write_text("" if ok else f"{CASE} {PART}: exit {r.returncode}, saved {saved}\n", encoding="utf-8")
shutil.rmtree(work)
print("exit", r.returncode, CASE, flush=True)
'''


def parse(job: str) -> tuple[str, str | None]:
    """'case' -> (case, None); 'case@i/n' -> part i of n; 'case@final/n' -> the final notebook."""
    case, _, part = job.partition("@")
    return case, part or None


def expand(job: str) -> list[str]:
    """'case@n' -> its n parts and the final notebook; anything else as given."""
    case, part = parse(job)
    if part is not None and part.isdigit():
        n = int(part)
        return [f"{case}@{i}/{n}" for i in range(n)] + [f"{case}@final/{n}"]
    return [job]


def parts_of(job: str) -> list[str]:
    """The part jobs a final notebook reads (none for any other job)."""
    case, part = parse(job)
    if part is None or not part.startswith("final/"):
        return []
    n = int(part.split("/")[1])
    return [f"{case}@{i}/{n}" for i in range(n)]


def kernel(job: str) -> str:
    case, part = parse(job)
    tail = ("" if part is None else "-final" if part.startswith("final/")
            else "-p{}of{}".format(*part.split("/")))
    return "leakcheck-clean-" + case.replace(":", "-").replace("_", "-") + tail


def owner(job: str) -> str:
    return OWNERS.get(job, k.USER)


def ref(job: str) -> str:
    return f"{owner(job)}/{kernel(job)}"


def notebook_source(job: str, sha: str) -> str:
    case, part = parse(job)
    part = "final" if part and part.startswith("final/") else part
    return (f"# leakcheck clean split {job}, code {sha}\nCASE = {case!r}\nPART = {part!r}\nSHA = {sha!r}\n"
            f"PINS = {k.PINS!r}\nDATASET = {DATASET_OF[case]!r}\n" + NOTEBOOK)


def push_ember() -> None:
    sys.path.insert(0, str(ROOT))
    from experiments.data import verify_local
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for p in verify_local("ember2018"):
            # Kaggle unpacks an uploaded archive, and the loader checks the archive's SHA-256,
            # so it goes up as <name>.bin (same bytes); the notebook maps the name back
            shutil.copy2(p, d / (p.name + ".bin"))
        k._upload(EMBER, d, "EMBER 2018 archive")
    print(f"uploaded {k.USER}/{EMBER}")


def metadata(job: str) -> dict:
    case = parse(job)[0]
    return {"id": ref(job), "title": kernel(job), "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{k.USER}/{k.CODE}", *SOURCES[case]],
            "competition_sources": [], "kernel_sources": [ref(p) for p in parts_of(job)]}


def start(case: str, sha: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(notebook_source(case, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps(metadata(case)), encoding="utf-8")
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


def _download(case: str, d: Path) -> bool:
    """The notebook's output into d; True only for a whole download (failed.txt there, no partial file)."""
    k.kaggle("kernels", "output", ref(case), "-p", str(d), "--file-pattern",
             r".*(\.json|\.gz|\.txt|\.log|\.npz)$", check=False)
    files = [f for f in d.iterdir() if f.is_file()]
    return (d / "failed.txt").exists() and not any(f.name.endswith(".kaggle-partial") for f in files)


def pull(case: str, tries: int = 3) -> bool | None:
    """Copy a finished notebook's output into results/kaggle/<kernel>/ and report whether it succeeded.

    The download goes to a temporary folder first, and is retried: a dropped connection leaves what was
    pulled before untouched (it once replaced a verified part with a partial file and read as failed)."""
    dest = ROOT / "results" / "kaggle" / kernel(case)
    slug = parse(case)[0].replace(":", "__")
    for attempt in range(tries):
        with tempfile.TemporaryDirectory() as d:
            if not _download(case, Path(d)):
                if attempt + 1 < tries:
                    time.sleep(PULL_RETRY)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            # top level only: a notebook that fails early leaves a copy of the repo in its output, whose
            # splits/<case>.json (the committed one) must not replace the new result
            for f in Path(d).iterdir():
                if f.is_file():
                    shutil.copy2(f, ROOT / "splits" / f.name if f.name in (slug + ".json", slug + ".drop.txt.gz")
                                 else dest / f.name)
        failed = (dest / "failed.txt").read_text(encoding="utf-8").strip()
        print(f"{time.strftime('%H:%M')} {case}: pulled; {'FAILED ' + failed if failed else 'ok'}", flush=True)
        return not failed
    print(f"{time.strftime('%H:%M')} {case}: download incomplete after {tries} tries; earlier files kept, "
          f"tried again next round", flush=True)
    return None


def state(job: str) -> str:
    out = k.kaggle("kernels", "status", ref(job), check=False).stdout
    return next((t for t in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL") if t in out), "")


def handoff(cases: list[str]) -> None:
    """Each friend's notebooks, in HANDOFF/<account>/<kernel>/ (run.py, kernel-metadata.json)."""
    jobs = [j for c in cases for j in expand(c) if owner(j) != k.USER]
    sha = k.push_code()
    for j in jobs:
        d = HANDOFF / owner(j) / kernel(j)
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.py").write_text(notebook_source(j, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps(metadata(j), indent=1), encoding="utf-8")
        print(f"{owner(j)}: {d}")


def watch_only(job: str, st: str) -> bool:
    """Whether a restarted runner watches (and pulls) this job instead of pushing it.

    A friend's part is started by the friend. One of ours that is running, queued or complete is never
    pushed again: a push starts a new version, which replaces a running one and re-runs a finished one
    (a restart once re-ran two finished 2.9 h parts). Only a job that never ran, failed or was cancelled
    is pushed."""
    return owner(job) != k.USER or st in ("RUNNING", "QUEUED", "COMPLETE")


def run(cases: list[str]) -> None:
    cases = [j for c in cases for j in expand(c)]
    sha = k.push_code()
    pending, running, ok, told = list(cases), [], set(), set()
    for c in cases:
        if watch_only(c, state(c)):
            pending.remove(c)
            running.append(c)
    while pending or running:
        for c in list(pending):
            if not set(parts_of(c)) <= ok:      # a final notebook waits for every part it reads
                if any(p not in pending and p not in running and p not in ok for p in parts_of(c)):
                    sys.exit(f"{time.strftime('%H:%M')} {c}: a part failed; not starting it")
                continue
            if start(c, sha):
                pending.remove(c)
                running.append(c)
            else:
                break
        time.sleep(3600 if pending and not running else k.WAIT)
        for c in list(running):
            st = state(c)                       # a friend's notebook not yet pushed or shared: "" (waits)
            if c in told and st in ("RUNNING", "QUEUED"):
                told.discard(c)                 # the friend is running it again
            if st in ("COMPLETE", "ERROR", "CANCEL") and c not in told:
                got = pull(c)
                if got is None:                 # the download broke off: try again next round
                    continue
                running.remove(c)
                if got:
                    ok.add(c)
                elif owner(c) != k.USER:        # the final notebook waits for the friend to run it again
                    print(f"{time.strftime('%H:%M')} {c} on {owner(c)}'s account FAILED: ask them to run it "
                          f"again", flush=True)
                    running.append(c)
                    told.add(c)
    print(f"{time.strftime('%H:%M')} all done: {', '.join(cases)}", flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"push-ember": lambda: push_ember(), "run": lambda: run(args), "handoff": lambda: handoff(args),
     "pull": lambda: [pull(c) for c in args]}.get(cmd, lambda: sys.exit(__doc__))()
