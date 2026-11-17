"""Measure the GPU search on a real Kaggle T4 before the giant parts are handed out.

    python scripts/gpu_bench.py <kaggle account>    writes ../leakcheck_handoff/<account>/leakcheck-gpu-bench/

The notebook carries the committed leakcheck package inside itself (a zip, base64), so it needs no
code dataset and cannot disturb notebooks that read one. It times:
  1. one shortlist step (1,024 test rows x 250,000 training rows) at the giants' widths, 77
     (CSE-CIC-IDS2018) and 46 (CIC-IoT2023): the old float32 pass (torch.cdist without the matrix
     product), the new float32 matrix-product pass, and float64;
  2. the real near_torch.nearest on one synthetic bucket of 3M training rows (float32 embedding, as
     on the giants), float32-certified and float64-only, with the same distances required of both;
  3. from those, the hours one part of each giant needs, at 8 and 16 parts.
Everything is printed and written to bench.json.
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_e14 import PINS  # noqa: E402

BODY = r'''
import base64, io, json, pathlib, subprocess, sys, time, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
src = pathlib.Path("/kaggle/working/src")
zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG))).extractall(src)
sys.path.insert(0, str(src))
import numpy as np, pandas as pd, torch
from leakcheck import Split
from leakcheck.checks import near, near_torch
dev = "cuda"
out = {"code": SHA, "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
print("code", SHA, "|", out["device"], "| torch", torch.__version__, flush=True)
torch.backends.cuda.matmul.allow_tf32 = False


def timeit(f, n=5):
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


STEP = 1024 * 250_000
out["step"] = {}
for d in (77, 46):
    g = torch.Generator(device=dev).manual_seed(0)
    q32 = torch.randn(1024, d, device=dev, generator=g); t32 = torch.randn(250_000, d, device=dev, generator=g)
    q64, t64 = q32.double(), t32.double()
    tb2, qa2, tn64 = (t32 * t32).sum(1), (q32 * q32).sum(1), (t64 * t64).sum(1)

    def old():
        s = torch.cdist(q32, t32, compute_mode="donot_use_mm_for_euclid_dist"); torch.topk(s, 9, dim=1, largest=False)

    def new():
        s = torch.addmm(tb2[None, :], q32, t32.T, alpha=-2.0); s += qa2[:, None]; s.clamp_(min=0).sqrt_(); s -= 1e-7
        torch.topk(s, 9, dim=1, largest=False)

    def f64():
        s = (q64 * q64).sum(1, keepdim=True) + tn64[None, :] - 2.0 * (q64 @ t64.T); torch.topk(s, 8, dim=1, largest=False)

    for name, f in (("old_cdist_f32", old), ("new_matmul_f32", new), ("float64", f64)):
        sec = timeit(f)
        out["step"][f"d{d}_{name}"] = STEP / sec
        print(f"step d={d} {name}: {sec * 1000:.1f} ms, {STEP / sec:.3e} pairs/s", flush=True)
    del q32, t32, q64, t64, tb2, qa2, tn64
    torch.cuda.empty_cache()

# the real search on one bucket: clusters of near copies (as the giants: most rows have a copy within
# 0.01) and repeated rows, the embedding in float32 (near.LARGE as on the giants)
near.LARGE = 0
out["search"] = {}
rng = np.random.default_rng(0)
for d in (77, 46):
    ntr, nte = 3_000_000, 50_000
    centres = rng.normal(size=(200_000, d)) * rng.lognormal(0, 1, size=d)
    x = centres[rng.integers(0, len(centres), ntr)] + rng.normal(0, 1e-3, size=(ntr, d)) * (rng.random((ntr, 1)) < 0.7)
    te = centres[rng.integers(0, len(centres), nte)] + rng.normal(0, 1e-3, size=(nte, d))
    cols = [f"f{i}" for i in range(d)]
    s = Split(pd.DataFrame(x, columns=cols).assign(y=0), pd.DataFrame(te, columns=cols).assign(y=0), "y")
    del x
    s.train_hash  # hashed once, outside the timings
    res = {}
    for fast in (True, False):
        t0 = time.time()
        dd, nn = near_torch.nearest(s, fast=fast, progress=0)
        sec = time.time() - t0
        res[fast] = (dd, nn)
        pairs = near_torch.nearest.searched * nte
        key = f"d{d}_{'certified_f32' if fast else 'float64_only'}"
        out["search"][key] = {"seconds": sec, "pairs": pairs, "pairs_per_s": pairs / sec,
                              "redone_share": near_torch.nearest.redone / nte if fast else 1.0}
        print(f"search d={d} {key}: {sec:.0f} s, {pairs / sec:.3e} pairs/s, redone "
              f"{out['search'][key]['redone_share']:.1%}", flush=True)
    same = bool(np.array_equal(res[True][0], res[False][0]))
    out["search"][f"d{d}_equal"] = same
    print(f"search d={d}: certified float32 and float64 distances equal: {same}", flush=True)
    assert same
    del s, res

# hours per part: float32 pass on every pair, float64 on the redone share (0.57 measured on
# CSE-CIC-IDS2018 rows; 1.0 the worst case), at the measured whole-search rates
cases = {"cseids2018:no_identifiers": (77, 8_325_605 * 1_924_281),
         "ciciot2023 (upper bound)": (46, int(37_349_263 * (1 - 0.386)) * int(9_337_316 * (1 - 0.443)))}
out["hours_per_part"] = {}
for case, (d, pairs) in cases.items():
    s32, s64 = out["search"][f"d{d}_certified_f32"], out["search"][f"d{d}_float64_only"]
    f32_only = s32["seconds"] - s32["redone_share"] * s64["seconds"]  # the float32 pass alone
    per_pair32, per_pair64 = max(f32_only, 0) / s32["pairs"], s64["seconds"] / s64["pairs"]
    for parts in (8, 16):
        for share in (0.57, 1.0):
            h = pairs / parts * (per_pair32 + share * per_pair64) / 3600
            out["hours_per_part"][f"{case} | {parts} parts | redone {share}"] = h
            print(f"{case}: {parts} parts, redone share {share}: {h:.2f} h per part", flush=True)
pathlib.Path("/kaggle/working/bench.json").write_text(json.dumps(out, indent=1))
print("bench done", flush=True)
'''


def package_zip() -> str:
    """The committed leakcheck package (HEAD, not the working tree), as base64."""
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "leakcheck"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if n.endswith(".py"):
                blob = subprocess.run(["git", "show", f"HEAD:{n}"], cwd=ROOT, capture_output=True, check=True).stdout
                z.writestr(n, blob)
    return base64.b64encode(buf.getvalue()).decode()


def main(account: str) -> None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    d = ROOT.parent / "leakcheck_handoff" / account / "leakcheck-gpu-bench"
    d.mkdir(parents=True, exist_ok=True)
    src = f"# leakcheck GPU benchmark, code {sha}\nSHA = {sha!r}\nPINS = {PINS!r}\nPKG = {package_zip()!r}\n" + BODY
    (d / "run.py").write_text(src, encoding="utf-8")
    meta = {"id": f"{account}/leakcheck-gpu-bench", "title": "leakcheck-gpu-bench", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(d)


if __name__ == "__main__":
    main(sys.argv[1])
