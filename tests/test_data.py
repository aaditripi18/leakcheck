import gzip
import hashlib
import json
import zipfile

import pytest

from experiments import data
from experiments.ledger import record


def _registry(src, sha):
    return {"toy": {"files": [{"path": "toy/f.bin", "url": src.as_uri(), "sha256": sha}]}}


def test_fetch_verifies_and_writes_manifest(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    sha = hashlib.sha256(b"hello").hexdigest()
    man = tmp_path / "manifest.json"
    (p,) = data.fetch("toy", registry=_registry(src, sha), raw=tmp_path / "raw", manifest=man)
    assert p.read_bytes() == b"hello"
    entry = json.loads(man.read_text())["toy/f.bin"]
    assert entry["sha256"] == sha and entry["size"] == 5


def test_fetch_rejects_wrong_hash_and_leaves_nothing(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"tampered")
    raw = tmp_path / "raw"
    with pytest.raises(data.HashMismatch):
        data.fetch("toy", registry=_registry(src, "0" * 64), raw=raw, manifest=tmp_path / "m.json")
    assert not list(raw.rglob("*.bin")) and not list(raw.rglob("*.part"))


def test_fetch_refuses_unpinned_without_pin(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    reg = _registry(src, None)
    with pytest.raises(data.HashMismatch, match="no pinned"):
        data.fetch("toy", registry=reg, raw=tmp_path / "raw", manifest=tmp_path / "m.json")
    data.fetch("toy", registry=reg, raw=tmp_path / "raw", manifest=tmp_path / "m.json", pin=True)
    assert reg["toy"]["files"][0]["sha256"] == hashlib.sha256(b"x").hexdigest()


def test_verify_local_detects_modified_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    reg = _registry(src, hashlib.sha256(b"hello").hexdigest())
    (p,) = data.fetch("toy", registry=reg, raw=tmp_path / "raw", manifest=tmp_path / "m.json")
    p.write_bytes(b"hellO")
    with pytest.raises(data.HashMismatch):
        data.verify_local("toy", registry=reg, raw=tmp_path / "raw")


def test_zip_member_exact_name_only(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("KDDTrain+.txt", "x")
        zf.writestr("__MACOSX/._KDDTrain+.txt", "junk")
        zf.writestr("../evil/KDDTest+.txt", "y")
    with zipfile.ZipFile(z) as zf:
        assert data._zip_member(zf, "KDDTrain+.txt") == "KDDTrain+.txt"
        with pytest.raises(KeyError):
            data._zip_member(zf, "KDDTest+.txt")


def test_kdd_reader_strips_label_dot(tmp_path):
    row = ",".join(["0", "tcp", "http", "SF"] + ["0"] * 37 + ["normal."])
    buf = gzip.compress((row + "\n" + row.replace("normal.", "smurf.") + "\n").encode())
    df = data._read_kdd(gzip.open(__import__("io").BytesIO(buf)), [])
    assert df["label"].tolist() == ["normal", "smurf"] and df["is_attack"].tolist() == [0, 1]


def test_ledger_line(tmp_path):
    led = tmp_path / "l.jsonl"
    record("x", {"a": 1}, {"b": 2}, ledger=led)
    line = json.loads(led.read_text())
    assert line["experiment"] == "x" and len(line["git"]["commit"]) >= 7 and "pandas" in line["env"]


# --- resumable download -------------------------------------------------------------

import http.server  # noqa: E402
import threading  # noqa: E402


class _Flaky(http.server.BaseHTTPRequestHandler):
    payload = bytes(range(256)) * 400  # 102,400 bytes
    cut = 30_000  # bytes sent per response before the connection is dropped

    def do_GET(self):
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = self.payload[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        self.end_headers()
        self.wfile.write(body[: self.cut])  # then hang up early
        self.close_connection = True

    def log_message(self, *a):
        pass


def test_download_resumes_after_dropped_connections(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "f.part"
        data.download(f"http://127.0.0.1:{srv.server_port}/f", out, wait=0.01)
        assert out.read_bytes() == _Flaky.payload
    finally:
        srv.shutdown()


# --- EMBER reader -------------------------------------------------------------------

import io  # noqa: E402
import tarfile  # noqa: E402

from experiments import ember  # noqa: E402


def _ember_record(i, label, appeared="2018-01", tweak=0):
    return {
        "sha256": f"{i:064x}", "md5": "x", "appeared": appeared, "label": label, "avclass": "fam" if label == 1 else "",
        "histogram": [i + 1] * 256, "byteentropy": [1] * 256,
        "strings": {"numstrings": 3, "avlength": 5.0, "printabledist": [1] * 96, "printables": 96, "entropy": 4.0,
                    "paths": 0, "urls": tweak, "registry": 0, "MZ": 1},
        "general": {k: 1 for k in ember.GENERAL},
        "header": {"coff": {"timestamp": 100, "machine": "I386", "characteristics": []},
                   "optional": {"subsystem": "WINDOWS_GUI", "magic": "PE32", **{k: 1 for k in ember.OPTIONAL}}},
        "section": {"entry": ".text", "sections": [{"name": ".text", "size": 10, "entropy": 6.0, "vsize": 12, "props": []}]},
        "imports": {"kernel32.dll": ["a", "b"]}, "exports": [],
        "datadirectories": [{"name": "X", "size": 1, "virtual_address": 2}] * 15,
    }


def _tar(path, members):
    with tarfile.open(path, "w:bz2") as tf:
        for name, recs in members.items():
            raw = "\n".join(json.dumps(r) for r in recs).encode() + b"\n"
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def test_ember_parse_views(tmp_path):
    p = tmp_path / "e.tar.bz2"
    same_a = _ember_record(1, 1)
    same_b = dict(_ember_record(1, 0), sha256="f" * 64, appeared="2018-11")  # same features, other file/label
    _tar(p, {
        "ember2018/train_features_0.jsonl": [same_a, _ember_record(2, 0)],
        "ember2018/test_features.jsonl": [same_b, _ember_record(3, 1, tweak=1)],
        "ember2018/../evil/train_features_1.jsonl": [_ember_record(9, 1)],
        "ember2018/ember_model_2018.txt": [],
    })
    df = ember.parse(p)
    assert df["split"].tolist() == ["train", "train", "test", "test"]
    assert len(df.columns) == 6 + len(ember.columns()) + len(ember.CATEGORICAL)
    d = df["raw_digest"].tolist()
    assert d[0] == d[2] and len(set(d)) == 3  # meta fields excluded from the digest
    assert df.loc[0, "hist_0"] == pytest.approx(1 / 256)
