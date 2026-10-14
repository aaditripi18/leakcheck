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
