"""The Kaggle runner's job names, accounts and the final notebook's sources (scripts/kaggle_clean.py)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("kaggle_clean", Path(__file__).parent.parent / "scripts" / "kaggle_clean.py")
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)


def test_parts_and_final():
    jobs = kc.expand("ciciot2023@8")
    assert jobs[:2] == ["ciciot2023@0/8", "ciciot2023@1/8"] and jobs[-1] == "ciciot2023@final/8"
    assert kc.parts_of("ciciot2023@final/8") == jobs[:-1] and kc.parts_of("ciciot2023@3/8") == []
    assert kc.kernel("cseids2018:no_identifiers@2/8") == "leakcheck-clean-cseids2018-no-identifiers-p2of8"
    assert kc.kernel("ciciot2023@final/8") == "leakcheck-clean-ciciot2023-final"


def test_final_reads_every_part_from_its_own_account():
    for case in ("ciciot2023", "cseids2018:no_identifiers"):
        final = f"{case}@final/8"
        assert kc.owner(final) == kc.k.USER
        md = kc.metadata(final)
        assert md["kernel_sources"] == [f"{kc.owner(p)}/{kc.kernel(p)}" for p in kc.parts_of(final)]
        for p in kc.parts_of(final):
            assert kc.metadata(p)["id"] == kc.ref(p) and kc.metadata(p)["enable_gpu"]
            # friends' notebooks read the project's code dataset (shared with them)
            assert f"{kc.k.USER}/{kc.k.CODE}" in kc.metadata(p)["dataset_sources"]


def test_notebook_pins_the_part():
    src = kc.notebook_source("ciciot2023@5/8", "abc1234")
    assert "PART = '5/8'" in src and "SHA = 'abc1234'" in src and "CASE = 'ciciot2023'" in src
    assert "PART = 'final'" in kc.notebook_source("ciciot2023@final/8", "abc1234")
