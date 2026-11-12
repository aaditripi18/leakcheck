"""The Kaggle notebook for E14 is valid Python and names the case, the commit and the pins."""
import ast
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("kaggle_e14", Path(__file__).parents[1] / "scripts" / "kaggle_e14.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


def test_notebook_parses_and_names_its_inputs():
    src = k.notebook_source("ton_network:no_identifiers", "abc1234")
    ast.parse(src)
    assert "CASE = 'ton_network:no_identifiers'" in src and "SHA = 'abc1234'" in src
    assert "scikit-learn==1.9.1" in src and '"full", CASE' in src


def test_kernel_slugs_are_kaggle_safe():
    s = k.kernel("ton_garage_door:no_identifiers")
    assert s == "leakcheck-e14-ton-garage-door-no-identifiers" and len(s) <= 50


def test_ledger_takes_the_commit_from_the_environment_without_git(monkeypatch, tmp_path):
    from experiments import ledger
    monkeypatch.setattr(ledger, "ROOT", tmp_path)          # no repository here
    monkeypatch.setenv("LEAKCHECK_COMMIT", "abc1234")
    assert ledger.git_state() == {"commit": "abc1234", "dirty": False, "where": "kaggle"}
