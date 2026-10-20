"""Exercise real benchmark journals across a partially completed synthetic pilot grid."""

import importlib.util
from pathlib import Path

import pytest

from indic_ocr_research.datasets import read_manifest


def test_resume_continues_new_grid_cases(manifest, tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/run_hf_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        row | {"language": "hin", "task": "block", "split": "test"}
        for row in read_manifest(manifest)
    ]
    receipt = {
        "spec": {
            "task": "block",
            "repo_id": "unit/fixture",
            "revision": "a" * 40,
            "license": "cc-by-4.0",
            "license_url": "https://example.test/fixture",
            "attribution": "synthetic test fixture",
        },
        "artifacts": [],
    }
    monkeypatch.setattr(module, "import_dataset", lambda *a, **k: receipt)
    monkeypatch.setattr(module, "read_manifest", lambda *a, **k: rows)
    calls = []
    interrupt = [True]

    class Engine:
        version = "fixture-v1"

        def __init__(self, language, psm):
            self.language, self.psm = language, psm

        def recognize(self, image):
            if len(calls) == 4 and interrupt[0]:
                interrupt[0] = False
                raise KeyboardInterrupt()
            calls.append((self.language, self.psm, image.name))
            return {"text": "ab", "elapsed_seconds": 0.01, "engine": "unit-fixture"}

    monkeypatch.setattr(module, "Tesseract", Engine)
    out = tmp_path / "pilot"
    with pytest.raises(KeyboardInterrupt):
        module.run(tmp_path / "config.yaml", tmp_path / "data", tmp_path / "cache", out)
    assert len(calls) == 4
    assert not (out / "runs/hin-indic-plus-english-psm3/predictions.jsonl").exists()
    summary = module.run(
        tmp_path / "config.yaml", tmp_path / "data", tmp_path / "cache", out, resume=True
    )
    assert len(summary["runs"]) == 4 and all(r["samples"] == 3 for r in summary["runs"])
    assert len(calls) == 12  # Every successful sample/setting exactly once across both sessions.
    assert (out / "summary.csv").is_file()
