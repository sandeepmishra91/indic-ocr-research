import json
from pathlib import Path

import pytest

from indic_ocr_research.datasets import read_manifest
from indic_ocr_research.evaluation import benchmark


@pytest.fixture
def evaluation_manifest(manifest):
    rows = read_manifest(manifest)
    for i, row in enumerate(rows):
        row.update(split="test", language="hin" if i < 2 else "mal")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


class CountingEngine:
    version = "fixture-v1"

    def __init__(self, interrupt_after=None, fail=None):
        self.calls = []
        self.interrupt_after = interrupt_after
        self.fail = fail

    def recognize(self, image: Path):
        if self.interrupt_after is not None and len(self.calls) == self.interrupt_after:
            raise KeyboardInterrupt()
        self.calls.append(image.name)
        if image.name == self.fail:
            raise RuntimeError("fixture failure")
        return {
            "text": "ab" if image.name != "2.png" else "x",
            "elapsed_seconds": 0.01,
            "engine": "unit-fixture",
            "version": self.version,
        }


def test_interrupt_resume_and_language_metrics(evaluation_manifest, tmp_path):
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        benchmark(evaluation_manifest, CountingEngine(interrupt_after=1), out, task="line")
    engine = CountingEngine()
    result = benchmark(evaluation_manifest, engine, out, task="line", resume=True)
    assert engine.calls == ["1.png", "2.png"]
    assert result["resumed_samples"] == 1 and result["session_attempted_samples"] == 2
    assert result["total_attempts"] == 3 and result["evaluated_samples"] == 3
    assert result["by_language"]["hin"]["normalized"]["cer"] == 0
    assert result["by_language"]["mal"]["normalized"]["cer"] == 1
    assert result["latency_percentiles_seconds"]["95"] == 0.01
    assert result["worst_samples"][0]["sample_id"] == "2"
    replay_engine = CountingEngine()
    replay = benchmark(evaluation_manifest, replay_engine, out, task="line", resume=True)
    assert replay_engine.calls == [] and replay["resumed_samples"] == 3
    assert replay["elapsed_seconds"] == result["elapsed_seconds"]
    assert replay["throughput_samples_per_second"] == result["throughput_samples_per_second"]
    assert replay["run_id"] == result["run_id"]
    with pytest.raises(ValueError, match="Output already"):
        benchmark(evaluation_manifest, CountingEngine(), out, task="line")


@pytest.mark.parametrize("change", ["engine", "normalization", "labels", "image"])
def test_resume_rejects_changed_identity(evaluation_manifest, tmp_path, change):
    out = tmp_path / "identity"
    benchmark(evaluation_manifest, CountingEngine(), out, task="line")
    engine, options = CountingEngine(), {}
    if change == "engine":
        engine.version = "fixture-v2"
    elif change == "normalization":
        options["nfc"] = False
    elif change == "labels":
        evaluation_manifest.write_text(evaluation_manifest.read_text().replace('"ab"', '"ba"'))
    else:
        with (tmp_path / "0.png").open("ab") as stream:
            stream.write(b"different bytes")
    with pytest.raises(ValueError, match="identity mismatch"):
        benchmark(evaluation_manifest, engine, out, task="line", resume=True, **options)
    assert engine.calls == []


def test_retry_failed_only(evaluation_manifest, tmp_path):
    out = tmp_path / "failures"
    failed = benchmark(evaluation_manifest, CountingEngine(fail="1.png"), out, task="line")
    assert failed["failed_samples"] == 1
    reuse = CountingEngine()
    still_failed = benchmark(evaluation_manifest, reuse, out, task="line", resume=True)
    assert reuse.calls == [] and still_failed["failed_samples"] == 1
    retry = CountingEngine()
    fixed = benchmark(evaluation_manifest, retry, out, task="line", resume=True, retry_failed=True)
    assert retry.calls == ["1.png"] and fixed["failed_samples"] == 0
    assert fixed["total_attempts"] == 4
    with pytest.raises(ValueError, match="requires"):
        benchmark(evaluation_manifest, retry, tmp_path / "wrong", task="line", retry_failed=True)


def test_torn_final_record_recovery(evaluation_manifest, tmp_path):
    out = tmp_path / "torn"
    benchmark(evaluation_manifest, CountingEngine(), out, task="line")
    journal = out / "predictions.jsonl"
    clean = journal.read_bytes()
    with journal.open("ab") as stream:
        stream.write(b'{"unfinished":')
    result = benchmark(evaluation_manifest, CountingEngine(), out, task="line", resume=True)
    assert result["resumed_samples"] == 3 and journal.read_bytes() == clean
    with journal.open("ab") as stream:
        stream.write(b"not-json\n")
    with pytest.raises(ValueError, match="Corrupt prediction journal"):
        benchmark(evaluation_manifest, CountingEngine(), out, task="line", resume=True)


def test_missing_image_with_leakage_in_id_is_a_sample_failure(evaluation_manifest, tmp_path):
    rows = read_manifest(evaluation_manifest)
    rows[0]["sample_id"] = "contains-leakage-but-is-not-a-structural-error"
    rows[0]["image_path"] = str(tmp_path / "missing.png")
    evaluation_manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    class ReadingEngine(CountingEngine):
        def recognize(self, image):
            image.read_bytes()
            return super().recognize(image)

    result = benchmark(evaluation_manifest, ReadingEngine(), tmp_path / "missing", task="line")
    assert result["failed_samples"] == 1 and result["evaluated_samples"] == 2
