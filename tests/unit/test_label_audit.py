import json

import pytest
from typer.testing import CliRunner

from indic_ocr_research.cli import app
from indic_ocr_research.datasets import read_manifest
from indic_ocr_research.label_audit import audit_manifest
from indic_ocr_research.text_audit import audit_pair


@pytest.mark.parametrize(
    "text",
    [
        "thought**Image Analysis:** example",
        "<think>analysis</think>caption",
        "Image Analysis: the user has provided an image",
    ],
)
def test_possible_model_analysis_is_flagged(text):
    assert "possible-annotation-meta-text" in audit_pair(text, "", "mal")["reference_review_flags"]


def test_review_queue_preserves_original_labels(manifest, tmp_path):
    rows = read_manifest(manifest, nfc=False)
    rows[2]["transcription"] = "thought**Image Analysis:** not a literal transcription"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    original = manifest.read_bytes()
    out, review = tmp_path / "audit.json", tmp_path / "review.jsonl"
    result = audit_manifest(manifest, out, review)
    assert result["flagged_samples"] == 1 and manifest.read_bytes() == original
    queued = read_manifest(review, nfc=False)
    assert len(queued) == 1 and queued[0]["transcription"] == rows[2]["transcription"]
    assert queued[0]["review_status"] == "needs-human-review"
    cli = CliRunner().invoke(
        app, ["manifest", "audit", "--path", str(manifest), "--out", str(out), "--fail-on-flags"]
    )
    assert cli.exit_code == 1 and "flagged_samples" in cli.output
    with pytest.raises(ValueError, match="overwrite"):
        audit_manifest(manifest, manifest)


def test_legitimate_mixed_language_is_not_auto_rejected():
    audit = audit_pair("This image shows Malayalam", "", "mal")
    assert "possible-annotation-meta-text" not in audit["reference_review_flags"]
    assert "low-expected-script-fraction" in audit["reference_review_flags"]


def test_benchmark_flags_bad_labels_even_when_inference_fails(manifest, tmp_path):
    from indic_ocr_research.evaluation import benchmark

    rows = read_manifest(manifest, nfc=False)
    rows[2]["transcription"] = "thought**Image Analysis:** not the image transcription"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    class FailedEngine:
        def recognize(self, image):
            raise RuntimeError("fixture engine failure")

    result = benchmark(manifest, FailedEngine(), tmp_path / "failed", task="line")
    assert result["quality_status"] == "label-review-required"
    assert result["diagnostics"]["annotation_meta_text_references"] == 1
    assert result["evaluated_samples"] == 0 and result["raw"]["cer"] is None
