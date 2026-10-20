import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indic_ocr_research.cli import app
from indic_ocr_research.common import normalize
from indic_ocr_research.datasets import read_manifest, split_documents, validate_rows
from indic_ocr_research.engines import Tesseract
from indic_ocr_research.evaluation import benchmark, edit_distance, metrics
from indic_ocr_research.reporting import compare


@pytest.mark.parametrize(
    "reference,prediction,distance",
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "ab", 2),
        ("kitten", "sitting", 3),
        ("हिन्दी", "हिंदी", 2),
        (["one", "two"], ["one", "three"], 1),
    ],
)
def test_edit_distance(reference, prediction, distance):
    assert edit_distance(reference, prediction) == distance


def test_metrics():
    score = metrics([("abc def", "abc xef")])
    assert score["cer"] == pytest.approx(1 / 7)
    assert score["wer"] == 0.5
    assert score["exact_accuracy"] == 0
    assert metrics([])["cer"] is None
    assert metrics([("", "x")])["cer"] is None
    assert metrics([("x", "")])["cer"] == 1
    assert metrics([("", "")])["exact_accuracy"] == 1


@pytest.mark.parametrize("text", ["e\u0301", "क़", "മലയാളം", "हिन्दी"])
def test_unicode(text):
    assert normalize(normalize(text)) == normalize(text)
    assert normalize("e\u0301") == "é"
    assert normalize(text, nfc=False) == text
    assert normalize(" a\n b ", whitespace=True) == "a b"


def test_manifest(manifest):
    rows = read_manifest(manifest)
    report = validate_rows(rows)
    assert report["valid"] and report["samples"] == 3
    assert Path(rows[0]["image_path"]).is_absolute()
    rows[1]["document_id"] = rows[0]["document_id"]
    assert any("Document split leakage" in e for e in validate_rows(rows)["errors"])
    rows[1]["split"] = "train"
    assert validate_rows(rows)["valid"]  # Multiple lines per document are legal.
    rows[1]["sample_id"] = rows[0]["sample_id"]
    assert any("Duplicate sample_id" in e for e in validate_rows(rows)["errors"])


def test_conflicts(manifest):
    rows = read_manifest(manifest)
    rows[1]["image_path"] = rows[0]["image_path"]
    rows[1]["transcription"] = "different"
    report = validate_rows(rows)
    assert not report["valid"]
    assert any("Conflicting" in e for e in report["errors"])
    assert any("content leakage" in e for e in report["errors"])


def test_splits(manifest):
    rows = read_manifest(manifest) * 10
    a = split_documents(rows, 2)
    assert a == split_documents(rows, 2)
    assert all(
        len({r["split"] for r in a if r["document_id"] == doc}) == 1
        for doc in {r["document_id"] for r in a}
    )
    with pytest.raises(ValueError):
        split_documents(rows, ratios=(0.1, 0.1, 0.1))


@pytest.mark.parametrize("content", ["", "{}", "[]", "not-json"])
def test_bad_manifest(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_manifest(path)


def test_missing_image(manifest):
    rows = read_manifest(manifest)
    Path(rows[0]["image_path"]).unlink()
    assert not validate_rows(rows)["valid"]


def test_tesseract_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="executable missing"):
        Tesseract()


def test_tesseract_language_check(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "tesseract")
    monkeypatch.setattr(
        Tesseract,
        "_run",
        lambda self, args: (
            "List of available languages (1):\neng\n"
            if args == ["--list-langs"]
            else "tesseract test"
        ),
    )
    with pytest.raises(RuntimeError, match="language packs"):
        Tesseract("hin")


class MeasuredTestEngine:
    # Unit test protocol fixture; never presented as an OCR accuracy result.
    def recognize(self, image):
        return {
            "text": "ab",
            "elapsed_seconds": 0.1,
            "engine": "unit-fixture",
            "version": "test",
            "configuration": {},
        }


def test_benchmark_and_report(manifest, tmp_path):
    result = benchmark(manifest, MeasuredTestEngine(), tmp_path / "reports", task="line")
    assert result["evaluated_samples"] == 1 and result["failed_samples"] == 0
    assert result["skipped_samples"] == 2 and result["raw"]["cer"] == 0
    assert result["dataset"]["synthetic"]
    assert result["dataset"]["image_sha256"]["2"]
    assert result["dataset"]["manifest_sha256"]
    assert result["predictions"][0]["version"] == "test"
    assert json.loads((tmp_path / "reports/benchmark.json").read_text())["schema_version"] == 1
    assert "| raw |" in (tmp_path / "reports/benchmark.md").read_text()
    assert compare(tmp_path / "reports", tmp_path / "comparison.md") == 1
    assert (tmp_path / "comparison.csv").exists()
    with pytest.raises(ValueError, match="No labeled"):
        benchmark(manifest, MeasuredTestEngine(), tmp_path / "none", task="page")


def test_benchmark_all_fail(manifest, tmp_path):
    class FailedEngine:
        def recognize(self, image):
            raise RuntimeError("Deliberate failure")

    result = benchmark(manifest, FailedEngine(), tmp_path / "failed", task="line")
    assert result["failed_samples"] == 1
    assert result["evaluated_samples"] == 0 and result["raw"]["cer"] is None


@pytest.mark.parametrize(
    "args",
    [
        ["pdf", "inspect", "missing.pdf"],
        ["baseline", "run"],
        ["pdf", "render", "--dpi", "0"],
        ["train", "--config", "missing.yaml"],
    ],
)
def test_cli_invalid(args):
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0


def test_cli_help_and_manifest(manifest):
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["manifest", "validate", "--path", str(manifest)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["valid"]


def test_raw_normalized_distinction(manifest, tmp_path):
    rows = read_manifest(manifest, nfc=False)
    rows[2]["transcription"] = "e\u0301"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    class CanonicalEngine:
        def recognize(self, image):
            return {"text": "é", "elapsed_seconds": 0.01, "engine": "unit-fixture"}

    result = benchmark(manifest, CanonicalEngine(), tmp_path / "unicode", task="line")
    assert result["raw"]["cer"] == 1
    assert result["normalized"]["cer"] == 0
    assert result["predictions"][0]["reference"] == "e\u0301"


def test_cli_bad_yaml(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("bad: [")
    result = CliRunner().invoke(app, ["train", "--config", str(config)])
    assert result.exit_code == 1
    assert "Error:" in result.output


@pytest.mark.parametrize("fail", [False, True])
def test_tesseract_uses_only_validated_frame(tmp_path, monkeypatch, fail):
    from PIL import Image

    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: "tesseract")
    image = tmp_path / "frames.tiff"
    Image.new("RGB", (20, 10), "white").save(
        image, save_all=True, append_images=[Image.new("RGB", (40, 30), "black")]
    )
    prepared_paths = []

    def invoke(self, args):
        if args == ["--version"]:
            return "tesseract test"
        if args == ["--list-langs"]:
            return "List of available languages (1):\neng\n"
        prepared = Path(args[0])
        prepared_paths.append(prepared)
        with Image.open(prepared) as raster:
            assert raster.format == "PNG" and raster.n_frames == 1
            assert raster.size == (20, 10)
            assert raster.getpixel((0, 0)) == (255, 255, 255)
        if fail:
            raise RuntimeError("test subprocess failure")
        return "fixture text"

    monkeypatch.setattr(Tesseract, "_run", invoke)
    engine = Tesseract()
    if fail:
        with pytest.raises(RuntimeError, match="test subprocess failure"):
            engine.recognize(image)
    else:
        result = engine.recognize(image)
        assert result["text"] == "fixture text" and result["elapsed_seconds"] > 0
    assert prepared_paths and not prepared_paths[0].exists()
