import io
import json
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from indic_ocr_research.datasets import read_manifest
from indic_ocr_research.hub import import_dataset, plan_dataset, recipe


@pytest.fixture
def config(tmp_path):
    value = {
        "repo_id": "fixture/dataset",
        "revision": "a" * 40,
        "license": "cc-by-4.0",
        "license_url": "https://huggingface.co/datasets/fixture/dataset",
        "attribution": "fixture",
        "task": "block",
        "annotation_method": "synthetic-unit-fixture",
        "image_column": "image",
        "text_column": "text",
        "id_column": "id",
        "max_download_bytes": 100000,
        "max_samples_per_file": 2,
        "files": [{"path": "data/test.parquet", "language": "hin", "script": "Devanagari"}],
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.mark.parametrize(
    "key,value",
    [
        ("revision", "main"),
        ("license", "unknown"),
        ("max_download_bytes", 0),
        ("task", "word"),
        ("files", [{"path": "../secret.parquet", "language": "hin", "script": "Devanagari"}]),
    ],
)
def test_reject_unsafe_recipe(config, key, value):
    spec = yaml.safe_load(config.read_text())
    spec[key] = value
    config.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        recipe(config)


@pytest.fixture
def fake_api(monkeypatch):
    info = SimpleNamespace(
        sha="a" * 40,
        card_data={"license": "cc-by-4.0"},
        siblings=[SimpleNamespace(rfilename="data/test.parquet", size=50, lfs=None)],
    )
    calls = []

    class Api:
        def __init__(self, **kwargs):
            assert kwargs == {"endpoint": "https://huggingface.co"}

        def dataset_info(self, *args, **kwargs):
            calls.append(kwargs)
            return info

    def no_download(**kwargs):
        raise AssertionError("metadata inspection must not download data")

    monkeypatch.setattr("indic_ocr_research.hub._api", lambda: (Api, no_download))
    return info, calls, Api


def test_plan_offline_and_budget(config, tmp_path, fake_api, monkeypatch):
    info, calls, _ = fake_api
    plan = plan_dataset(config, tmp_path / "cache")
    assert plan["total_bytes"] == 50 and len(calls) == 1
    monkeypatch.setattr("indic_ocr_research.hub._api", lambda: pytest.fail("offline network call"))
    assert plan_dataset(config, tmp_path / "cache", offline=True) == plan
    with pytest.raises(ValueError, match="No cached plan"):
        plan_dataset(config, tmp_path / "empty", offline=True)


@pytest.mark.parametrize("change", ["license", "revision", "size", "missing", "budget"])
def test_metadata_guardrails(config, tmp_path, fake_api, change):
    info, _, _ = fake_api
    if change == "license":
        info.card_data["license"] = "unknown"
    elif change == "revision":
        info.sha = "b" * 40
    elif change == "size":
        info.siblings[0].size = None
    elif change == "missing":
        info.siblings = []
    else:
        info.siblings[0].size = 100001
    with pytest.raises(ValueError):
        plan_dataset(config, tmp_path / "cache")


def test_network_errors_do_not_leak_token(config, tmp_path, monkeypatch):
    secret = "unit-test-sensitive-value"

    class Api:
        def __init__(self, **kwargs):
            assert kwargs == {"endpoint": "https://huggingface.co"}

        def dataset_info(self, *args, **kwargs):
            raise RuntimeError(secret)

    monkeypatch.setattr("indic_ocr_research.hub._api", lambda: (Api, None))
    with pytest.raises(RuntimeError) as caught:
        plan_dataset(config, tmp_path / "cache")
    assert secret not in str(caught.value) and "metadata request failed" in str(caught.value)


def test_actual_parquet_import_and_offline_reuse(config, tmp_path, fake_api, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    info, _, Api = fake_api
    image = Image.new("RGB", (40, 12), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    cached = tmp_path / "test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"image": {"bytes": buffer.getvalue(), "path": None}, "text": "क", "id": "source1"},
            ]
        ),
        cached,
    )
    info.siblings[0].size = cached.stat().st_size
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr("indic_ocr_research.hub._api", lambda: (Api, download))
    out, cache = tmp_path / "dataset", tmp_path / "cache"
    imported = import_dataset(config, out, cache)
    assert imported["samples"] == 1 and not imported["ground_truth_verified"]
    rows = read_manifest(out / "manifest.jsonl")
    assert rows[0]["task"] == "block" and rows[0]["transcription"] == "क"
    assert rows[0]["split"] == "test" and rows[0]["hf_revision"] == "a" * 40
    assert json.loads((out / "validation.json").read_text())["valid"]
    old = (out / "manifest.jsonl").read_bytes()
    imported = import_dataset(config, out, cache, offline=True)
    assert calls[-1]["local_files_only"] is True and calls[-1]["token"] is False
    assert (out / "manifest.jsonl").read_bytes() == old
    cached.write_bytes(cached.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="size/hash"):
        import_dataset(config, out, cache, offline=True)
