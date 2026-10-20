import json

import pytest

torch = pytest.importorskip("torch")

from indic_ocr_research.common import device_info  # noqa: E402
from indic_ocr_research.model import CRNN, Vocabulary, collate, greedy  # noqa: E402
from indic_ocr_research.training import CRNNEngine, select_device, train  # noqa: E402


def test_shapes_and_lengths():
    torch.set_num_threads(2)
    vocab = Vocabulary(["क", "ा", "മ"])
    encoded = vocab.encode("का")
    assert encoded == [1, 2]
    assert vocab.decode([0, 1, 1, 0, 1, 2]) == "कका"
    with pytest.raises(ValueError, match="absent"):
        vocab.encode("z")
    batch = [(torch.ones(1, 32, 65), [1, 2], "का"), (torch.ones(1, 32, 48), [3], "മ")]
    images, targets, lengths, target_lengths, _ = collate(batch)
    assert lengths.tolist() == [16, 12]
    assert target_lengths.tolist() == [2, 1] and targets.numel() == 3
    model = CRNN(4, 8)
    logits = model(images, lengths)
    assert logits.shape == (16, 2, 4)
    loss = torch.nn.CTCLoss()(logits.log_softmax(2), targets, lengths, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    assert len(greedy(logits, lengths, vocab)) == 2
    with pytest.raises(ValueError, match="alignment"):
        collate([(torch.ones(1, 32, 8), [1, 1], "कक")])


def test_train_resume_infer_benchmark(manifest, tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        "manifest: manifest.jsonl\noutput_dir: run\nepochs: 1\n"
        "batch_size: 1\nhidden: 8\ndevice: cpu\namp: false\n"
    )
    result = train(config)
    checkpoint = tmp_path / "run/model.pt"
    assert checkpoint.exists() and len(result["history"]) == 1
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["epoch"] == 1 and saved["optimizer"]["state"]
    engine = CRNNEngine(checkpoint, "cpu")
    prediction = engine.recognize(tmp_path / "2.png")
    assert isinstance(prediction["text"], str) and prediction["elapsed_seconds"] > 0
    from indic_ocr_research.evaluation import benchmark

    result = benchmark(manifest, engine, tmp_path / "benchmark", task="line")
    assert result["evaluated_samples"] == 1 and result["failed_samples"] == 0
    with pytest.raises(ValueError, match="selected task"):
        benchmark(manifest, engine, tmp_path / "wrong-task", task="page")
    config.write_text(config.read_text().replace("epochs: 1", "epochs: 2"))
    resumed = train(config, checkpoint)
    assert [h["epoch"] for h in resumed["history"]] == [1, 2]
    experiment = json.loads((tmp_path / "run/experiment.json").read_text())
    assert experiment["hardware"]["torch_installed"]
    assert experiment["history"][-1]["validation"]["samples"] == 1
    assert experiment["resumed_from"]
    assert (tmp_path / "run/history.csv").exists()
    resumed_state = torch.load(checkpoint, weights_only=True)["model"]
    separate_config = tmp_path / "continuous.yaml"
    separate_config.write_text(
        config.read_text().replace("output_dir: run", "output_dir: continuous")
    )
    train(separate_config)
    continuous_state = torch.load(tmp_path / "continuous/model.pt", weights_only=True)["model"]
    assert all(torch.equal(resumed_state[k], continuous_state[k]) for k in resumed_state)
    with pytest.raises(ValueError, match="exceed"):
        train(config, checkpoint)
    with (tmp_path / "0.png").open("ab") as stream:
        stream.write(b"changed-image-bytes")
    with pytest.raises(ValueError, match="dataset differs"):
        train(config, checkpoint)
    from indic_ocr_research.reporting import compare

    assert compare(tmp_path, tmp_path / "comparison.md") == 3


def test_device_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    assert str(select_device("cuda")) == "cpu"
    assert device_info()["backend"] == "cpu"
    with pytest.raises(ValueError):
        select_device("bad")


def test_rocm_metadata(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "test-runtime")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert device_info()["backend"] == "rocm"
