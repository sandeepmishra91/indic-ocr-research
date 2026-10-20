import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import device_info, provenance, write_json
from .datasets import read_manifest, validate_rows
from .documents import source_id
from .evaluation import metrics
from .model import CRNN, Vocabulary, collate, greedy, line_tensor
from .reporting import training_history


def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda (also used by ROCm)")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


class Lines(Dataset):
    def __init__(self, rows: list[dict], vocab: Vocabulary, height: int, max_width: int):
        self.rows, self.vocab, self.height, self.max_width = rows, vocab, height, max_width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple:
        row = self.rows[index]
        return (
            line_tensor(Path(row["image_path"]), self.height, self.max_width),
            self.vocab.encode(row["transcription"]),
            row["transcription"],
        )


def train(config_path: Path, resume: Path | None = None) -> dict:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid training YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    required = {"manifest", "output_dir"}
    if required - config.keys():
        raise ValueError(f"Missing config values: {sorted(required - config.keys())}")
    defaults = {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "hidden": 64,
        "height": 32,
        "max_width": 2048,
        "seed": 42,
        "device": "auto",
        "optimizer": "adam",
        "gradient_clip": 5.0,
        "amp": True,
        "threads": 2,
    }
    unknown = config.keys() - defaults.keys() - required
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = defaults | config
    for key in ["epochs", "batch_size", "hidden", "height", "max_width", "threads"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["learning_rate"] <= 0 or config["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    torch.set_num_threads(config["threads"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = (config_path.parent / config["manifest"]).resolve()
    out = (config_path.parent / config["output_dir"]).resolve()
    rows = read_manifest(manifest)
    report = validate_rows(rows)
    if not report["valid"]:
        raise ValueError("Invalid dataset: " + "; ".join(report["errors"]))
    selected = [r for r in rows if r["split"] in {"train", "validation"}]
    if any(r["task"] != "line" for r in selected):
        raise ValueError("CRNN training accepts cropped line samples only")
    training = [r for r in selected if r["split"] == "train"]
    validation = [r for r in selected if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Nonempty train and validation splits are required")
    vocab = Vocabulary(sorted({c for r in training for c in r["transcription"]}))
    if not vocab.characters:
        raise ValueError("Training vocabulary is empty")
    dataset_id = source_id(manifest)
    # Fingerprint bytes as well as the manifest to catch in-place image changes on resume.
    data_hash = hashlib.sha256()
    for row in selected:
        data_hash.update(source_id(Path(row["image_path"])).encode())
    fingerprint = data_hash.hexdigest()
    device = select_device(config["device"])
    model = CRNN(len(vocab.characters) + 1, config["hidden"]).to(device)
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    else:
        raise ValueError("optimizer must be adam or sgd")
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    history, start_epoch = [], 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["dataset_id"] != dataset_id or saved["image_fingerprint"] != fingerprint:
            raise ValueError("Resume dataset differs from checkpoint")
        for key in [
            "hidden",
            "height",
            "max_width",
            "optimizer",
            "batch_size",
            "seed",
            "learning_rate",
            "gradient_clip",
        ]:
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        if saved["characters"] != vocab.characters:
            raise ValueError("Resume vocabulary differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved.get("gpu_rng"):
            torch.cuda.set_rng_state_all(saved["gpu_rng"])
        history, start_epoch = saved["history"], saved["epoch"]
        if config["epochs"] <= start_epoch:
            raise ValueError("epochs must exceed completed checkpoint epochs")
    loaders = [
        DataLoader(
            Lines(part, vocab, config["height"], config["max_width"]),
            batch_size=config["batch_size"],
            shuffle=index == 0,
            collate_fn=collate,
            generator=generator,
            num_workers=0,
        )
        for index, part in enumerate([training, validation])
    ]
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=False)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    for epoch in range(start_epoch, config["epochs"]):
        losses, val_losses, pairs = [], [], []
        for is_training, loader in [(True, loaders[0]), (False, loaders[1])]:
            model.train(is_training)
            for images, targets, lengths, target_lengths, texts in loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(is_training):
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits = model(images, lengths)
                    # CTC runs in float32 on both CUDA and ROCm.
                    loss = loss_fn(logits.float().log_softmax(2), targets, lengths, target_lengths)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite CTC loss; inspect alignments and inputs")
                    if is_training:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                        scaler.step(optimizer)
                        scaler.update()
                (losses if is_training else val_losses).append((loss.item(), len(texts)))
                if not is_training:
                    pairs.extend(zip(texts, greedy(logits, lengths, vocab), strict=True))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(v * n for v, n in losses) / sum(n for _, n in losses),
                "validation_loss": sum(v * n for v, n in val_losses)
                / sum(n for _, n in val_losses),
                "validation": metrics(pairs),
            }
        )
        state = {
            "schema_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "characters": vocab.characters,
            "config": config,
            "epoch": epoch + 1,
            "history": history,
            "dataset_id": dataset_id,
            "image_fingerprint": fingerprint,
            "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "gpu_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        training_history(history, out / "history.md")
        write_json(
            out / "experiment.json",
            {
                "kind": "training",
                "model_version": source_id(checkpoint),
                "image_fingerprint": fingerprint,
                "synthetic": any("synthetic" in r["annotation_method"].lower() for r in selected),
                "config": config,
                "environment": provenance(),
                "hardware": device_info(),
                "device": str(device),
                "dataset_id": dataset_id,
                "split": ["train", "validation"],
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
                "resumed_from": str(resume) if resume else None,
                "artifacts": {"checkpoint": str(checkpoint)},
            },
        )
    return {"checkpoint": str(checkpoint), "history": history, "device": str(device)}


class CRNNEngine:
    supported_tasks = {"line"}

    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.device = select_device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = saved["config"]
        self.vocab = Vocabulary(saved["characters"])
        self.model = CRNN(len(self.vocab.characters) + 1, self.config["hidden"]).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.version = source_id(checkpoint)

    def recognize(self, image: Path) -> dict:
        started = time.perf_counter()
        tensor = line_tensor(image, self.config["height"], self.config["max_width"])
        lengths = torch.tensor([tensor.shape[-1] // 4])
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device), lengths)
            text = greedy(logits, lengths, self.vocab)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return {
            "text": text,
            "engine": "crnn",
            "version": self.version,
            "elapsed_seconds": time.perf_counter() - started,
            "configuration": {
                "device": str(self.device),
                "height": self.config["height"],
                "max_width": self.config["max_width"],
                "task": "line",
            },
        }
