"""Validate training settings before importing datasets or allocating model tensors."""

import math
from pathlib import Path

import yaml

DEFAULTS = {
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


def load_training_config(path: Path) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid training YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    required = {"manifest", "output_dir"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config values: {sorted(missing)}")
    unknown = config.keys() - DEFAULTS.keys() - required
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(map(str, unknown))}")
    config = DEFAULTS | config
    for key in required:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
    for key, minimum in {
        "epochs": 1,
        "batch_size": 1,
        "hidden": 1,
        "height": 8,
        "max_width": 4,
        "threads": 1,
        "seed": 0,
    }.items():
        value = config[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    for key in ("learning_rate", "gradient_clip"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if type(config["amp"]) is not bool:
        raise ValueError("amp must be a YAML boolean (true or false)")
    for key, choices in {"device": {"auto", "cpu", "cuda"}, "optimizer": {"adam", "sgd"}}.items():
        if not isinstance(config[key], str) or config[key] not in choices:
            raise ValueError(f"{key} must be one of {sorted(choices)}")
    return config
