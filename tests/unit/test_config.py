import pytest
import yaml

from indic_ocr_research.config import load_training_config


@pytest.mark.parametrize(
    "key,value",
    [
        ("manifest", 12),
        ("output_dir", None),
        ("epochs", True),
        ("height", 4),
        ("max_width", 2),
        ("threads", 0),
        ("learning_rate", "fast"),
        ("learning_rate", float("nan")),
        ("gradient_clip", float("inf")),
        ("gradient_clip", True),
        ("amp", "false"),
        ("device", []),
        ("optimizer", {}),
        ("seed", -1),
        ("seed", 2**32),
    ],
)
def test_invalid_training_settings(tmp_path, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"manifest": "data.jsonl", "output_dir": "run", key: value}))
    with pytest.raises(ValueError, match=key):
        load_training_config(path)


def test_valid_training_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("manifest: data.jsonl\noutput_dir: run\namp: false\nseed: 0\n")
    result = load_training_config(path)
    assert result["amp"] is False and result["seed"] == 0
    assert result["learning_rate"] == 0.001
