"""Generate OWN synthetic English lines. Mechanics only; no quality benchmark."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def generate(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Every split has distinct images and documents; all share the train vocabulary.
    for index, (text, split) in enumerate(
        [("ab", "train"), ("ba", "train"), ("aa", "train"), ("ab", "validation"), ("ba", "test")]
    ):
        image = Image.new("RGB", (80 + index * 4, 32), "white")
        ImageDraw.Draw(image).text((5 + index, 8), text, fill="black")
        name = f"line-{index}.png"
        image.save(out / name)
        rows.append(
            {
                "sample_id": f"synthetic-{index}",
                "image_path": name,
                "transcription": text,
                "language": "eng",
                "script": "Latin",
                "document_id": f"synthetic-doc-{index}",
                "split": split,
                "source_url": "",
                "license": "Apache-2.0",
                "task": "line",
                "annotation_method": "synthetic-rendered-not-human-ground-truth",
            }
        )
    manifest = out / "lines.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/smoke-data"))
    print(generate(parser.parse_args().out))
