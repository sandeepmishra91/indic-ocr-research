import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    rows = []
    for i, split in enumerate(["train", "validation", "test"]):
        image = Image.new("RGB", (80 + i * 4, 32), "white")
        ImageDraw.Draw(image).text((4 + i, 8), "ab", fill="black")
        path = tmp_path / f"{i}.png"
        image.save(path)
        rows.append(
            {
                "sample_id": str(i),
                "image_path": path.name,
                "transcription": "ab",
                "language": "eng",
                "script": "Latin",
                "document_id": f"doc{i}",
                "split": split,
                "source_url": "",
                "license": "Apache-2.0",
                "annotation_method": "synthetic",
                "task": "line",
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path
