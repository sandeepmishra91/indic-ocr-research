import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import normalize
from .documents import load_image, source_id

REQUIRED = {
    "sample_id",
    "image_path",
    "transcription",
    "language",
    "script",
    "document_id",
    "split",
    "source_url",
    "license",
    "annotation_method",
    "task",
}


def read_manifest(path: Path, nfc: bool = True) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected object")
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"Line {number}: missing {sorted(missing)}")
        if any(not isinstance(row[key], str) for key in REQUIRED):
            raise ValueError(f"Line {number}: metadata must be strings")
        if any(not row[key].strip() for key in REQUIRED - {"transcription", "source_url"}):
            raise ValueError(f"Line {number}: empty required metadata")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Line {number}: invalid split")
        if row["task"] not in {"line", "block", "page"}:
            raise ValueError(f"Line {number}: task must be line, block, or page")
        row["transcription"] = normalize(row["transcription"], nfc=nfc)
        row["image_path"] = str((path.parent / row["image_path"]).resolve())
        rows.append(row)
    if not rows:
        raise ValueError("Manifest contains no samples")
    return rows


def validate_rows(rows: list[dict]) -> dict:
    errors, warnings, issues = [], [], []

    def issue(code: str, message: str) -> None:
        errors.append(message)
        issues.append({"code": code, "message": message})

    ids, documents, hashes = set(), defaultdict(set), defaultdict(list)
    for row in rows:
        sid = row["sample_id"]
        if sid in ids:
            issue("sample_id_duplicate", f"Duplicate sample_id: {sid}")
        ids.add(sid)
        documents[row["document_id"]].add(row["split"])
        try:
            image = Path(row["image_path"])
            load_image(image).close()
            hashes[source_id(image)].append(row)
        except (OSError, ValueError) as exc:
            issue("image_unreadable", f"{sid}: {exc}")
        if not row["transcription"].strip():
            warnings.append(f"{sid}: empty transcription (valid blank target)")
    for doc, splits in documents.items():
        if len(splits) > 1:
            issue("document_leakage", f"Document split leakage: {doc}: {sorted(splits)}")
    for samples in hashes.values():
        if len(samples) > 1:
            warnings.append(f"Duplicate image content: {[r['sample_id'] for r in samples]}")
            if len({r["transcription"] for r in samples}) > 1:
                issue("image_conflict", "Conflicting transcriptions for identical image content")
            if len({r["split"] for r in samples}) > 1:
                issue("image_leakage", "Image content leakage across splits")
    return {
        "valid": not errors,
        "samples": len(rows),
        "documents": len(documents),
        "splits": dict(Counter(r["split"] for r in rows)),
        "languages": dict(Counter(r["language"] for r in rows)),
        "tasks": dict(Counter(r["task"] for r in rows)),
        "errors": errors,
        "issues": issues,
        "warnings": warnings,
    }


def split_documents(
    rows: list[dict], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> list[dict]:
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three nonnegative split ratios must sum to one")
    docs = sorted({row["document_id"] for row in rows})
    random.Random(seed).shuffle(docs)
    first, second = int(len(docs) * ratios[0]), int(len(docs) * sum(ratios[:2]))
    mapping = {
        doc: "train" if i < first else "validation" if i < second else "test"
        for i, doc in enumerate(docs)
    }
    return [row | {"split": mapping[row["document_id"]]} for row in rows]
