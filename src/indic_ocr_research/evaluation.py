import time
from pathlib import Path
from typing import Protocol, Sequence

from .common import normalize, provenance, write_json
from .datasets import read_manifest, validate_rows
from .documents import source_id


class Engine(Protocol):
    def recognize(self, image: Path) -> dict: ...


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    previous = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(prediction, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual))
            )
        previous = current
    return previous[-1]


def metrics(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"samples": 0, "cer": None, "wer": None, "exact_accuracy": None}
    char_count = sum(len(ref) for ref, _ in pairs)
    word_count = sum(len(ref.split()) for ref, _ in pairs)
    chars = sum(edit_distance(ref, pred) for ref, pred in pairs)
    words = sum(edit_distance(ref.split(), pred.split()) for ref, pred in pairs)
    return {
        "samples": len(pairs),
        "character_edits": chars,
        "reference_characters": char_count,
        "word_edits": words,
        "reference_words": word_count,
        "cer": chars / char_count if char_count else None,
        "wer": words / word_count if word_count else None,
        "exact_accuracy": sum(ref == pred for ref, pred in pairs) / len(pairs),
    }


def benchmark(
    manifest: Path,
    engine: Engine,
    out: Path,
    split: str = "test",
    task: str = "page",
    nfc: bool = True,
    whitespace: bool = True,
) -> dict:
    if task not in getattr(engine, "supported_tasks", {"line", "page"}):
        raise ValueError("Engine does not support the selected task")
    rows = read_manifest(manifest, nfc=False)
    validation = validate_rows(rows)
    # Missing/broken images are recorded per sample; structural leakage is a hard error.
    structural = [
        e
        for e in validation["errors"]
        if "leakage" in e or "Duplicate sample_id" in e or "Conflicting" in e
    ]
    if structural:
        raise ValueError("; ".join(structural))
    if task not in {"line", "page"}:
        raise ValueError("task must be line or page")
    chosen = [r for r in rows if r["split"] == split and r["task"] == task]
    if not chosen:
        raise ValueError("No labeled samples match the requested split and task")
    image_hashes = {}
    for row in chosen:
        try:
            image_hashes[row["sample_id"]] = source_id(Path(row["image_path"]))
        except OSError:
            image_hashes[row["sample_id"]] = None
    results, raw, normalized, failures = [], [], [], []
    start = time.perf_counter()
    for row in chosen:
        try:
            prediction = engine.recognize(Path(row["image_path"]))
            ref, pred = row["transcription"], prediction["text"]
            raw.append((ref, pred))
            normalized.append((normalize(ref, nfc, whitespace), normalize(pred, nfc, whitespace)))
            results.append({"sample_id": row["sample_id"], "reference": ref, **prediction})
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"sample_id": row["sample_id"], "error": str(exc)})
    elapsed = time.perf_counter() - start
    latencies = [r["elapsed_seconds"] for r in results]
    result = {
        "schema_version": 1,
        "kind": "benchmark",
        "environment": provenance(),
        "dataset": {
            "manifest_sha256": source_id(manifest),
            "path": str(manifest.resolve()),
            "split": split,
            "task": task,
            "image_sha256": image_hashes,
            "annotation_methods": sorted({r["annotation_method"] for r in chosen}),
            "synthetic": any("synthetic" in r["annotation_method"].lower() for r in chosen),
        },
        "configuration": {
            "nfc": nfc,
            "collapse_whitespace": whitespace,
            "engine": type(engine).__name__,
            "engine_version": getattr(engine, "version", None),
            "language": getattr(engine, "language", None),
            "psm": getattr(engine, "psm", None),
        },
        "selected_samples": len(chosen),
        "evaluated_samples": len(results),
        "failed_samples": len(failures),
        "skipped_samples": len(rows) - len(chosen),
        "raw": metrics(raw),
        "normalized": metrics(normalized),
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": len(results) / elapsed,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "predictions": results,
        "failures": failures,
    }
    write_json(out / "benchmark.json", result)
    lines = [
        "# OCR benchmark",
        "",
        f"Task: {task}; split: {split}",
        "",
        f"Evaluated: {len(results)}; failures: {len(failures)}; "
        f"skipped: {result['skipped_samples']}",
        "",
        "Accuracy covers successful samples only. Inspect failures before comparing runs.",
        "",
        "| Text | CER | WER | Exact accuracy |",
        "|---|---:|---:|---:|",
    ]
    if result["dataset"]["synthetic"]:
        lines.insert(2, "SYNTHETIC MECHANICS TEST: not a real-world quality benchmark.")
    for mode in ("raw", "normalized"):
        m = result[mode]
        lines.append(f"| {mode} | {m['cer']} | {m['wer']} | {m['exact_accuracy']} |")
    lines += [
        "",
        f"Mean latency (seconds): {result['mean_latency_seconds']}",
        f"Throughput (samples/second): {result['throughput_samples_per_second']}",
        "",
        "Full predictions, engine configuration and provenance: benchmark.json",
    ]
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return result
