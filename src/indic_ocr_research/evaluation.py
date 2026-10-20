import math
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .common import normalize, provenance, write_json
from .datasets import read_manifest, validate_rows
from .documents import source_id
from .journal import PredictionJournal, fingerprint


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


def engine_identity(engine: Engine) -> dict:
    if callable(getattr(engine, "identity", None)):
        return engine.identity()
    return {
        "engine": type(engine).__module__ + "." + type(engine).__qualname__,
        "version": getattr(engine, "version", None),
        "language": getattr(engine, "language", None),
        "psm": getattr(engine, "psm", None),
    }


def _prediction(value: dict) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise ValueError("Engine returned invalid text")
    latency = value.get("elapsed_seconds")
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
        raise ValueError("Engine returned invalid elapsed_seconds")
    return value


def benchmark(
    manifest: Path,
    engine: Engine,
    out: Path,
    split: str = "test",
    task: str = "page",
    nfc: bool = True,
    whitespace: bool = True,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict:
    if task not in {"line", "block", "page"}:
        raise ValueError("task must be line, block, or page")
    if task not in getattr(engine, "supported_tasks", {"line", "block", "page"}):
        raise ValueError("Engine does not support the selected task")
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    rows = read_manifest(manifest, nfc=False)
    validation = validate_rows(rows)
    structural = [
        issue["message"] for issue in validation["issues"] if issue["code"] != "image_unreadable"
    ]
    if structural:
        raise ValueError("; ".join(structural))
    chosen = [r for r in rows if r["split"] == split and r["task"] == task]
    if not chosen:
        raise ValueError("No labeled samples match the requested split and task")
    image_hashes = {}
    for row in chosen:
        try:
            image_hashes[row["sample_id"]] = source_id(Path(row["image_path"]))
        except OSError:
            image_hashes[row["sample_id"]] = None
    descriptor = engine_identity(engine)
    if resume and not descriptor.get("version"):
        raise ValueError("Resume requires a stable engine version or checkpoint hash")
    configuration = {"nfc": nfc, "collapse_whitespace": whitespace, **descriptor}
    identity = {
        "manifest_sha256": source_id(manifest),
        "images": image_hashes,
        "configuration": configuration,
        "split": split,
        "task": task,
    }
    started = time.perf_counter()
    resumed_samples, session_attempts = 0, 0
    with PredictionJournal(out / "predictions.jsonl", identity, resume) as journal:
        if set(journal.latest) - set(image_hashes):
            raise ValueError("Journal contains samples outside the selected dataset")
        for row in chosen:
            sid = row["sample_id"]
            cached = journal.latest.get(sid)
            if cached and (cached["status"] == "success" or not retry_failed):
                resumed_samples += 1
                continue
            attempt_start = time.perf_counter()
            try:
                value = _prediction(engine.recognize(Path(row["image_path"])))
                event = {
                    "kind": "prediction",
                    "sample_id": sid,
                    "status": "success",
                    "prediction": value,
                }
            except (OSError, ValueError, RuntimeError) as exc:
                event = {
                    "kind": "prediction",
                    "sample_id": sid,
                    "status": "failure",
                    "error": str(exc),
                }
            event["attempt_seconds"] = time.perf_counter() - attempt_start
            journal.append(event)
            session_attempts += 1
        latest, events = dict(journal.latest), list(journal.events)
    raw, normalized, results, failures = [], [], [], []
    by_language = {}
    sample_errors = []
    for row in chosen:
        sid, language = row["sample_id"], row["language"]
        group = by_language.setdefault(
            language, {"raw_pairs": [], "pairs": [], "failed_samples": 0}
        )
        event = latest[sid]
        if event["status"] == "failure":
            failures.append({"sample_id": sid, "language": language, "error": event["error"]})
            group["failed_samples"] += 1
            continue
        prediction = _prediction(event["prediction"])
        ref, pred = row["transcription"], prediction["text"]
        pair = (normalize(ref, nfc, whitespace), normalize(pred, nfc, whitespace))
        raw.append((ref, pred))
        normalized.append(pair)
        group["raw_pairs"].append((ref, pred))
        group["pairs"].append(pair)
        results.append({"sample_id": sid, "reference": ref, "language": language, **prediction})
        score = metrics([pair])
        sample_errors.append(
            {
                "sample_id": sid,
                "language": language,
                "cer": score["cer"],
                "wer": score["wer"],
                "character_edits": score["character_edits"],
                "reference_characters": score["reference_characters"],
            }
        )
    grouped = {
        language: {
            "raw": metrics(group["raw_pairs"]),
            "normalized": metrics(group["pairs"]),
            "failed_samples": group["failed_samples"],
        }
        for language, group in by_language.items()
    }
    latencies = [r["elapsed_seconds"] for r in results]
    inference_elapsed = sum(e["attempt_seconds"] for e in events)
    result = {
        "schema_version": 2,
        "run_id": fingerprint(identity),
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
            "licenses": sorted({r["license"] for r in chosen}),
            "sources": sorted({r["source_url"] for r in chosen}),
            "attributions": sorted({r["attribution"] for r in chosen if r.get("attribution")}),
            "unverified_ground_truth_samples": sum(
                r.get("ground_truth_verified") is False for r in chosen
            ),
        },
        "configuration": configuration,
        "selected_samples": len(chosen),
        "evaluated_samples": len(results),
        "failed_samples": len(failures),
        "skipped_samples": len(rows) - len(chosen),
        "raw": metrics(raw),
        "normalized": metrics(normalized),
        "by_language": grouped,
        "elapsed_seconds": inference_elapsed,
        "session_elapsed_seconds": time.perf_counter() - started,
        "session_attempted_samples": session_attempts,
        "resumed_samples": resumed_samples,
        "total_attempts": len(events),
        "throughput_samples_per_second": len(results) / inference_elapsed
        if inference_elapsed
        else None,
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "latency_percentiles_seconds": {
            str(p): float(np.percentile(latencies, p)) if latencies else None for p in (50, 90, 95)
        },
        "worst_samples": sorted(
            sample_errors,
            key=lambda s: (
                s["character_edits"] / max(1, s["reference_characters"]),
                s["character_edits"],
            ),
            reverse=True,
        )[:20],
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
        f"Session attempts: {session_attempts}; reused records: {resumed_samples}",
        "",
        "Accuracy covers successful samples only. Inspect failures and dataset provenance.",
        "",
        "| Text | CER | WER | Exact accuracy |",
        "|---|---:|---:|---:|",
    ]
    if result["dataset"]["synthetic"]:
        lines.insert(2, "SYNTHETIC MECHANICS TEST: not a real-world quality benchmark.")
    if result["dataset"]["unverified_ground_truth_samples"]:
        lines.insert(2, "PILOT: publisher labels have not been independently human-verified.")
    for mode in ("raw", "normalized"):
        m = result[mode]
        lines.append(f"| {mode} | {m['cer']} | {m['wer']} | {m['exact_accuracy']} |")
    lines += [
        "",
        "| Language | Samples | Failed | Normalized CER | Normalized WER |",
        "|---|---:|---:|---:|---:|",
    ]
    for language, group in sorted(grouped.items()):
        score = group["normalized"]
        lines.append(
            f"| {language} | {score['samples']} | {group['failed_samples']} | "
            f"{score['cer']} | {score['wer']} |"
        )
    lines += [
        "",
        f"Mean latency (seconds): {result['mean_latency_seconds']}",
        f"Latency p95 (seconds): {result['latency_percentiles_seconds']['95']}",
        f"Cumulative throughput (samples/second): {result['throughput_samples_per_second']}",
        "",
        "Throughput uses all recorded inference attempts, including failed/retried attempts.",
        "Cache replay does not count as fresh inference. Full provenance: benchmark.json.",
    ]
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return result
