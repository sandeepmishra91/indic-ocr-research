import csv
import io
import json
from pathlib import Path


def write_table(rows: list[dict], out: Path, title: str, note: str) -> None:
    if not rows:
        raise ValueError("No rows to report")
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        cells = [str(row[c]).replace("|", "&#124;").replace("\n", " ") for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines), encoding="utf-8")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    out.with_suffix(".csv").write_text(stream.getvalue(), encoding="utf-8")


def training_history(history: list[dict], out: Path) -> None:
    rows = [
        {
            "epoch": h["epoch"],
            "train_loss": h["train_loss"],
            "validation_loss": h["validation_loss"],
            "cer": h["validation"]["cer"],
            "wer": h["validation"]["wer"],
            "exact": h["validation"]["exact_accuracy"],
        }
        for h in history
    ]
    write_table(
        rows, out, "Training history", "Validation on NFC code points and whitespace words."
    )


def compare(inputs: Path, out: Path) -> int:
    rows = []
    for path in sorted(inputs.rglob("*.json")):
        if path.name not in {"benchmark.json", "experiment.json"}:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("kind") == "benchmark":
            score = result["normalized"]
            row = {
                "run": str(path.parent),
                "kind": "benchmark",
                "task": result["dataset"]["task"],
                "dataset_sha256": result["dataset"]["manifest_sha256"],
                "split": result["dataset"]["split"],
                "configuration": result["configuration"],
                "samples": result["evaluated_samples"],
                "failed": result["failed_samples"],
                "cer": score["cer"],
                "wer": score["wer"],
                "exact": score["exact_accuracy"],
                "latency_seconds": result["mean_latency_seconds"],
                "synthetic": result["dataset"].get("synthetic", False),
            }
        elif result.get("kind") == "training" and result.get("history"):
            score = result["history"][-1]["validation"]
            row = {
                "run": str(path.parent),
                "kind": "training-validation",
                "task": "line",
                "dataset_sha256": result["dataset_id"],
                "split": "validation",
                "configuration": result["config"],
                "samples": score["samples"],
                "failed": 0,
                "cer": score["cer"],
                "wer": score["wer"],
                "exact": score["exact_accuracy"],
                "latency_seconds": None,
                "synthetic": result.get("synthetic", False),
            }
        else:
            continue
        row["configuration"] = json.dumps(row["configuration"], sort_keys=True)
        rows.append(row)
    if not rows:
        raise ValueError("No benchmark or training reports found")
    write_table(
        rows,
        out,
        "Experiment comparison",
        "Compare only matching dataset, split, task and normalization. "
        "Synthetic runs are mechanics tests, not quality benchmarks.",
    )
    return len(rows)
