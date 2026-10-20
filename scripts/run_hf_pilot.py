"""Run a tiny publisher-labeled block OCR pilot; no neural training or model downloads."""

import argparse
import json
import os
from pathlib import Path

from indic_ocr_research.common import provenance, write_json
from indic_ocr_research.datasets import read_manifest
from indic_ocr_research.engines import Tesseract
from indic_ocr_research.evaluation import benchmark
from indic_ocr_research.hub import import_dataset
from indic_ocr_research.reporting import compare, write_table


def run(
    config: Path,
    data: Path,
    cache: Path,
    out: Path,
    offline: bool = False,
    resume: bool = False,
    psms: tuple[int, ...] = (3, 6),
) -> dict:
    receipt = import_dataset(config, data, cache, offline=offline, env_file=Path(".env"))
    rows = read_manifest(data / "manifest.jsonl", nfc=False)
    if receipt["spec"]["task"] != "block":
        raise ValueError("This pilot runner expects block-level evaluation data")
    runs = []
    for language in sorted({row["language"] for row in rows}):
        subset = out / "inputs" / f"{language}.jsonl"
        subset.parent.mkdir(parents=True, exist_ok=True)
        selected = [r for r in rows if r["language"] == language]
        subset.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8"
        )
        for policy, requested in [
            ("indic-only", language),
            ("indic-plus-english", f"{language}+eng"),
        ]:
            for psm in psms:
                result = benchmark(
                    subset,
                    Tesseract(requested, psm),
                    out / "runs" / f"{language}-{policy}-psm{psm}",
                    task="block",
                    resume=resume
                    and (
                        out / "runs" / f"{language}-{policy}-psm{psm}" / "predictions.jsonl"
                    ).is_file(),
                )
                runs.append(
                    {
                        "language": language,
                        "language_policy": policy,
                        "psm": psm,
                        "run_id": result["run_id"],
                        "samples": result["evaluated_samples"],
                        "failed": result["failed_samples"],
                        "raw": result["raw"],
                        "normalized": result["normalized"],
                        "mean_latency_seconds": result["mean_latency_seconds"],
                        "p95_latency_seconds": result["latency_percentiles_seconds"]["95"],
                        "throughput_samples_per_second": result["throughput_samples_per_second"],
                        "engine": result["configuration"],
                        "diagnostics": result["diagnostics"],
                        "quality_status": result["quality_status"],
                        "sample_diagnostics": result["sample_diagnostics"],
                        "worst_samples": result["worst_samples"],
                        "dataset_sha256": result["dataset"]["manifest_sha256"],
                        "image_sha256": result["dataset"]["image_sha256"],
                    }
                )
    summary = {
        "kind": "publisher-labeled-small-pilot",
        "environment": provenance(),
        "cpu_count": os.cpu_count(),
        "omp_thread_limit": os.getenv("OMP_THREAD_LIMIT"),
        "source": {
            key: receipt["spec"][key]
            for key in ("repo_id", "revision", "license", "license_url", "attribution")
        },
        "artifacts": receipt["artifacts"],
        "runs": runs,
        "limitations": [
            "Ten convenience-sampled blocks per language, not representative.",
            "Publisher labels have not been independently human-verified.",
            "Analysis/meta-text in references needs human correction before accuracy claims.",
            "Original document grouping is unverified; blocks may be correlated.",
            "Not a full-page or cropped-line benchmark; no 90% claim.",
            "PSM/language-pack comparisons are exploratory; keep a separate test corpus.",
            "Latency includes loading and preprocessing; no warmup protocol.",
        ],
    }
    write_json(out / "summary.json", summary)
    table = [
        {
            "language": r["language"],
            "psm": r["psm"],
            "language_policy": r["language_policy"],
            "empty_predictions": r["diagnostics"]["empty_predictions"],
            "samples": r["samples"],
            "failed": r["failed"],
            "normalized_cer": r["normalized"]["cer"],
            "normalized_wer": r["normalized"]["wer"],
            "exact_accuracy": r["normalized"]["exact_accuracy"],
            "mean_latency_seconds": r["mean_latency_seconds"],
        }
        for r in runs
    ]
    write_table(
        table,
        out / "summary.md",
        "Small IndicDLP block pilot",
        "Publisher labels, not independently human-verified. "
        "Ten blocks per language cannot establish production accuracy. "
        f"Source: {summary['source']['license_url']}; "
        f"attribution: {summary['source']['attribution']}; CC-BY-4.0.",
    )
    compare(out / "runs", out / "comparison.md")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hf_indicdlp.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/indicdlp-pilot"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/hf-cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/hf-pilot"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(args.config, args.data, args.cache, args.out, args.offline, args.resume)
    print(json.dumps({"runs": len(summary["runs"]), "summary": str(args.out / "summary.json")}))
    if any(r["failed"] for r in summary["runs"]):
        raise SystemExit(1)
