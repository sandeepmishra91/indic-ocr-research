"""Generate a review queue without altering, filtering or claiming to verify ground truth."""

import json
from pathlib import Path

from .common import write_json
from .datasets import read_manifest
from .documents import source_id
from .text_audit import audit_pair


def audit_manifest(path: Path, out: Path, review_manifest: Path | None = None) -> dict:
    if out.resolve() == path.resolve() or (
        review_manifest and review_manifest.resolve() == path.resolve()
    ):
        raise ValueError("Audit outputs must not overwrite the source manifest")
    if review_manifest and out.resolve() == review_manifest.resolve():
        raise ValueError("Audit report and review manifest need distinct output paths")
    rows = read_manifest(path, nfc=False)
    samples, flagged = [], []
    for row in rows:
        audit = audit_pair(row["transcription"], "", row["language"])
        flags = audit["reference_review_flags"]
        samples.append(
            {
                "sample_id": row["sample_id"],
                "language": row["language"],
                "label_characters": len(row["transcription"]),
                "label_words": len(row["transcription"].split()),
                "expected_script_fraction": audit["reference_expected_script_fraction"],
                "review_flags": flags,
            }
        )
        if flags:
            flagged.append(row | {"review_flags": flags, "review_status": "needs-human-review"})
    result = {
        "kind": "label-audit",
        "manifest_sha256": source_id(path),
        "samples": len(rows),
        "flagged_samples": len(flagged),
        "sample_audits": samples,
        "status": "needs-human-review" if flagged else "no-heuristic-flags",
        "notice": "Heuristics do not verify labels. No samples have been corrected or excluded.",
    }
    write_json(out, result)
    if review_manifest:
        review_manifest.parent.mkdir(parents=True, exist_ok=True)
        review_manifest.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in flagged), encoding="utf-8"
        )
    return result
