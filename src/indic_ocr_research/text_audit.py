"""Conservative script diagnostics; never silently remove samples or combining marks."""

import re
import unicodedata
from collections import Counter

EXPECTED = {
    "hin": "Devanagari",
    "hi": "Devanagari",
    "mal": "Malayalam",
    "ml": "Malayalam",
    "eng": "Latin",
    "en": "Latin",
}


def script_counts(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for character in text:
        if unicodedata.category(character)[0] not in {"L", "M"}:
            continue
        name = unicodedata.name(character, "")
        script = next(
            (s for s in ("Devanagari", "Malayalam", "Latin") if s.upper() in name), "Other"
        )
        counts[script] += 1
    return dict(counts)


def audit_pair(reference: str, prediction: str, language: str) -> dict:
    ref_counts, pred_counts = script_counts(reference), script_counts(prediction)
    expected = EXPECTED.get(language)

    def fraction(counts: dict) -> float | None:
        return counts.get(expected, 0) / sum(counts.values()) if counts and expected else None

    flags = []
    lowered = reference.casefold()
    meta_markers = (
        "image analysis",
        "line-by-line transcription",
        "the user has provided an image",
    )
    if (
        re.search(r"^\s*(?:thought\b|<think>)", reference, re.IGNORECASE)
        or sum(marker in lowered for marker in meta_markers) >= 2
    ):
        flags.append("possible-annotation-meta-text")
    ratio = fraction(ref_counts)
    if ratio is not None and ratio < 0.5:
        flags.append("low-expected-script-fraction")
    return {
        "reference_review_flags": flags,
        "reference_scripts": ref_counts,
        "prediction_scripts": pred_counts,
        "reference_expected_script_fraction": fraction(ref_counts),
        "prediction_expected_script_fraction": fraction(pred_counts),
        "empty_reference": not reference.strip(),
        "empty_prediction": not prediction.strip(),
    }
