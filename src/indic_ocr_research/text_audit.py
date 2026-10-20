"""Conservative script diagnostics; never silently remove samples or combining marks."""

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

    return {
        "reference_scripts": ref_counts,
        "prediction_scripts": pred_counts,
        "reference_expected_script_fraction": fraction(ref_counts),
        "prediction_expected_script_fraction": fraction(pred_counts),
        "empty_reference": not reference.strip(),
        "empty_prediction": not prediction.strip(),
    }
