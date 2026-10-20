# Benchmark methodology

Supply real, licensed documents with human-provided labels for quality conclusions.
Synthetic datasets are only mechanics tests and must be reported as such. Select
one task (page, block or line) and one split per run. Prefer a stable corpus snapshot;
retain manifest and image hashes with your experiment archive. Manifest SHA identifies
labels and paths; selected per-sample image byte hashes are saved separately in each report.

Levenshtein distance counts unit substitutions, deletions and insertions. CER sums
code-point edits over reference code points across the corpus. WER uses whitespace
word tokens and sums word edits over reference words. Both can exceed 1. NFC merges
canonical equivalent sequences; whitespace normalization strips and collapses spacing.
Raw metrics preserve all code points, including newlines and trailing OCR whitespace.
Normalized metrics apply the selected NFC/whitespace policy to both sides. No case
folding, punctuation removal, Indic mark deletion or transliteration is performed.

Combining marks count separately in code-point CER; a grapheme error can therefore
count as several edits. Whitespace WER may not align with linguistic word boundaries.
A future grapheme metric must be labeled separately and not silently replace CER.
For an aggregate with no reference characters/words the respective rate is null,
including all-blank references. Insertions still appear in edit counts. Empty
predictions against nonempty references count all reference units as deletions.
Exact accuracy is the proportion of equal complete strings, including blank pairs.

Engine inference is actually invoked once per selected sample. Latency includes
image loading/preprocessing and execution as measured by the adapter, excluding
engine construction/checkpoint loading. GPU synchronization occurs before returning.
Throughput is unique successful samples divided by cumulative measured inference-attempt
wall time, including failed/retried attempts. Journal I/O, setup and cache replay are excluded. No warmup or batched throughput is claimed. Environment,
versions, configuration, dataset identity, actual predictions and failures are
retained in JSON. Markdown gives readable scores. Comparison CSV/Markdown includes
task, dataset and normalization settings; inspect these before drawing conclusions.

Broken sample inference is recorded with its sample ID. Structural leakage and
conflicting labels abort the benchmark. Failed samples are excluded from accuracy
but counted; inspect coverage alongside accuracy. All failures yield null accuracy,
not a successful zero-error score. Unselected task/split rows are counted as skipped.
No matching labeled samples is an error. Reports may contain sensitive text and
absolute paths; they are ignored by Git. Audit reports before sharing.

## Durable resume and detailed reports

Schema version 2 adds per-language raw/normalized scores, failure counts, interpolated
p50/p90/p95 successful-inference latencies and up to 20 worst sample IDs by normalized
CER. Inspect those images and labels before attributing errors to the model.

`predictions.jsonl` is flushed and fsynced after each completed inference attempt.
The header binds manifest SHA, selected image hashes, engine identity, split/task and
normalization settings. `--resume` refuses any mismatch. Tesseract identity includes
PSM, requested languages, timeout, preprocessing policy and traineddata SHA256 when
its directory is discoverable. CRNN identity includes checkpoint SHA, preprocessing,
device and torch version. Do not modify engine implementations without updating their
identity/version. A custom engine needs a stable version for resume.

Interrupted runs retain completed attempts. A torn final journal fragment can be
removed; corruption elsewhere is an error. Resume keeps failures unless `--retry-failed`
is explicitly supplied. Reports distinguish session attempts from reused records and
retain cumulative attempt timing; replaying cached predictions cannot inflate throughput.
Use one writer per output directory. Existing results are protected from accidental
overwrite; new experiments require new directories. Journals and detailed reports
contain sensitive text and are ignored alongside other outputs.

Publisher-provided data imported from Hugging Face is labeled unverified. The small
IndicDLP pilot uses block-level metrics, explicit CC-BY-4.0 attribution, source links,
commit hashes and a nonrepresentative-sample warning. It is not a substitute for
independently human-audited ground truth or production scan evaluation.

Script diagnostics count Unicode letters/marks whose names identify Devanagari,
Malayalam or Latin; remaining letters/marks are grouped as Other. This is a limited
name-based heuristic, not a complete Unicode Script-property implementation or a
language detector. Mixed-language and numeric captions can be valid. Low expected-script
fractions and empty predictions are reported for investigation, never filtered out
or used to remove combining marks from scoring.
