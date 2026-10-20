# Benchmark methodology

Supply real, licensed documents with human-provided labels for quality conclusions.
Synthetic datasets are only mechanics tests and must be reported as such. Select
one task (page or line) and one split per run. Prefer a stable corpus snapshot;
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
Throughput is successful samples divided by wall time of the inference loop,
including failures. No warmup or batched throughput is claimed. Environment,
versions, configuration, dataset identity, actual predictions and failures are
retained in JSON. Markdown gives readable scores. Comparison CSV/Markdown includes
task, dataset and normalization settings; inspect these before drawing conclusions.

Broken sample inference is recorded with its sample ID. Structural leakage and
conflicting labels abort the benchmark. Failed samples are excluded from accuracy
but counted; inspect coverage alongside accuracy. All failures yield null accuracy,
not a successful zero-error score. Unselected task/split rows are counted as skipped.
No matching labeled samples is an error. Reports may contain sensitive text and
absolute paths; they are ignored by Git. Audit reports before sharing.
