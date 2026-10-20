# Research plan

Milestone 1 implements CPU documents, offline baseline, manifests, metrics, CLI and CI.
Milestone 2 implements a real small line CRNN, validation, checkpoint resume and inference.
Milestone 3 implements device diagnostics, local tracking and GPU validation procedures.
Execution coverage and unavailable hardware are recorded in verification.md.

Next collect explicitly licensed, human-reviewed Hindi and Malayalam scanned lines
and pages. Freeze document-level train/validation/test partitions; quantify annotator
agreement and language/script distributions. Establish Tesseract page and line baselines
separately, then train the CRNN on line data and measure held-out errors, including
combining marks, conjuncts, numerals, punctuation and failures. Evaluate grapheme-cluster
tokens against code points under identical splits and normalization before adopting them.

Milestone 4 is planned: licensed pretrained backbones, document integration, improved
preprocessing, line detection, augmentation, ARM CPU optimization/quantization, measured
ROCm experiments and per-language evaluation. Add one tested extension at a time.
The research target exceeds 90% word recognition accuracy on specified corpora, with
confidence intervals and failure coverage. It is not a current achievement or guarantee.
