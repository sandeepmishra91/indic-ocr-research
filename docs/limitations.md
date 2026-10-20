# Limitations

- No licensed real-world corpus or pretrained weights are shipped. No independently human-audited or representative Hindi/Malayalam
  accuracy, 22-language support level, 90% result, ARM speedup or MI300X result exists.
- CRNN recognizes cropped horizontal lines. No neural detection, layout analysis,
  table/form extraction, reading-order recovery, handwriting or orientation classifier.
- Unicode code points preserve combining marks but do not model grapheme clusters.
  Validation characters unseen in training cause an error. Greedy decoding has no
  language model or lexicon. Long lines are rejected rather than silently squashed.
- Deskew only searches +/-5 degrees by projection, conservatively. Complex layouts
  and sparse/noisy pages may fail; it is opt-in through the Python preprocessing API.
- PDF no-text heuristics cannot prove a page is a scan. Extracted text may be garbled.
  Text extraction and PDF decoding can be expensive on malicious files. Pixel budgets
  bound raster output allocation, not every internal PDF parser/decompression allocation.
  Use OS resource isolation for untrusted documents; no hard parser memory sandbox exists.
- TIFF CLI baseline uses the first frame. Other frames require explicit Python loading.
- Byte hashing catches identical bytes, not visually identical re-encoded images.
  Automatic splits are not language-stratified and can produce empty small splits.
- Benchmark accuracy excludes failed inference; compare coverage as well as scores.
  Code-point CER and whitespace WER are not grapheme/morphological word accuracy.
- Epoch resume is supported; mid-batch resume and cross-version determinism are not.
  Single-process loading and the small CNN are baseline choices, not throughput tuning.
- Basic package install is CPU compatible; torch/renderer wheel availability depends
  on OS/architecture. GPU support is code readiness until tested on actual hardware.
- Hosted CI and a remote PR require a configured GitHub repository and credentials.

- The optional 20-block Hugging Face pilot has unverified publisher labels and source
  document grouping. It supports diagnostics, not a representative quality estimate.
- Benchmark journals support interruption recovery with a single writer per output
  directory. They are not a concurrent database or distributed experiment scheduler.
