# Changelog

## Unreleased

- Optional small Hugging Face imports with commit/license/size checks, local token loading,
  cached offline reuse, source attribution and publisher-label warnings.
- Durable benchmark resume, failure-only retries, per-language scores, latency percentiles,
  stable engine identities and traineddata hashes; explicit block-level evaluation.
- Non-destructive label audit and review queue after a genuine publisher-label defect
  was found; quality warnings prevent presenting those scores as validated accuracy.
- Opt-in hosted Hindi/Malayalam pilot comparing existing Tesseract PSM 3 and 6 without
  model downloads or publishing source images/labels.

## 0.1.0

Initial offline CPU OCR research framework: documents, manifests, metrics, Tesseract
adapter, CRNN-CTC training/inference, local reports, device diagnostics and CPU CI.
Real-world accuracy and GPU/ARM benchmarks remain unmeasured.

Pre-merge review: bound Tesseract to its validated image frame with temporary PNG
cleanup, include preparation in latency, validate training settings before allocation,
and skip absent AMP scaler state when restoring full-precision checkpoints.
