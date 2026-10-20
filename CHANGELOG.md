# Changelog

## 0.1.0

Initial offline CPU OCR research framework: documents, manifests, metrics, Tesseract
adapter, CRNN-CTC training/inference, local reports, device diagnostics and CPU CI.
Real-world accuracy and GPU/ARM benchmarks remain unmeasured.

Pre-merge review: bound Tesseract to its validated image frame with temporary PNG
cleanup, include preparation in latency, validate training settings before allocation,
and skip absent AMP scaler state when restoring full-precision checkpoints.
