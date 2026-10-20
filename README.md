# Indic OCR Research Lab

Offline, open-source OCR research for scanned Indian documents. Python 3.11+.
Hindi and Malayalam are initial research priorities. **No measured Indian-language
accuracy, trained production model, ARM performance, or MI300X results are claimed.**
The 90% word recognition goal and expansion to 22 languages are research targets.

## Implemented

- PDF inspection and selected-page rendering with page metadata and pixel limits.
- PNG/JPEG/TIFF loading, grayscale, aspect-preserving resize and conservative small-angle deskew.
- Offline Tesseract adapter with executable/language checks and explicit failure reporting.
- JSONL validation, image duplicate/conflict detection, document splits and leakage checks.
- Raw and normalized CER/WER, exact accuracy, measured latency, throughput and failures.
- Real cropped-line CRNN (CNN + bidirectional LSTM + CTC), Unicode vocabulary,
  CPU training/inference, validation, gradient clipping, AMP on GPUs and epoch resume.
- Local JSON/Markdown benchmark reports, CSV comparisons and training experiment history.
- CPU/CUDA/ROCm diagnostics and an executable GPU validation procedure.

These are working research foundations, not a full-page neural document detector.
See [local verification](docs/verification.md) for actual execution status.

## Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv/Scripts/Activate.ps1
python -m pip install -e ".[dev]"
indicocr doctor
```

If the console script directory is absent from PATH, use
`python -m indic_ocr_research.cli` instead of `indicocr` in every example.
The basic install does not require PyTorch or a GPU. For CPU training, install
an appropriate CPU build of PyTorch from [the official selector](https://pytorch.org/get-started/locally/)
then `python -m pip install -e ".[train,dev]"`. On ARM64, check wheel availability
for the exact Python/OS combination; source builds may be necessary. No ARM execution
has been performed here. Do not install GPU drivers on the ARM production server.

Tesseract is a separate system dependency. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mal
```

On Windows install upstream Tesseract-compatible binaries and add the executable
to PATH. Verify installed packs with `tesseract --list-langs`. English (`eng`),
Hindi (`hin`) and Malayalam (`mal`) are accepted when installed; no real-world
language accuracy has yet been independently validated. Other installed packs
can be selected by name. Tesseract is an existing baseline, not our trained model.

## Documents and baseline

```bash
indicocr pdf inspect document.pdf
indicocr pdf render --input document.pdf --output-dir outputs/pages --dpi 200 --page 1
indicocr baseline run --input outputs/page.png --lang hin+eng --psm 3
```

`--page` is 1 based and repeatable; omit it to render every page. Rendering creates
SHA-prefixed filenames and `pages.json`; use the actual emitted image path for OCR.
Password-protected PDFs accept `--password`. Raster TIFF loading defaults to the
first frame; Python `load_image(..., frame=N)` selects other frames. PDF text
classification only checks extractable non-whitespace characters; blank/vector
pages may also have no text, and invisible OCR layers count as text-bearing.

## Supply and validate data

See [dataset schema and annotation guidance](docs/datasets.md). Paths in manifests
resolve relative to the JSONL file. No third-party PDFs or labels are included.

```bash
indicocr manifest validate --path data/lines.jsonl --out outputs/validation.json
indicocr manifest split --path data/lines.jsonl --out outputs/split.jsonl --seed 42
```

Splits use document IDs, never random individual lines; validate the resulting
split file to detect duplicate-content leakage. Default proportions are 80/10/10
of documents. Small corpora can have empty splits; training requires both train
and validation samples. Multiple lines sharing one document ID are expected.

## Train and recognize

For genuine research, populate `data/lines.jsonl` with licensed human-annotated
cropped lines and use:

```bash
indicocr train --config configs/crnn_cpu.yaml
indicocr recognize --checkpoint outputs/crnn_cpu/model.pt --image data/line.png --device cpu
indicocr train --config configs/crnn_cpu.yaml --resume outputs/crnn_cpu/model.pt
```

To resume, increase `epochs` above the completed epoch count. Other training settings
and dataset fingerprints must match. Config paths resolve relative to the YAML file.
Vocabulary is learned from training labels only. Validation characters absent from
that vocabulary produce an error; adjust training coverage rather than using test labels.
The recognizer uses NFC Unicode code points, not ASCII. It does not implement
language-aware grapheme segmentation, detection, page layout or reading-order recovery.

A fully reproducible **synthetic mechanics test** is available:

```bash
python examples/make_smoke_data.py --out outputs/smoke-data
indicocr train --config configs/smoke.yaml
indicocr recognize --checkpoint outputs/smoke/model.pt --image outputs/smoke-data/line-4.png --device cpu
indicocr benchmark run --manifest outputs/smoke-data/lines.jsonl --engine crnn --checkpoint outputs/smoke/model.pt --task line --out outputs/smoke-benchmark
indicocr report compare --inputs outputs --out outputs/comparison.md
```

The generated labels are synthetic, not human ground truth. This runs actual training
and inference but says nothing about recognition quality on real documents.

## Evaluate real documents

```bash
indicocr benchmark run --manifest data/eval.jsonl --engine tesseract --lang hin+eng --task page --split test --out reports/tesseract
indicocr benchmark run --manifest data/lines.jsonl --engine crnn --checkpoint outputs/crnn_cpu/model.pt --task line --out reports/crnn
indicocr report compare --inputs reports --out reports/comparison.md
```

Use human labels and matching `task` metadata. `--no-nfc` and `--no-whitespace`
control normalized scoring; raw scoring always preserves original labels/predictions.
No empty-dataset accuracy is emitted. Failures are recorded and CLI exits nonzero;
accuracy covers successful samples only. Never compare line and page scores as though
they measure the same problem. Details: [benchmark methodology](docs/benchmark-methodology.md).

## Development and research

```bash
python -m ruff check .
python -m pytest -q
```

CI runs core tests without torch, and CPU training plus real English Tesseract integration
in a separate job. No passing CI badge is claimed until a hosted run actually succeeds.
See [architecture](docs/architecture.md), [reproducibility](docs/reproducibility.md),
[ROCm guide](docs/amd-rocm.md), [limitations](docs/limitations.md), and
[research plan](docs/research-plan.md). Pretrained backbones, document layout,
grapheme tokenization, stronger augmentation, quantization and multi-language quality
studies remain planned. Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

Our source is Apache-2.0. Dependencies retain their own licenses; see
[third-party notices](docs/third-party-licenses.md). This repository is independent
of any private production API and uses no paid OCR APIs.
