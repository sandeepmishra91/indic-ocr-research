# Execution record: 20 October 2026

## Local environment

Windows 11 AMD64, Python 3.12.5. No usable GPU and no Tesseract executable on PATH.
No ARM64 or MI300X execution, genuine Indian-document quality benchmark, or hosted
CI result is claimed by this local record.

## Actually executed

| Check | Result |
|---|---|
| Fresh virtual environment, `pip install -e ".[dev]"` | Passed, with no torch installed |
| Fresh environment core tests | 34 passed, 1 skipped (Tesseract absent) |
| Separate environment with CPU torch 2.7.1 | 38 passed, 1 skipped (Tesseract absent) |
| Ruff lint | Passed |
| CLI help and doctor | Passed; correctly reported CPU and missing Tesseract |
| PDF inspect/render CLI | Passed on a generated synthetic image PDF; 144 DPI output 400x200 |
| PDF integration | Passed: selected pages, dimensions, password, empty/corrupt files, budgets |
| Manifest validate CLI | Passed on 5 explicitly synthetic samples from 5 documents |
| Synthetic CPU training CLI | Completed 1 epoch and wrote a reloadable checkpoint |
| Checkpoint recognize CLI | Executed actual inference; blank prediction after 1 smoke epoch |
| Train/resume tests | Passed; weights equal uninterrupted CPU training; changed images rejected |
| CRNN benchmark CLI | Generated JSON/Markdown from actual smoke-model predictions |
| Report compare CLI | Generated Markdown and CSV |
| GPU validation script | Exited nonzero as intended: no usable GPU; GPU validation not run |
| Actual Tesseract OCR locally | Not run: executable absent; integration test skipped |

Fresh core dependency versions: Pillow 13.0.0, NumPy 2.5.4, pypdf 6.21.1,
pypdfium2 5.14.0, Typer 0.27.3, PyYAML 6.0.3, pytest 9.1.1, Ruff 0.17.1.
Training environment reused installed CPU torch 2.7.1 and older compatible Pillow,
NumPy and PDFium versions; pypdf and Ruff were resolved to declared requirements.
These are observations, not a universal dependency lockfile.

## Reproduce from the repository

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python examples/make_smoke_data.py --out outputs/smoke-data
# Install a compatible CPU torch build before training:
python -m indic_ocr_research.cli train --config configs/smoke.yaml
python -m indic_ocr_research.cli recognize --checkpoint outputs/smoke/model.pt --image outputs/smoke-data/line-4.png --device cpu
python -m indic_ocr_research.cli benchmark run --manifest outputs/smoke-data/lines.jsonl --engine crnn --checkpoint outputs/smoke/model.pt --task line --out outputs/smoke-benchmark
python -m indic_ocr_research.cli report compare --inputs outputs/smoke-benchmark --out outputs/comparison.md
```

All smoke images/labels are generated, not real-world benchmark ground truth.
They verify mechanics only; no accuracy result is published from them here.
Local outputs remain ignored by Git. For restricted temporary-directory permissions,
create outputs/ and select a new disposable pytest `--basetemp` inside it.

## Source layout

```text
.github/workflows/ci.yml
configs/                 # CPU training, smoke training, benchmark reference settings
data/README.md           # Bring-your-own-data policy
docs/                    # Architecture, methodology, licenses, ROCm, verification
examples/make_smoke_data.py
scripts/validate_gpu.py
src/indic_ocr_research/
  cli.py                 # Implemented command surface
  common.py              # Normalization, provenance, diagnostics
  documents.py           # PDF and raster processing
  datasets.py            # Manifests, validation and document splitting
  engines.py             # Offline Tesseract adapter
  model.py               # Unicode CRNN and CTC tensor handling
  training.py            # Training, validation, resume and inference adapter
  evaluation.py          # Edit distance and measured benchmark runs
  reporting.py           # Local JSON-derived Markdown and CSV tables
tests/unit/
tests/integration/
pyproject.toml
README.md, LICENSE, NOTICE, AGENTS.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md
```

## Remaining validation

Run the real English Tesseract integration in Linux CI, then supply licensed human
Hindi/Malayalam annotations for quality evaluation. Verify ARM installation and CPU
latency on the actual host. Validate ROCm/CUDA forward/CTC/backward/AMP on actual
compatible hardware before any acceleration claim. Milestone 4 extensions remain
planned; see research-plan.md and limitations.md. Workflow run status is available
on the pull request; this file records local execution only.
