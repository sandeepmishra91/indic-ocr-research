# Third-party license review

Our original source is Apache-2.0; LICENSE contains the full Apache text downloaded
from the Apache Software Foundation. No external source code, model weights, fonts
or scanned documents have been vendored. Synthetic images use Pillow's built-in
font at runtime. There is no automatic data or weight download.

The dependency metadata and upstream license sources were reviewed during initial
implementation. This is an inventory, not a replacement for distribution notices.
Check licenses for the exact wheel/platform and its bundled native libraries before
redistributing an application image or binary bundle.

| Component | License / upstream notice |
|---|---|
| Pillow | [HPND](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| NumPy | [BSD-3-Clause and bundled library notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pypdf | [BSD-3-Clause](https://github.com/py-pdf/pypdf/blob/main/LICENSE) |
| pypdfium2 | [Apache-2.0 OR BSD-3-Clause, plus PDFium third-party notices](https://github.com/pypdfium2-team/pypdfium2/blob/main/LICENSE) |
| PDFium bundled renderer | [BSD-style and third-party components](https://github.com/pypdfium2-team/pypdfium2/tree/main/BUILD_LICENSES) |
| Typer | [MIT](https://github.com/fastapi/typer/blob/master/LICENSE) |
| PyYAML | [MIT](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| PyTorch, optional | [BSD-3-Clause and third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Tesseract, external executable | [Apache-2.0](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) |
| Tesseract language data, user installed | [Review chosen tessdata distribution](https://github.com/tesseract-ocr/tessdata/blob/main/LICENSE) |
| pytest, development only | [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| Ruff, development only | [MIT](https://github.com/astral-sh/ruff/blob/main/LICENSE) |

Transitive packages (including Click, Rich, torch runtime libraries and renderer
binaries) retain their own notices. No claim is made that a source license alone
covers every accelerator/runtime image. Datasets and trained weights require a
separate documented license and provenance review. None are included in this release.
