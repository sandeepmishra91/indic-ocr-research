import hashlib
import math
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pypdf import PdfReader

from .common import write_json

MAX_PIXELS = 25_000_000


def source_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path, max_pixels: int = MAX_PIXELS, frame: int = 0) -> Image.Image:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    try:
        with Image.open(path) as image:
            image.seek(frame)
            if image.width * image.height > max_pixels:
                raise ValueError(f"Image exceeds {max_pixels} pixel budget")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, EOFError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def preprocess(
    image: Image.Image, grayscale: bool = True, max_width: int | None = None, deskew: bool = False
) -> Image.Image:
    image = image.convert("L" if grayscale or deskew else "RGB")
    if max_width is not None:
        if max_width < 1:
            raise ValueError("max_width must be positive")
        if image.width > max_width:
            image = image.resize((max_width, max(1, round(image.height * max_width / image.width))))
    if deskew:
        import numpy as np

        scores = []
        angles = np.arange(-5, 5.01, 0.5)
        for angle in angles:
            rows = (np.asarray(image.rotate(float(angle), fillcolor=255)) < 128).sum(axis=1)
            scores.append(float(np.square(np.diff(rows.astype(float))).sum()))
        best = int(np.argmax(scores))
        center = len(angles) // 2
        if scores[best] > scores[center] * 1.05 and best not in (0, len(angles) - 1):
            image = image.rotate(float(angles[best]), expand=True, fillcolor=255)
    return image


def inspect_pdf(path: Path, password: str | None = None) -> dict:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(password or ""):
            raise ValueError("PDF is password protected; supply a valid password")
        if not reader.pages:
            raise ValueError("PDF contains no pages")
        pages = []
        for index, page in enumerate(reader.pages):
            chars = len("".join((page.extract_text() or "").split()))
            pages.append(
                {
                    "page": index + 1,
                    "width_points": float(page.mediabox.width),
                    "height_points": float(page.mediabox.height),
                    "text_characters": chars,
                    "classification": "text-bearing" if chars else "image-only-or-no-text",
                }
            )
        return {
            "document_id": source_id(path),
            "source": str(path.resolve()),
            "page_count": len(pages),
            "pages": pages,
        }
    except (OSError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Cannot inspect PDF: {exc}") from exc


def render_pdf(
    path: Path,
    output: Path,
    dpi: int = 200,
    pages: list[int] | None = None,
    password: str | None = None,
    max_pixels: int = MAX_PIXELS,
) -> list[dict]:
    import pypdfium2 as pdfium

    if not 36 <= dpi <= 600:
        raise ValueError("DPI must be between 36 and 600")
    try:
        pdf = pdfium.PdfDocument(path, password=password)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc
    records = []
    try:
        selected = list(range(1, len(pdf) + 1)) if pages is None else pages
        if not selected or any(p < 1 or p > len(pdf) for p in selected):
            raise ValueError("Empty PDF/selection or page number out of range")
        output.mkdir(parents=True, exist_ok=True)
        doc_id = source_id(path)
        for number in selected:
            page = pdf[number - 1]
            try:
                width, height = page.get_size()
                scale = dpi / 72
                if math.ceil(width * scale) * math.ceil(height * scale) > max_pixels:
                    raise ValueError("Rendered page exceeds pixel budget; reduce DPI")
                bitmap = page.render(scale=scale)
                try:
                    target = output / f"{doc_id[:16]}-page-{number:04d}.png"
                    bitmap.to_pil().save(target)
                finally:
                    bitmap.close()
                records.append(
                    {
                        "document_id": doc_id,
                        "page": number,
                        "dpi": dpi,
                        "source": str(path.resolve()),
                        "image_path": str(target.resolve()),
                    }
                )
            finally:
                page.close()
    finally:
        pdf.close()
    write_json(output / "pages.json", records)
    return records
