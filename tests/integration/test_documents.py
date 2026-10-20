import pytest
from PIL import Image
from pypdf import PdfWriter

from indic_ocr_research.documents import inspect_pdf, load_image, preprocess, render_pdf


def test_pdf(tmp_path):
    path = tmp_path / "image.pdf"
    Image.new("RGB", (200, 100), "white").save(path, "PDF", resolution=72)
    info = inspect_pdf(path)
    assert info["page_count"] == 1
    assert info["pages"][0]["classification"] == "image-only-or-no-text"
    pages = render_pdf(path, tmp_path / "pages", dpi=144, pages=[1])
    image = load_image(__import__("pathlib").Path(pages[0]["image_path"]))
    assert image.size == (400, 200)
    assert pages[0]["document_id"] == info["document_id"]
    with pytest.raises(ValueError, match="budget"):
        render_pdf(path, tmp_path / "huge", max_pixels=10)
    with pytest.raises(ValueError, match="out of range"):
        render_pdf(path, tmp_path / "bad", pages=[2])


def test_protected_and_empty(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    writer.encrypt("secret")
    path = tmp_path / "protected.pdf"
    writer.write(path)
    with pytest.raises(ValueError, match="password"):
        inspect_pdf(path)
    assert inspect_pdf(path, "secret")["page_count"] == 1
    assert len(render_pdf(path, tmp_path / "protected", password="secret")) == 1
    empty = tmp_path / "empty.pdf"
    PdfWriter().write(empty)
    with pytest.raises(ValueError, match="no pages"):
        inspect_pdf(empty)


def test_bad_files(tmp_path):
    for extension in ["pdf", "png"]:
        path = tmp_path / f"broken.{extension}"
        path.write_bytes(b"broken")
        with pytest.raises(ValueError):
            (inspect_pdf if extension == "pdf" else load_image)(path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_image(tmp_path / "file.txt")
    with pytest.raises(ValueError):
        render_pdf(tmp_path / "broken.pdf", tmp_path / "bad")


def test_preprocessing_and_tiff(tmp_path):
    path = tmp_path / "frames.tiff"
    image = Image.new("RGB", (100, 50), "white")
    image.save(path, save_all=True, append_images=[Image.new("RGB", (100, 50), "black")])
    assert load_image(path, frame=1).getpixel((0, 0)) == (0, 0, 0)
    result = preprocess(image, max_width=50, deskew=True)
    assert result.size == (50, 25) and result.mode == "L"
    with pytest.raises(ValueError, match="budget"):
        load_image(path, max_pixels=10)
