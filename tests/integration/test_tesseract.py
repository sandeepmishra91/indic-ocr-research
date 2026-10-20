import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from indic_ocr_research.engines import Tesseract


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_real_tesseract(tmp_path):
    path = tmp_path / "synthetic.png"
    image = Image.new("RGB", (500, 100), "white")
    ImageDraw.Draw(image).text(
        (10, 20), "HELLO WORLD", fill="black", font=ImageFont.load_default(size=40)
    )
    image.save(path)
    result = Tesseract("eng", 7).recognize(path)
    assert "HELLO" in result["text"].upper()
    assert result["elapsed_seconds"] > 0
    assert result["version"]
