import re
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .documents import load_image, source_id


class Tesseract:
    def __init__(self, language: str | None = "eng", psm: int = 3, timeout: int = 120):
        self.executable = shutil.which("tesseract")
        if not self.executable:
            raise RuntimeError("Tesseract executable missing; install it and add it to PATH")
        if psm not in range(14) or timeout <= 0:
            raise ValueError("PSM must be 0..13 and timeout positive")
        self.language, self.psm, self.timeout = language, psm, timeout
        self.version = self._run(["--version"]).splitlines()[0]
        listing = self._run(["--list-langs"]).splitlines()
        self.languages = [line.strip() for line in listing[1:]]
        directory = re.search(r'"([^"\n]+)"', listing[0]) if listing else None
        self.tessdata_dir = Path(directory.group(1)) if directory else None
        missing = set(language.split("+")) - set(self.languages) if language else set()
        if missing:
            raise RuntimeError(f"Missing Tesseract language packs: {sorted(missing)}")

    def identity(self) -> dict:
        return {
            "engine": "tesseract",
            "version": self.version,
            "language": self.language,
            "psm": self.psm,
            "timeout": self.timeout,
            "preprocessing": "validated-first-frame-png-v1",
            "traineddata_sha256": {
                language: source_id(self.tessdata_dir / f"{language}.traineddata")
                if self.tessdata_dir and (self.tessdata_dir / f"{language}.traineddata").is_file()
                else None
                for language in (self.language or "").split("+")
                if language
            },
        }

    def _run(self, arguments: list[str]) -> str:
        try:
            result = subprocess.run(
                [self.executable, *arguments],
                capture_output=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Tesseract execution failed: {exc}") from exc
        if result.returncode:
            raise RuntimeError(f"Tesseract failed ({result.returncode}): {result.stderr.strip()}")
        return result.stdout

    def recognize(self, image: Path) -> dict:
        if not self.language:
            raise ValueError("Choose a language for recognition")
        start = time.perf_counter()
        # Pass exactly the validated, EXIF-oriented first frame to the executable.
        # Passing the original TIFF would let Tesseract process unvalidated later frames.
        with load_image(image) as raster, TemporaryDirectory(prefix="indicocr-") as directory:
            prepared = Path(directory) / "input.png"
            options = {"dpi": raster.info["dpi"]} if "dpi" in raster.info else {}
            raster.save(prepared, **options)
            text = self._run([str(prepared), "stdout", "-l", self.language, "--psm", str(self.psm)])
        return {
            "text": text,
            "engine": "tesseract",
            "version": self.version,
            "elapsed_seconds": time.perf_counter() - start,
            "configuration": {"language": self.language, "psm": self.psm},
        }
