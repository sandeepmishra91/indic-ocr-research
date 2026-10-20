import shutil
import subprocess
import time
from pathlib import Path

from .documents import load_image


class Tesseract:
    def __init__(self, language: str | None = "eng", psm: int = 3, timeout: int = 120):
        self.executable = shutil.which("tesseract")
        if not self.executable:
            raise RuntimeError("Tesseract executable missing; install it and add it to PATH")
        if psm not in range(14) or timeout <= 0:
            raise ValueError("PSM must be 0..13 and timeout positive")
        self.language, self.psm, self.timeout = language, psm, timeout
        self.version = self._run(["--version"]).splitlines()[0]
        self.languages = self._run(["--list-langs"]).splitlines()[1:]
        missing = set(language.split("+")) - set(self.languages) if language else set()
        if missing:
            raise RuntimeError(f"Missing Tesseract language packs: {sorted(missing)}")

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
        load_image(image).close()
        start = time.perf_counter()
        text = self._run(
            [str(image.resolve()), "stdout", "-l", self.language, "--psm", str(self.psm)]
        )
        return {
            "text": text,
            "engine": "tesseract",
            "version": self.version,
            "elapsed_seconds": time.perf_counter() - start,
            "configuration": {"language": self.language, "psm": self.psm},
        }
