import importlib.metadata
import json
import platform
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize(text: str, nfc: bool = True, whitespace: bool = False) -> str:
    text = unicodedata.normalize("NFC", text) if nfc else text
    return " ".join(text.split()) if whitespace else text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def provenance() -> dict:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        sha = None
    versions = {}
    for name in [
        "indic-ocr-research",
        "Pillow",
        "numpy",
        "pypdf",
        "pypdfium2",
        "torch",
        "typer",
        "PyYAML",
        "huggingface_hub",
        "pyarrow",
        "python-dotenv",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
    }


def device_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"backend": "cpu", "torch_installed": False, "gpu_available": False}
    gpu = torch.cuda.is_available()
    info = {
        "backend": ("rocm" if torch.version.hip else "cuda") if gpu else "cpu",
        "torch_installed": True,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "cuda": torch.version.cuda,
        "gpu_available": gpu,
        "devices": [],
    }
    if gpu:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "name": props.name,
                    "memory_bytes": props.total_memory,
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
            )
    return info


def doctor() -> dict:
    import shutil

    from .engines import Tesseract

    result = provenance() | {"hardware": device_info(), "tesseract": shutil.which("tesseract")}
    if result["tesseract"]:
        engine = Tesseract(language=None)
        result.update(tesseract_version=engine.version, languages=engine.languages)
    return result
