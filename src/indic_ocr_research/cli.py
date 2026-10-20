import json
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from .common import doctor as diagnostics
from .common import write_json
from .datasets import read_manifest, split_documents, validate_rows
from .documents import inspect_pdf, render_pdf
from .engines import Tesseract
from .evaluation import benchmark
from .reporting import compare

app = typer.Typer(no_args_is_help=True, help="Indic OCR Research Lab: offline measured research")
pdf = typer.Typer(no_args_is_help=True)
baseline = typer.Typer(no_args_is_help=True)
manifest_app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
for name, sub in [
    ("pdf", pdf),
    ("baseline", baseline),
    ("manifest", manifest_app),
    ("benchmark", benchmark_app),
    ("report", report_app),
]:
    app.add_typer(sub, name=name)

File = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


def guarded(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapper


def emit(value: dict | list) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command()
@guarded
def doctor() -> None:
    """Report platform, dependency, Tesseract and GPU capabilities."""
    emit(diagnostics())


@pdf.command("inspect")
@guarded
def pdf_inspect(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], password: str | None = None
) -> None:
    emit(inspect_pdf(path, password))


@pdf.command("render")
@guarded
def pdf_render(
    input: File,
    output_dir: Annotated[Path, typer.Option()],
    dpi: Annotated[int, typer.Option(min=36, max=600)] = 200,
    page: Annotated[list[int] | None, typer.Option(min=1)] = None,
    password: str | None = None,
) -> None:
    """Render all pages or repeated --page N selections (1 based)."""
    emit(render_pdf(input, output_dir, dpi, page, password))


@baseline.command("run")
@guarded
def baseline_run(
    input: File, lang: str = "eng", psm: Annotated[int, typer.Option(min=0, max=13)] = 3
) -> None:
    emit(Tesseract(lang, psm).recognize(input))


@manifest_app.command("validate")
@guarded
def manifest_validate(path: File, out: Path | None = None, nfc: bool = True) -> None:
    result = validate_rows(read_manifest(path, nfc))
    if out:
        write_json(out, result)
    emit(result)
    if not result["valid"]:
        raise typer.Exit(1)


@manifest_app.command("split")
@guarded
def manifest_split(path: File, out: Annotated[Path, typer.Option()], seed: int = 42) -> None:
    rows = split_documents(read_manifest(path), seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    emit({"samples": len(rows), "output": str(out), "seed": seed})


@benchmark_app.command("run")
@guarded
def benchmark_run(
    manifest: File,
    out: Annotated[Path, typer.Option()],
    engine: str = "tesseract",
    checkpoint: Path | None = None,
    lang: str = "eng",
    psm: Annotated[int, typer.Option(min=0, max=13)] = 3,
    split: str = "test",
    task: str = "page",
    nfc: bool = True,
    whitespace: bool = True,
    device: str = "auto",
) -> None:
    if engine == "tesseract":
        adapter = Tesseract(lang, psm)
    elif engine == "crnn":
        if checkpoint is None or task != "line":
            raise ValueError("CRNN benchmarking requires --checkpoint and --task line")
        from .training import CRNNEngine

        adapter = CRNNEngine(checkpoint, device)
    else:
        raise ValueError("engine must be tesseract or crnn")
    result = benchmark(manifest, adapter, out, split, task, nfc, whitespace)
    emit({key: result[key] for key in ["evaluated_samples", "failed_samples", "raw", "normalized"]})
    if result["failed_samples"]:
        raise typer.Exit(1)


@app.command("train")
@guarded
def train_command(config: File, resume: Path | None = None) -> None:
    from .training import train

    emit(train(config, resume))


@app.command()
@guarded
def recognize(checkpoint: File, image: File, device: str = "auto") -> None:
    from .training import CRNNEngine

    emit(CRNNEngine(checkpoint, device).recognize(image))


@report_app.command("compare")
@guarded
def report_compare(
    inputs: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option()],
) -> None:
    emit({"runs": compare(inputs, out), "output": str(out)})


if __name__ == "__main__":
    app()
