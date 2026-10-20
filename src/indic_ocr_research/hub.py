"""Small, licensed, commit-pinned Parquet imports; no remote code or model downloads."""

import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import write_json
from .datasets import validate_rows
from .documents import MAX_PIXELS, source_id

OPEN_LICENSES = {"cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "apache-2.0", "mit", "bsd-3-clause"}


def recipe(path: Path) -> dict:
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid Hugging Face recipe YAML") from exc
    required = {
        "repo_id",
        "revision",
        "license",
        "license_url",
        "attribution",
        "task",
        "annotation_method",
        "image_column",
        "text_column",
        "id_column",
        "max_download_bytes",
        "max_samples_per_file",
        "files",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Recipe must contain exactly the documented keys")
    for key in required - {"files", "max_download_bytes", "max_samples_per_file"}:
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise ValueError(f"Recipe {key} must be a nonempty string")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", spec["repo_id"]):
        raise ValueError("repo_id must be owner/name")
    if not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
        raise ValueError("revision must be an immutable 40-character commit SHA")
    if spec["license"] not in OPEN_LICENSES:
        raise ValueError("Dataset license is not in the reviewed open-license allowlist")
    if spec["task"] not in {"line", "block", "page"}:
        raise ValueError("task must be line, block, or page")
    for key in ("max_download_bytes", "max_samples_per_file"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(spec["files"], list) or not spec["files"]:
        raise ValueError("files must be a nonempty explicit list")
    seen = set()
    for item in spec["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "language", "script"}:
            raise ValueError("Each file needs path, language and script")
        if any(not isinstance(v, str) or not v.strip() for v in item.values()):
            raise ValueError("File metadata must be nonempty strings")
        name = PurePosixPath(item["path"])
        if (
            name.is_absolute()
            or ".." in name.parts
            or "\\" in item["path"]
            or ":" in item["path"]
            or name.suffix != ".parquet"
            or item["path"] in seen
        ):
            raise ValueError("Files must be unique relative Parquet paths without traversal")
        seen.add(item["path"])
    return spec


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _token(env_file: Path | None) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token and env_file and env_file.is_file():
        from dotenv import dotenv_values

        token = dotenv_values(env_file).get("HF_TOKEN")
    return token or None


def _api():
    # Keep caches/transfers local and sequential on slow connections.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import HfApi, hf_hub_download

    return HfApi, hf_hub_download


def plan_dataset(
    config: Path, cache: Path, offline: bool = False, env_file: Path | None = None
) -> dict:
    spec = recipe(config)
    plan_path = cache / "plans" / f"{spec_hash(spec)}.json"
    if offline:
        if not plan_path.is_file():
            raise ValueError("No cached plan for this recipe; run hub inspect online once")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("spec") != spec:
            raise ValueError("Cached plan does not match the recipe")
    else:
        HfApi, _ = _api()
        try:
            info = HfApi(endpoint="https://huggingface.co").dataset_info(
                spec["repo_id"],
                revision=spec["revision"],
                files_metadata=True,
                token=_token(env_file) or False,
            )
        except Exception as exc:
            # HF exceptions may contain signed URLs. Do not emit their text or headers.
            raise RuntimeError(
                f"HF metadata request failed ({type(exc).__name__}); "
                "check network/access or use --offline"
            ) from None
        actual_license = (info.card_data or {}).get("license")
        if info.sha != spec["revision"] or actual_license != spec["license"]:
            raise ValueError("Pinned revision or dataset-card license does not match recipe")
        available = {item.rfilename: item for item in info.siblings}
        files = []
        for item in spec["files"]:
            remote = available.get(item["path"])
            if remote is None or remote.size is None:
                raise ValueError("Requested file is missing or has no known size")
            digest = getattr(remote.lfs, "sha256", None) if remote.lfs else None
            files.append(item | {"size": remote.size, "sha256": digest})
        plan = {"spec": spec, "files": files, "total_bytes": sum(f["size"] for f in files)}
    if plan["total_bytes"] > spec["max_download_bytes"]:
        raise ValueError("Requested files exceed the recipe download budget")
    if not offline:
        write_json(plan_path, plan)
    return plan


def import_dataset(
    config: Path, out: Path, cache: Path, offline: bool = False, env_file: Path | None = None
) -> dict:
    import pyarrow.parquet as pq

    plan = plan_dataset(config, cache, offline, env_file)
    spec = plan["spec"]
    receipt_path = out / "import.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("recipe_sha256") != spec_hash(spec):
            raise ValueError("Output directory belongs to a different import recipe")
    _, download = _api()
    token = None if offline else _token(env_file)
    rows, artifacts = [], []
    for item in plan["files"]:
        try:
            cached = Path(
                download(
                    repo_id=spec["repo_id"],
                    filename=item["path"],
                    repo_type="dataset",
                    revision=spec["revision"],
                    cache_dir=str(cache),
                    local_files_only=offline,
                    endpoint="https://huggingface.co",
                    token=token if token else False,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"HF file retrieval failed ({type(exc).__name__}); "
                "retry to reuse partial cache or check offline availability"
            ) from None
        digest = source_id(cached)
        if cached.stat().st_size != item["size"] or item["sha256"] not in (None, digest):
            raise ValueError("Cached file size/hash differs from pinned metadata")
        artifacts.append({"file": item["path"], "bytes": cached.stat().st_size, "sha256": digest})
        parquet = pq.ParquetFile(cached)
        columns = [spec["image_column"], spec["text_column"], spec["id_column"]]
        if not set(columns) <= set(parquet.schema_arrow.names):
            raise ValueError("Parquet columns do not match the recipe")
        for index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=columns)):
            if index >= spec["max_samples_per_file"]:
                break
            original = batch.to_pylist()[0]
            text, original_id = original[spec["text_column"]], original[spec["id_column"]]
            value = original[spec["image_column"]]
            if not isinstance(text, str) or not isinstance(original_id, str) or not original_id:
                raise ValueError("Dataset has invalid labels or source identifiers")
            if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
                raise ValueError("Only embedded image bytes are supported; no remote image URLs")
            sid = hashlib.sha256(
                f"{spec['repo_id']}@{spec['revision']}:{item['path']}:{index}".encode()
            ).hexdigest()[:24]
            image_path = out / "images" / f"{sid}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(io.BytesIO(value["bytes"])) as image:
                    if image.width * image.height > MAX_PIXELS:
                        raise ValueError("Embedded image exceeds pixel budget")
                    ImageOps.exif_transpose(image).convert("RGB").save(image_path)
            except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValueError("Unreadable embedded image") from exc
            rows.append(
                {
                    "sample_id": sid,
                    "image_path": str(image_path.resolve()),
                    "transcription": text,
                    "language": item["language"],
                    "script": item["script"],
                    "document_id": f"{spec['repo_id']}:{original_id}",
                    "split": "test",
                    "source_url": spec["license_url"],
                    "license": spec["license"],
                    "annotation_method": spec["annotation_method"],
                    "task": spec["task"],
                    "hf_revision": spec["revision"],
                    "source_image_name": original_id,
                    "source_file": item["path"],
                    "source_row": index,
                    "ground_truth_verified": False,
                    "attribution": spec["attribution"],
                    "document_grouping": "source-image-name-unverified",
                }
            )
    if not rows:
        raise ValueError("The selected Parquet files contain no samples")
    validation = validate_rows(rows)
    if not validation["valid"]:
        raise ValueError("Imported samples failed validation: " + "; ".join(validation["errors"]))
    manifest = out / "manifest.jsonl"
    portable = [row | {"image_path": f"images/{row['sample_id']}.png"} for row in rows]
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in portable), encoding="utf-8"
    )
    temporary.replace(manifest)
    receipt = {
        "recipe_sha256": spec_hash(spec),
        "spec": spec,
        "artifacts": artifacts,
        "samples": len(rows),
        "manifest": str(manifest.resolve()),
        "attribution": spec["attribution"],
        "ground_truth_verified": False,
        "notice": "Publisher labels, not independently verified human annotations. "
        "Small convenience sample, not a representative quality benchmark.",
    }
    write_json(receipt_path, receipt)
    write_json(out / "validation.json", validation)
    return receipt
