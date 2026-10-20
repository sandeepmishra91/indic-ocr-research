# Small Hugging Face imports on slow connections

Use the Hub as optional dataset storage, not a paid inference API. No pretrained
weights, remote dataset scripts, private code or complete large datasets are downloaded.
Base installation and offline OCR do not require Hub packages.

## Install and authenticate

```bash
python -m pip install -e ".[hub]"
indicocr hub inspect --config configs/hf_indicdlp.yaml
indicocr hub import --config configs/hf_indicdlp.yaml --out data/indicdlp-pilot
```

Hub commands read `HF_TOKEN` from the process environment first, then only that key
from `.env` (override with `--env-file`). The token is passed to the official Hub
client. It is never saved in recipes, receipts, reports or CLI output. `.env` remains
ignored. No login command or credential upload is needed for the public pilot.
Errors omit underlying HF exception text because signed URLs may be present there.

## Chosen pilot and license

The recipe pins [onath/indicdlp-eval-set-22-sample](https://huggingface.co/datasets/onath/indicdlp-eval-set-22-sample)
to commit `a290507d248c6fab61fcf64bbd9f7a7f3c3384f7`. The dataset card declares
CC-BY-4.0. Attribution: **IndicDLP public sample by onath on Hugging Face**.
The Hindi shard is 1,260,549 bytes and Malayalam is 413,687 bytes: **1,674,236 bytes**
total, 10 examples each. The recipe caps file downloads at 4,000,000 bytes and
materializes at most 10 rows per file. Metadata/HTTP overhead is additional.

These are cropped printed **blocks**, not full pages or single lines. Publisher labels
are not independently human-verified. The card does not establish an annotation QA
procedure or representativeness. Multiple blocks can come from the same source;
original document grouping is unverified. Use this as a small diagnostic pilot, not
proof of government-scan accuracy. Source rights are publisher-declared, not independently
audited. No source images or labels are committed or included in shared summary artifacts.

The importer compares the pinned revision and declared license with live metadata
before downloads. Unknown/mismatched licenses, floating revisions, missing file sizes,
traversal paths and budgets exceeded are errors. It reads only explicit Parquet shards,
uses embedded image bytes, enforces image pixel limits and validates the manifest.
The receipt retains exact file SHA256, license, attribution, source row/name and revision.
Imports are test-only: never train the CRNN on this pilot evaluation sample.

## Cache and offline reuse

The default cache is `outputs/hf-cache`; data goes to ignored `data/indicdlp-pilot`.
Only two sequential file downloads occur; the official Hub file cache reuses completed
files and partial transfers on retry. Run the same import again after interruption.
Do not delete `.incomplete` cache files when you want to resume. Avoid broad snapshot
downloads and streaming multi-gigabyte shards for this small evaluation.

After one successful online inspection/import:

```bash
indicocr hub inspect --config configs/hf_indicdlp.yaml --offline
indicocr hub import --config configs/hf_indicdlp.yaml --out data/indicdlp-pilot --offline
```

Offline mode requires the matching cached plan and files, uses `local_files_only=True`,
and performs no metadata request. It does not silently fall back to the network.
The output manifest uses relative image paths so its directory can be moved together.
The receipt contains local paths; review receipts before sharing them.

## Actual OCR experiment

Install Tesseract with `eng`, `hin` and `mal` packs. This runner tests PSM 3 (automatic
layout) and PSM 6 (single uniform block) using `hin+eng` or `mal+eng` per language:

```bash
python scripts/run_hf_pilot.py
# With data already cached:
python scripts/run_hf_pilot.py --offline --resume
```

Resume reuses successful predictions with matching dataset/engine identities. Summary
JSON/Markdown/CSV contain measured scores, failures, latency, engine version, traineddata
hashes when available, image hashes, configuration and provenance. They contain no
raw images or transcriptions. Detailed benchmark reports and journals stay local.
Exploring two PSM settings on these examples is exploratory tuning; obtain a separate
held-out human-audited corpus before making a quality claim.

If local installation is inconvenient, run the **Small HF block OCR pilot** workflow
manually from GitHub Actions, or add the `run-hf-pilot` label to a PR. The opt-in job
installs only CPU/system OCR dependencies, downloads the two public shards on the hosted
runner and uploads only summary artifacts for 30 days. It needs no HF secret and never
uploads `.env`. Ordinary CI tests use mock metadata and synthetic local Parquet files.

Upstream references: [file downloads](https://huggingface.co/docs/huggingface_hub/guides/download),
[cache behavior](https://huggingface.co/docs/huggingface_hub/guides/manage-cache),
[offline mode](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
