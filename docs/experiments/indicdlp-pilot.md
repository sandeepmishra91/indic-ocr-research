# Small block OCR pilot: label review required

Executed on 20 October 2026 with Tesseract 5.3.4 on a GitHub-hosted Linux x86_64 CPU runner.
Four logical CPUs were visible; OMP_THREAD_LIMIT was 2. No GPU or paid OCR API was used.

**The Malayalam aggregate is not a reliable OCR-quality estimate.** A known malformed
reference contains image-analysis prose rather than only the visible text. All original
samples and scores below are preserved; no labels were edited or cases removed.

[Executed workflow](https://github.com/sandeepmishra91/indic-ocr-research/actions/runs/39380976015) ·
[Machine-readable measurement and provenance](indicdlp-pilot-2026-10-20.json)

## Data and protocol

Source: [IndicDLP public sample by onath](https://huggingface.co/datasets/onath/indicdlp-eval-set-22-sample).
Publisher-declared license: CC-BY-4.0; attribution: IndicDLP public sample by onath on Hugging Face.
Pinned revision: `a290507d248c6fab61fcf64bbd9f7a7f3c3384f7`. Two shards total 1,674,236 bytes.
There are ten Hindi and ten Malayalam cropped blocks. Labels and original document grouping
have not been independently human-verified; these are not representative government scans.
No dataset images or source labels are included in this repository.

Each of eight configurations ran real OCR on the same ten blocks for its language: 80
completed attempts and zero execution failures. Empty predictions still count as errors.
CER uses code points; WER uses whitespace words. NFC and whitespace collapse are enabled.
Latency includes image loading/preparation and a new Tesseract subprocess per sample.
No warmup, repeated-trial timing, ARM, GPU or production throughput claim is made.

## Recorded scores

Values are diagnostics against the supplied labels, not independently validated accuracy.

| Language | Packs | PSM | CER | WER | Exact blocks | Empty predictions | Mean seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| hin | hin | 3 | 4.26% | 13.58% | 1/10 | 2 | 0.370 |
| hin | hin | 6 | 4.23% | 13.58% | 1/10 | 0 | 0.353 |
| hin | hin+eng | 3 | 4.94% | 13.94% | 1/10 | 2 | 0.648 |
| hin | hin+eng | 6 | 4.87% | 13.82% | 2/10 | 0 | 0.638 |
| mal | mal | 3 | 69.50% | 91.03% | 3/10 | 0 | 0.082 |
| mal | mal | 6 | 69.65% | 91.38% | 3/10 | 0 | 0.081 |
| mal | mal+eng | 3 | 70.84% | 91.72% | 3/10 | 0 | 0.192 |
| mal | mal+eng | 6 | 70.99% | 92.07% | 3/10 | 0 | 0.190 |

## Findings and next action

Hindi-only packs produced slightly lower CER/WER and shorter observed latency than
Hindi plus English on these ten examples. PSM 6 avoided the two empty outputs seen
with PSM 3. Exact-block outcomes differ, so this does not establish a universal winner.

Removing English did not explain the Malayalam aggregate. Post-run inspection found
sample `27ca9e6492063d4d1e33dd9b`: its small two-line image has a 1,290-character
publisher label containing English analysis and transcription commentary. After the
configured normalization, that single reference contributes 1,285 of 2,020 Malayalam
reference code points and 1,242 of 1,404 character edits in the mal-only PSM 3 run.
The label defect dominates the aggregate. These numbers must not be attributed solely
to recognition quality or used to select a production model.

The new label-audit command flags this sample and exports an unchanged review queue.
It does not silently trim commentary, relabel examples or publish improved scores.
Qualified human reviewers should verify all references and source grouping, then freeze
a separate held-out corpus before comparing engines or tuning segmentation policies.

Neural models were not trained on this evaluation set. No 90% word-accuracy result is claimed.
