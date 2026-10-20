# Contributing

Use Python 3.11+, a feature branch and a dedicated environment. Install `.[dev]`,
and a CPU torch build for training changes. Run `python -m ruff check .` and
`python -m pytest -q`. Add assertions for behavior changes. CI must work without
GPU hardware or paid APIs. Include actual command status and hardware in results.

Keep fixes focused and typed. Do not submit private production code, credentials,
model outputs, personal information, unlicensed scans or fabricated benchmark values.
Synthetic fixtures must be labeled. Document dependency/data/weight licenses before
adding them. Separate page and line evaluation. Source contributions use Apache-2.0.
