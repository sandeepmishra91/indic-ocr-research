# Reproducibility

Create a dedicated environment, install the package, run `indicocr doctor`, and save
`python -m pip freeze` with experiment artifacts. Version ranges allow installation
across platforms; they are not a universal lockfile. Archive exact installed versions
for each platform, dataset licenses, source document hashes, manifest and split seed.

Training seeds Python, NumPy and torch, disables cuDNN benchmarking and requests
deterministic cuDNN behavior. It uses zero-worker loaders and a seeded shuffle generator.
Checkpoints save model/optimizer/scaler, epoch history, torch and loader RNG, GPU RNG,
vocabulary, config, manifest SHA and selected image byte fingerprint. Resume restores
at an epoch boundary and checks dataset and architecture/settings compatibility.
Python/NumPy are seeded but have no stochastic training transforms here. Cross-hardware
and cross-version bitwise reproducibility is not guaranteed; ROCm CTC/LSTM availability
and deterministic implementations must be tested on the chosen runtime.

`experiment.json` records history, validation losses/metrics, config, UTC timestamp,
Git SHA when a commit exists, hardware, installed versions, elapsed session time,
resume source and checkpoint path. Reusing an output directory replaces the latest
checkpoint/experiment, retaining loaded history when resuming. Use distinct output
directories for independent experiments. Benchmark reports retain actual predictions;
comparison outputs CSV plus Markdown. Never populate an experiment with invented values.

On the initial Windows environment, pytest's default system temporary directory was
not accessible. Create `outputs` and use a fresh `--basetemp=outputs/pytest-UNIQUE`
when needed. Pytest deletes existing basetemp content, so choose only a disposable
test directory. Module invocation works when the console script folder is not on PATH.
