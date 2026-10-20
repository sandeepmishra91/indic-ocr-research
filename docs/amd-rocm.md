# AMD ROCm research readiness

No MI300X, ROCm GPU or NVIDIA GPU benchmark has been executed for this project.
No kernel drivers are installed by the package or scripts. Keep the Oracle ARM
production server on ordinary CPU inference dependencies.

Use a separate supported Linux AMD GPU research host. First check the current
[AMD compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
and [PyTorch compatibility guidance](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)
for the exact MI300X, host architecture, kernel, OS, ROCm and PyTorch combination.
Use your cloud provider's prepared ROCm image or a compatible official AMD PyTorch
container; pin its immutable digest and record the host driver version. Do not assume
an arbitrary ROCm wheel works with an arbitrary host. Follow
[AMD installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
for that host; this repository intentionally does not automate driver installation.

PyTorch ROCm intentionally uses the `torch.cuda` API; `torch.version.hip` identifies
the build. See [official HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).
Diagnostics distinguish an installed ROCm build from actual available hardware and
record runtime versions, device names, capabilities and memory when available.

After installing a compatible torch build and this package:

```bash
indicocr doctor
python scripts/validate_gpu.py
python examples/make_smoke_data.py --out outputs/smoke-data
```

The GPU validator refuses CPU fallback and executes real forward, float32 CTC and
backward passes. Save its stdout and runtime errors. Copy `configs/smoke.yaml` to
`configs/smoke_gpu.yaml`, change `device: cpu` to `device: cuda` and `amp: false`
to `amp: true`, choose a distinct output directory, then run:

```bash
indicocr train --config configs/smoke_gpu.yaml
```

`cuda` means the shared PyTorch GPU device API for NVIDIA or AMD. Ordinary training
falls back to CPU if unavailable; check recorded device before interpreting speed.
AMP uses torch.autocast and GradScaler on GPU; CTC uses float32. Validate both AMP
and full precision on the actual GPU stack. The code uses no custom CUDA kernels.

After mechanics pass, run licensed human-labeled Hindi/Malayalam lines, retain
held-out documents and repeat controlled CPU/GPU experiments. Measure warmup, batch
size, memory, latency distributions and energy/cost separately before making
performance claims. MI300X throughput, ARM latency and 90% word accuracy remain
unmeasured research goals.
