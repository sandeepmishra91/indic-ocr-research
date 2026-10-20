"""Run on an actual GPU host; fail rather than silently benchmarking CPU."""

import json

import torch

from indic_ocr_research.common import device_info
from indic_ocr_research.model import CRNN


def main() -> None:
    info = device_info()
    if not info["gpu_available"]:
        raise SystemExit("No usable CUDA/ROCm GPU; GPU validation was not executed")
    model = CRNN(8).to("cuda")
    images = torch.randn(2, 1, 32, 128, device="cuda")
    lengths = torch.tensor([32, 32])
    logits = model(images, lengths)
    targets = torch.tensor([1, 2, 3, 2, 3, 4], device="cuda")
    loss = torch.nn.CTCLoss()(logits.log_softmax(2), targets, lengths, torch.tensor([3, 3]))
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite GPU loss")
    print(json.dumps({"hardware": info, "loss": loss.item(), "shape": list(logits.shape)}))


if __name__ == "__main__":
    main()
