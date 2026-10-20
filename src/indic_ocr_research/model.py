"""Compact Unicode code-point CRNN. Time dimension is floor(image_width / 4)."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import normalize
from .documents import load_image


class Vocabulary:
    def __init__(self, characters: list[str]):
        if len(set(characters)) != len(characters) or any(len(c) != 1 for c in characters):
            raise ValueError("Vocabulary must contain unique Unicode code points")
        self.characters = characters
        self.indices = {c: i + 1 for i, c in enumerate(characters)}

    def encode(self, text: str) -> list[int]:
        try:
            return [self.indices[c] for c in normalize(text)]
        except KeyError as exc:
            raise ValueError(f"Character absent from training vocabulary: {exc}") from exc

    def decode(self, sequence: list[int]) -> str:
        result, previous = [], 0
        for token in sequence:
            if token and token != previous:
                result.append(self.characters[token - 1])
            previous = token
        return "".join(result)


def line_tensor(path: Path, height: int = 32, max_width: int = 2048) -> torch.Tensor:
    if height < 8 or max_width < 4:
        raise ValueError("height >= 8 and max_width >= 4 required")
    image = load_image(path).convert("L")
    width = max(4, round(image.width * height / image.height))
    if width > max_width:
        raise ValueError(
            f"Line width {width} exceeds {max_width}; crop or configure a larger budget"
        )
    image = image.resize((width, height))
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).unsqueeze(0) / 127.5 - 1


def collate(samples: list[tuple[torch.Tensor, list[int], str]]) -> tuple:
    widths = torch.tensor([image.shape[-1] for image, _, _ in samples], dtype=torch.long)
    images = torch.ones(len(samples), 1, samples[0][0].shape[1], int(widths.max()))
    targets, texts = [], []
    for index, (image, target, text) in enumerate(samples):
        images[index, :, :, : image.shape[-1]] = image
        targets.extend(target)
        texts.append(text)
    target_lengths = torch.tensor([len(target) for _, target, _ in samples], dtype=torch.long)
    lengths = widths // 4
    for (_, target, _), length in zip(samples, lengths.tolist(), strict=True):
        repeats = sum(a == b for a, b in zip(target, target[1:]))
        if len(target) + repeats > length:
            raise ValueError(
                "CTC alignment impossible: target and repeated tokens exceed time steps"
            )
    return images, torch.tensor(targets, dtype=torch.long), lengths, target_lengths, texts


class CRNN(nn.Module):
    def __init__(self, classes: int, hidden: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.sequence = nn.LSTM(64, hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden * 2, classes)

    def forward(self, images: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        features = self.features(images).mean(dim=2).transpose(1, 2)
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        sequence, _ = self.sequence(packed)
        padded, _ = pad_packed_sequence(sequence, batch_first=True, total_length=features.shape[1])
        return self.classifier(padded).transpose(0, 1)


def greedy(logits: torch.Tensor, lengths: torch.Tensor, vocab: Vocabulary) -> list[str]:
    tokens = logits.argmax(dim=2).transpose(0, 1).cpu().tolist()
    return [
        vocab.decode(row[:length]) for row, length in zip(tokens, lengths.tolist(), strict=True)
    ]
