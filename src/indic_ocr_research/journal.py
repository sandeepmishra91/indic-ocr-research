"""Durable single-writer prediction journal for interrupted local benchmarks."""

import hashlib
import json
import math
import os
from pathlib import Path


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class PredictionJournal:
    def __init__(self, path: Path, identity: dict, resume: bool):
        self.path, self.identity = path, identity
        self.events: list[dict] = []
        self.latest: dict[str, dict] = {}
        self.stream = None
        if resume:
            if not path.is_file():
                raise ValueError("No prediction journal exists to resume")
            content = path.read_bytes()
            boundary = content.rfind(b"\n") + 1
            complete = content[:boundary]
            try:
                records = [json.loads(line) for line in complete.splitlines()]
            except (ValueError, UnicodeDecodeError):
                raise ValueError(
                    "Corrupt prediction journal; only an incomplete final line is recoverable"
                ) from None
            if not records or records[0] != {"kind": "run", "identity": identity}:
                raise ValueError("Resume identity mismatch: data, engine or configuration changed")
            for event in records[1:]:
                if not isinstance(event, dict):
                    raise ValueError("Invalid prediction journal record")
                sid = event.get("sample_id")
                if (
                    event.get("kind") != "prediction"
                    or not isinstance(sid, str)
                    or event.get("status") not in ("success", "failure")
                    or type(event.get("attempt_seconds")) not in (int, float)
                    or not math.isfinite(event["attempt_seconds"])
                    or event["attempt_seconds"] < 0
                ):
                    raise ValueError("Invalid prediction journal record")
                if event["status"] == "success":
                    prediction = event.get("prediction")
                    if not isinstance(prediction, dict) or not isinstance(
                        prediction.get("text"), str
                    ):
                        raise ValueError("Invalid cached prediction")
                    latency = prediction.get("elapsed_seconds")
                    if (
                        type(latency) not in (int, float)
                        or not math.isfinite(latency)
                        or latency < 0
                    ):
                        raise ValueError("Invalid cached prediction latency")
                elif not isinstance(event.get("error"), str):
                    raise ValueError("Invalid cached failure")
                if sid in self.latest and self.latest[sid]["status"] == "success":
                    raise ValueError("Duplicate completed sample in prediction journal")
                self.events.append(event)
                self.latest[sid] = event
            # Truncate only the torn final record, after validating the header and prior records.
            if boundary != len(content):
                with path.open("r+b") as stream:
                    stream.truncate(boundary)
        else:
            if path.exists() or (path.parent / "benchmark.json").exists():
                raise ValueError(
                    "Output already contains a benchmark; use --resume or a new directory"
                )

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a" if self.path.exists() else "x", encoding="utf-8")
        if not self.path.stat().st_size:
            self._write({"kind": "run", "identity": self.identity})
        return self

    def _write(self, record: dict) -> None:
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def append(self, event: dict) -> None:
        self._write(event)
        self.events.append(event)
        self.latest[event["sample_id"]] = event

    def __exit__(self, *exc):
        self.stream.close()
