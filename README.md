# Indian Language OCR Research

**Status:** Research and development in progress

## Overview

This project explores high-accuracy optical character recognition for scanned Indian-language documents, including government notifications, forms, and printed PDFs.

The research focuses on multilingual text recognition, efficient model training, and practical deployment on CPU-based infrastructure.

## Research Objectives

- Evaluate open-source OCR architectures for Indian scripts, initially focusing on Hindi and Malayalam.
- Target over 90% word-level accuracy on scanned printed documents.
- Benchmark recognition using Character Error Rate (CER) and Word Error Rate (WER).
- Investigate model compression and CPU inference optimization.
- Evaluate AMD ROCm and PyTorch for GPU-accelerated OCR training.

## Planned AMD GPU Experiments

I am exploring AMD Instinct MI300X GPU access through the AMD AI Developer Program.

The planned experiments include:

1. Testing PyTorch and ROCm compatibility.
2. Fine-tuning multilingual OCR recognition models.
3. Comparing accuracy and training performance against baseline implementations.
4. Evaluating optimization strategies for inference on ARM CPUs.

## Infrastructure

The project is being developed using Python and CPU-based cloud infrastructure. GPU acceleration is being explored for training and benchmarking.

## Project Status

The research is ongoing. Benchmark results and reproducible experiments may be published as they become available.

The complete application source code is currently maintained privately.
