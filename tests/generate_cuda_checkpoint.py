#!/usr/bin/env python
"""Generate a CUDA-saved checkpoint for integration testing.

Run this script on a machine with a CUDA GPU to produce a checkpoint file
that can be used to verify the CUDA-to-CPU loading fallback in a CI
environment (which typically has no GPU).

Usage::

    # Default output path
    python tests/generate_cuda_checkpoint.py

    # Custom output path
    python tests/generate_cuda_checkpoint.py --output /tmp/cuda_checkpoint.pt

    # Verify an existing checkpoint
    python tests/generate_cuda_checkpoint.py --verify tests/test_data/cuda_checkpoint.pt

Requirements:
    - PyTorch with CUDA support (``torch.cuda.is_available()`` must be True)
    - CEBRA installed (``pip install -e .`` from the repo root)

The generated file is a standard ``torch.save`` checkpoint in the CEBRA
sklearn format.  It contains CUDA tensors, so loading it on a CPU-only
machine *without* the fallback logic will fail with::

    RuntimeError: Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False.
"""

import argparse
import os
import sys

import numpy as np
import torch


def generate(output_path: str) -> None:
    """Train a minimal CEBRA model on CUDA and save the checkpoint."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.  Run this on a GPU machine.",
              file=sys.stderr)
        sys.exit(1)

    import cebra

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Train a tiny model on GPU
    X = np.random.uniform(0, 1, (200, 10)).astype(np.float32)
    model = cebra.CEBRA(
        model_architecture="offset1-model",
        max_iterations=10,
        batch_size=64,
        output_dimension=4,
        device="cuda",
        verbose=False,
    )
    model.fit(X)

    # Sanity-check: model params should live on CUDA
    param_device = next(model.solver_.model.parameters()).device
    assert param_device.type == "cuda", f"Expected cuda, got {param_device}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    print(f"Saved CUDA checkpoint to {output_path}")

    # Verify round-trip on GPU
    loaded = cebra.CEBRA.load(output_path)
    emb = loaded.transform(X)
    assert emb.shape == (200, 4), f"Unexpected shape: {emb.shape}"
    print("Round-trip verification on GPU: OK")


def verify(path: str) -> None:
    """Load a checkpoint on CPU and confirm the fallback works."""
    import cebra

    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from {path} ...")
    model = cebra.CEBRA.load(path)
    print(f"  device_: {model.device_}")
    print(f"  device:  {model.device}")

    X = np.random.uniform(0, 1, (50, model.n_features_)).astype(np.float32)
    emb = model.transform(X)
    print(f"  transform shape: {emb.shape}")
    print("Verification: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        default="tests/test_data/cuda_checkpoint.pt",
        help="Output path for the generated checkpoint (default: tests/test_data/cuda_checkpoint.pt)",
    )
    parser.add_argument(
        "--verify",
        metavar="PATH",
        help="Instead of generating, verify an existing checkpoint can be loaded.",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
    else:
        generate(args.output)


if __name__ == "__main__":
    main()
