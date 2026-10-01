#!/usr/bin/env python3
"""Convert a trusted rl_games checkpoint into the WorkBench data-only format.

Only the actor MLP and observation statistics are retained. The browser never
deserializes optimizer or training state. Run this script in the legacy Isaac
Gym Python environment.

PyTorch versions used by the archive predate safe ``weights_only`` loading, so
the CLI requires ``--trusted`` and must never be used on downloaded checkpoints.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(source: Path, destination: Path) -> tuple[int, int]:
    checkpoint = torch.load(source, map_location="cpu")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("expected an rl_games checkpoint containing model")

    prefix = "a2c_network.actor_mlp."
    indices = sorted(
        int(key[len(prefix):].split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    if not indices or indices != list(range(0, indices[-1] + 1, 2)):
        raise ValueError("expected an ELU actor MLP with alternating Linear layers")

    mean = state["running_mean_std.running_mean"].float()
    variance = state["running_mean_std.running_var"].float()
    if mean.ndim != 1 or variance.shape != mean.shape or torch.any(variance < 0):
        raise ValueError("invalid rl_games observation normalization")
    variance = variance + 1e-5
    standard_deviation = torch.sqrt(variance)

    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    layer = 0
    for index in indices:
        weight = state[f"{prefix}{index}.weight"].detach().float().cpu()
        bias = state[f"{prefix}{index}.bias"].detach().float().cpu()
        if index == 0 and weight.shape[1] != mean.numel():
            raise ValueError("normalizer and actor input dimensions differ")
        output[f"mlp.{layer}.weight"] = weight.contiguous()
        output[f"mlp.{layer}.bias"] = bias.contiguous()
        layer += 2

    output[f"mlp.{layer}.weight"] = state["a2c_network.mu.weight"].detach().float().cpu().contiguous()
    output[f"mlp.{layer}.bias"] = state["a2c_network.mu.bias"].detach().float().cpu().contiguous()
    if output[f"mlp.{layer}.weight"].shape[1] != output[f"mlp.{layer - 2}.weight"].shape[0]:
        raise ValueError("actor output layer does not follow the MLP")
    output["obs_normalizer._mean"] = mean.unsqueeze(0).contiguous()
    output["obs_normalizer._std"] = standard_deviation.unsqueeze(0).contiguous()
    output["obs_normalizer._var"] = variance.unsqueeze(0).contiguous()
    output["obs_normalizer.count"] = state["running_mean_std.count"].detach().long().cpu()

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": output, "normalization_type": "rl_games"}, destination)
    output_size = output[f"mlp.{layer}.bias"].numel()
    print(f"exported {source.name}: {mean.numel()} -> {output_size} to {destination}")
    return mean.numel(), output_size


def convert_directory(source: Path, destination: Path) -> None:
    for checkpoint in sorted(source.glob("*.pth")):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint.stem)
        output_dir = destination / f"IsaacGym-{name}"
        output = output_dir / f"{name}.pt"
        input_size, output_size = convert(checkpoint, output)
        provenance = {
            "sourceProject": str(source.parent),
            "sourceCheckpoint": f"pth/{checkpoint.name}",
            "sourceSha256": digest(checkpoint),
            "exportedCheckpoint": output.name,
            "exportedSha256": digest(output),
            "network": {"inputSize": input_size, "outputSize": output_size, "activation": "elu"},
            "conversion": "scripts/export-legacy-isaacgym.py",
        }
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted legacy rl_games .pth checkpoint or directory")
    parser.add_argument("destination", type=Path, help="output .pt path or checkpoint catalog directory")
    parser.add_argument("--trusted", action="store_true", help="confirm source checkpoints are trusted pickle files")
    args = parser.parse_args()
    if not args.trusted:
        parser.error("refusing to execute PyTorch pickle without --trusted")
    if args.source.is_dir():
        convert_directory(args.source, args.destination)
    else:
        convert(args.source, args.destination)


if __name__ == "__main__":
    main()
