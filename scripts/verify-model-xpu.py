#!/usr/bin/env python3
"""Opt-in real-checkpoint correctness checks and separate XPU measurements.

Download the checkpoint first. Run with PYTHONPATH=src, ZIMT_DEVICE=xpu,
HF_HOME pointing at that cache, HF_HUB_OFFLINE=1, and the Intel driver directory
on LD_LIBRARY_PATH. Uses the application's public load and generation APIs;
never substitutes a dummy when the GPU or weights are absent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from huggingface_hub import try_to_load_from_cache
from PIL import Image, ImageStat
import torch

from zimt.device import DEVICE
from zimt.generate import GenConfig, generate, load_spec
from zimt.memory import MemStrategy
from zimt.models.registry import MODELS

GIB = 1024 ** 3
SEED = 42
PROMPT = 'A red ceramic teapot on a wooden table, soft window light, a small card reading "ZIMT".'


@dataclass(frozen=True)
class Measurement:
    path: str
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    mode: str
    pixel_sha256: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if DEVICE != "xpu" or not torch.xpu.is_available():
        raise RuntimeError("This verification requires ZIMT_DEVICE=xpu and an accessible Intel GPU")
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    cached_config = try_to_load_from_cache(spec.repo_id, "model_index.json")
    if not isinstance(cached_config, str):
        raise RuntimeError(f"Download {spec.repo_id} before running hardware verification")
    config = GenConfig.from_spec(spec)
    if args.steps is not None:
        config.steps = args.steps
    if args.size is not None:
        config.width = config.height = args.size
    report = {
        "model": spec.name,
        "checkpoint_revision": Path(cached_config).parent.name,
        "device": torch.xpu.get_device_name(),
        "total_memory_gib": torch.xpu.get_device_properties(0).total_memory / GIB,
        "versions": {name: version(name) for name in (
            "torch", "torchvision", "diffusers", "transformers", "accelerate", "peft",
            "huggingface-hub", "safetensors",
        )},
        "memory_strategy": "cpuoffload",
        "steps": config.steps,
        "cfg": config.cfg,
        "width": config.width,
        "height": config.height,
        "seed": SEED,
        "prompt": PROMPT,
    }
    print(json.dumps(report, indent=2), flush=True)
    measurements: list[Measurement] = []
    backend = None
    try:
        started = time.perf_counter()
        backend = load_spec(spec, MemStrategy(mode="cpuoffload"))
        torch.xpu.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        for _run in range(args.runs):
            torch.xpu.reset_peak_memory_stats()
            started = time.perf_counter()
            path = generate(backend, config, PROMPT, SEED, raw=True,
                            out_dir=str(args.output_dir))
            torch.xpu.synchronize()
            seconds = time.perf_counter() - started
            with Image.open(path) as image:
                if image.size != (config.width, config.height):
                    raise AssertionError(f"Unexpected output size: {image.size}")
                if image.info["model"] != spec.name or image.info["seed"] != str(SEED):
                    raise AssertionError("PNG provenance does not match generation inputs")
                if image.info["device"] != "xpu":
                    raise AssertionError("PNG does not identify XPU execution")
                if spec.family == "qwenimage21" and image.mode != "RGBA":
                    raise AssertionError("Qwen's alpha channel was lost")
                if max(ImageStat.Stat(image.convert("RGB")).stddev) < 1:
                    raise AssertionError("Output is effectively blank")
                measurement = Measurement(
                    path=path, seconds=seconds,
                    peak_allocated_gib=torch.xpu.max_memory_allocated() / GIB,
                    peak_reserved_gib=torch.xpu.max_memory_reserved() / GIB,
                    mode=image.mode,
                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                )
            measurements.append(measurement)
            print(json.dumps(asdict(measurement)), flush=True)
        report["status"] = "passed"
        report["identical_pixels_across_runs"] = len({m.pixel_sha256 for m in measurements}) == 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["measurements"] = [asdict(m) for m in measurements]
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if backend is not None:
            backend.unload()


if __name__ == "__main__":
    main()
