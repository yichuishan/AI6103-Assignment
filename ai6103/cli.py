"""Command-line interface; test-set evaluation is an explicit final command."""
import argparse
from pathlib import Path
import sys

import torch

from .data import load_json, prepare
from .engine import run_pipeline, test_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Validate data and compute training-only normalization")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/prepared.json"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--allow-resplit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Allow tiny synthetic fixture; never assignment results")
    for cmd in ("run", "test"):
        p = commands.add_parser(cmd)
        p.add_argument("--prepared", type=Path, default=Path("data/prepared.json"))
        p.add_argument("--runs", type=Path, default=Path("runs/full"))
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--threads", type=int, default=4)
        if cmd == "run":
            p.add_argument("--stage", choices=["all", "lr", "schedule", "decay", "mixup"], default="all")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p = commands.add_parser("plot")
    p.add_argument("--runs", type=Path, default=Path("runs/full"))
    p.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    p = commands.add_parser("theory")
    p.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    p = commands.add_parser("report", help="Generate measured result tables and draft/final report sections")
    p.add_argument("--prepared", type=Path, default=Path("data/prepared.json"))
    p.add_argument("--runs", type=Path, default=Path("runs/full"))
    p.add_argument("--output", type=Path, default=Path("report/generated"))
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepared = prepare(args.data_root, args.output, args.manifest,
                               args.allow_resplit, args.seed, strict=not args.smoke)
            print(prepared["counts"], prepared["statistics"])
        elif args.command in ("run", "test"):
            torch.set_num_threads(args.threads)
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable; install CUDA PyTorch and attach a GPU")
            prepared = load_json(args.prepared)
            if args.command == "run":
                if args.precision != "fp32" and not args.device.startswith("cuda"):
                    raise ValueError("Mixed precision in this workflow requires a CUDA device")
                run_pipeline(prepared, args.runs, args.stage, args.device, args.workers, args.seed, args.smoke, args.precision)
            else:
                print(test_once(prepared, args.runs, args.device, args.workers))
        elif args.command == "plot":
            from .visualize import curves
            curves(args.runs, args.output)
        elif args.command == "theory":
            from .visualize import theoretical_figures
            theoretical_figures(args.output)
        elif args.command == "report":
            from .reporting import generate
            generate(args.prepared, args.runs, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as e:
        parser.exit(2, f"Error: {e}\n")


if __name__ == "__main__":
    main()
