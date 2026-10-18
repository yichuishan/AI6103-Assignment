"""Tiny deterministic image fixture; NEVER use as dog-breed experiment results."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite existing fixture")
    rng = np.random.default_rng(142)
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for label in range(3):
            folder = args.output / split / f"synthetic_{label}"
            folder.mkdir(parents=True)
            for i in range(count):
                image = rng.integers(0, 100, size=(110, 130, 3), dtype=np.uint8)
                image[..., label] += 100
                Image.fromarray(image).save(folder / f"{i}.png")
    print(f"Synthetic fixture written to {args.output}; not assignment data")


if __name__ == "__main__":
    main()
