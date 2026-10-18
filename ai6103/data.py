"""Data discovery, split validation, training-only statistics and transforms."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

SPLIT_COUNTS = {"train": 7155, "val": 1533, "test": 1534}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def read_manifest(path, root):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires path,label,split columns")
        rows = [dict(path=str((root / r["path"].replace("\\", "/")).resolve()),
                     label=r["label"], split=r["split"]) for r in reader]
    return rows


def discover(root, manifest=None, allow_resplit=False, seed=42):
    root = Path(root).resolve()
    if manifest:
        return read_manifest(manifest, root), "provided manifest"
    val_dirs = [root / n for n in ("val", "valid", "validation") if (root / n).is_dir()]
    if len(val_dirs) > 1:
        raise ValueError("Ambiguous validation directories; provide a manifest")
    if (root / "train").is_dir() and val_dirs and (root / "test").is_dir():
        rows = []
        for split, folder in (("train", root / "train"), ("val", val_dirs[0]), ("test", root / "test")):
            for image in sorted(folder.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    if image.parent == folder:
                        raise ValueError("Unlabelled images: use path,label,split manifest")
                    rows.append(dict(path=str(image.resolve()), label=image.parent.name, split=split))
        return rows, "provided split directories"
    if not allow_resplit:
        raise ValueError("No labelled split directories found. Provide --manifest, or use "
                         "--allow-resplit ONLY for an unsplit labels.csv dataset.")
    labels = root / "labels.csv"
    if not labels.is_file():
        raise ValueError("Expected labels.csv and train/<id>.jpg")
    with labels.open(newline="", encoding="utf-8-sig") as f:
        raw = sorted(csv.DictReader(f), key=lambda r: r["id"])
    if len(raw) != sum(SPLIT_COUNTS.values()):
        raise ValueError("Expected 10,222 labelled images, not Kaggle's unlabelled test set")
    # Per-class largest-remainder allocation gives exact global split counts.
    classes = sorted({r["breed"] for r in raw})
    grouped = [[r for r in raw if r["breed"] == c] for c in classes]
    remaining = np.array([len(g) for g in grouped])
    rng = np.random.default_rng(seed)
    for g in grouped:
        rng.shuffle(g)
    rows = []
    for split, desired in list(SPLIT_COUNTS.items())[:-1]:
        ideal = remaining * desired / remaining.sum()
        allocated = np.floor(ideal).astype(int)
        for i in np.argsort(-(ideal - allocated), kind="stable")[:desired - allocated.sum()]:
            allocated[i] += 1
        for i, g in enumerate(grouped):
            for r in g[:allocated[i]]:
                rows.append(dict(path=str((root / "train" / (r["id"] + ".jpg")).resolve()),
                                 label=r["breed"], split=split))
            del g[:allocated[i]]
        remaining -= allocated
    for g in grouped:
        for r in g:
            rows.append(dict(path=str((root / "train" / (r["id"] + ".jpg")).resolve()),
                             label=r["breed"], split="test"))
    return rows, f"new stratified exact-count split, seed={seed}; not an official supplied split"


def validate_rows(rows, strict=True):
    counts = Counter(r["split"] for r in rows)
    if set(counts) != set(SPLIT_COUNTS):
        raise ValueError(f"Expected train/val/test; got {dict(counts)}")
    if strict and dict(counts) != SPLIT_COUNTS:
        raise ValueError(f"Expected {SPLIT_COUNTS}; got {dict(counts)}")
    classes = sorted({r["label"] for r in rows if r["split"] == "train"})
    if strict and len(classes) != 120:
        raise ValueError(f"Expected 120 dog breeds, got {len(classes)}")
    seen_paths, seen_content = set(), {}
    for r in rows:
        p = Path(r["path"]).resolve()
        if r["label"] not in classes:
            raise ValueError(f"Label not in training classes: {r['label']}")
        if p in seen_paths:
            raise ValueError(f"Repeated image path: {p}")
        seen_paths.add(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        # Byte-level leakage detection reads bytes, never test labels/predictions for selection.
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen_content and seen_content[digest] != r["split"]:
            raise ValueError(f"Identical image bytes across splits: {p}")
        seen_content[digest] = r["split"]
        r["sha256"] = digest
    return classes, dict(counts)


def statistics(rows):
    resize = T.Resize((100, 100), antialias=True)
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for r in rows:
        if r["split"] != "train":
            continue
        with Image.open(r["path"]) as im:
            pixels = np.asarray(resize(im.convert("RGB")), dtype=np.float64).reshape(-1, 3) / 255.
        sums += pixels.sum(0)
        squares += np.square(pixels).sum(0)
        count += len(pixels)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    if np.any(std < 1e-8):
        raise ValueError("A training color channel has zero variance")
    return {"mean": mean.tolist(), "std": std.tolist(), "pixels": count,
            "definition": "RGB in [0,1], resize 100x100 bilinear; population std, training only"}


def transform(stats, augment=False):
    ops = [T.Resize((100, 100), antialias=True)]
    if augment:
        ops += [T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(100, padding=12, padding_mode="constant", fill=0)]
    ops += [T.ToTensor(), T.Normalize(stats["mean"], stats["std"])]
    return T.Compose(ops)


class DogDataset(Dataset):
    def __init__(self, prepared, split, augment=False):
        self.rows = [r for r in prepared["rows"] if r["split"] == split]
        self.labels = {c: i for i, c in enumerate(prepared["classes"])}
        self.transform = transform(prepared["statistics"], augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r["path"]) as im:
            image = self.transform(im.convert("RGB"))
        return image, self.labels[r["label"]]


def prepare(root, output, manifest=None, allow_resplit=False, seed=42, strict=True):
    if Path(output).exists():
        raise FileExistsError(f"Prepared data already exists: {output}; preserve it or choose a new output")
    rows, provenance = discover(root, manifest, allow_resplit, seed)
    classes, counts = validate_rows(rows, strict)
    stats = statistics(rows)
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    prepared = {"rows": rows, "classes": classes, "counts": counts, "statistics": stats,
                "fingerprint": fingerprint, "split_provenance": provenance,
                "assignment_data": strict, "seed": seed}
    save_json(output, prepared)
    return prepared
