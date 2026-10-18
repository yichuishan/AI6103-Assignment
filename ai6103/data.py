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


def imagefolder_split_rows(folders):
    """Infer the breed directory from training paths, allowing image wrappers."""
    images = {}
    for split,folder in folders:
        images[split]=[(p,p.relative_to(folder).parts[:-1]) for p in sorted(folder.rglob("*"))
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    training=images["train"]
    if not training or any(not parts for _,parts in training):
        raise ValueError("Unlabelled/empty image directories: use the dataset's label CSV or a path,label,split manifest")
    levels=[]
    for level in range(min(len(parts) for _,parts in training)):
        names={parts[level] for _,parts in training}
        levels.append((level,names))
    # The assignment's class count resolves wrappers such as views/<view>/<breed>.
    candidates=[(level,names) for level,names in levels if len(names)==120]
    if not candidates:
        candidates=[(level,names) for level,names in levels if len(names)>1] or levels[:1]
    level,classes=candidates[0]
    rows=[]
    for split,items in images.items():
        for image,parts in items:
            if level>=len(parts) or parts[level] not in classes:
                raise ValueError(f"Breed directories do not match training layout for {image}; provide a label manifest")
            rows.append(dict(path=str(image.resolve()),label=parts[level],split=split))
    return rows


def discover(root, manifest=None, allow_resplit=False, seed=42):
    root = Path(root).resolve()
    if manifest:
        return read_manifest(manifest, root), "provided manifest"
    val_dirs = [root / n for n in ("val", "valid", "validation") if (root / n).is_dir()]
    if len(val_dirs) > 1:
        raise ValueError("Ambiguous validation directories; provide a manifest")
    if (root / "train").is_dir() and val_dirs and (root / "test").is_dir():
        folders=(("train",root/"train"),("val",val_dirs[0]),("test",root/"test"))
        return imagefolder_split_rows(folders), "provided split directories; breed directory inferred from training paths"
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


def validate_rows(rows, strict=True, duplicate_policy="error", audit=None):
    if duplicate_policy not in {"error","report"}:
        raise ValueError("duplicate_policy must be error or report")
    counts = Counter(r["split"] for r in rows)
    if set(counts) != set(SPLIT_COUNTS):
        raise ValueError(f"Expected train/val/test; got {dict(counts)}")
    if strict and dict(counts) != SPLIT_COUNTS:
        raise ValueError(f"Expected {SPLIT_COUNTS}; got {dict(counts)}")
    classes = sorted({r["label"] for r in rows if r["split"] == "train"})
    if strict and len(classes) != 120:
        examples=[r["path"] for r in rows if r["split"]=="train"][:3]
        raise ValueError(f"Expected 120 dog breeds, got {len(classes)}. "
                         f"Detected labels: {classes[:8]}; training path examples: {examples}. "
                         "Check label CSVs or nested image folders; do not remove the 120-class validation.")
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
        seen_content.setdefault(digest,[]).append(r)
        r["sha256"] = digest
    cross=[]
    overlap={"train_val":0,"train_test":0,"val_test":0}
    within_split_groups=0
    for digest,items in seen_content.items():
        if len(items)<2:
            continue
        labels={r["label"] for r in items}
        if len(labels)>1:
            raise ValueError(f"Identical image bytes have conflicting labels: {sorted(labels)}; "
                             f"paths: {[r['path'] for r in items]}")
        splits={r["split"] for r in items}
        if len(splits)<2:
            within_split_groups+=1
            continue
        cross.append({"sha256":digest,"label":items[0]["label"],
                      "images":[{"path":r["path"],"split":r["split"]} for r in items]})
        for pair in overlap:
            earlier,later=pair.split("_")
            if earlier in splits and later in splits:
                overlap[pair]+=sum(r["split"]==later for r in items)
    quality={"cross_split_duplicate_groups":len(cross),
             "images_in_cross_split_groups":sum(len(g["images"]) for g in cross),
             "within_split_duplicate_groups":within_split_groups,
             "overlap_images":overlap,"groups":cross,
             "policy":"preserve supplied split membership and disclose byte-identical overlap" if duplicate_policy=="report"
                      else "reject cross-split byte-identical images"}
    if audit is not None:
        audit.update(quality)
    if cross and duplicate_policy=="error":
        raise ValueError(f"Identical image bytes across splits: {cross[0]['images'][-1]['path']}")
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
    quality={}
    # The assignment fixes supplied split membership and exact counts. Preserve
    # those splits, disclose existing overlap, and keep rejecting label conflicts.
    # Newly generated splits still require no cross-split byte duplicates.
    new_split=provenance.startswith("new stratified exact-count split")
    classes, counts = validate_rows(rows, strict, duplicate_policy="error" if new_split else "report",audit=quality)
    if quality["cross_split_duplicate_groups"]:
        print("DATA QUALITY: preserving supplied split sizes; byte-identical overlap:",quality["overlap_images"],
              "; details will be saved and disclosed in the report.",flush=True)
    stats = statistics(rows)
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    prepared = {"rows": rows, "classes": classes, "counts": counts, "statistics": stats,
                "fingerprint": fingerprint, "split_provenance": provenance,
                "assignment_data": strict, "seed": seed,"data_quality_audit":quality}
    save_json(output, prepared)
    return prepared
