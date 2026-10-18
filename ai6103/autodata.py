"""Safe ZIP preparation and conservative detection of common dataset layouts."""
from collections import Counter, deque
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
import zlib


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_zip(source, work):
    """Extract into a content-addressed directory, never touch the source archive."""
    source, work = Path(source).resolve(), Path(work).resolve()
    digest = file_sha256(source)
    target = work / "extracted" / digest[:16]
    marker = target / ".ai6103-extraction.json"
    if marker.is_file() and json.loads(marker.read_text()).get("sha256") == digest:
        return target, digest
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = []
        seen = set()
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or (name.parts and ":" in name.parts[0]):
                raise ValueError(f"Unsafe ZIP path: {entry.filename}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"ZIP symlinks are not supported: {entry.filename}")
            if "__MACOSX" in name.parts or ".git" in name.parts or name.name == ".DS_Store":
                continue
            destination = target.joinpath(*name.parts)
            if not destination.resolve().is_relative_to(target):
                raise ValueError(f"ZIP entry escapes extraction directory: {entry.filename}")
            if entry.is_dir():
                continue
            if str(destination) in seen:
                raise ValueError(f"Duplicate ZIP destination: {entry.filename}")
            seen.add(str(destination))
            members.append((entry, destination))
        required = sum(entry.file_size for entry, destination in members if not destination.exists())
        if required > shutil.disk_usage(target).free:
            raise ValueError(f"Insufficient disk for extraction ({required/1024**3:.2f} GiB required)")
        for entry, destination in members:
            if destination.exists():
                if not destination.is_file():
                    raise ValueError(f"Existing extraction path is not a file: {destination}")
                crc = 0
                with destination.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024*1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if destination.stat().st_size != entry.file_size or crc != entry.CRC:
                    raise ValueError(f"Existing extracted file differs; preserve it and use a new output directory: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".ai6103-partial")
            try:
                with archive.open(entry) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                partial.replace(destination)
            finally:
                if partial.exists():
                    partial.unlink()
    marker.write_text(json.dumps({"sha256": digest, "files": len(members)}, indent=2)+"\n")
    return target, digest


def csv_fields(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return csv.DictReader(f).fieldnames or []


def split_csv_manifest(root, output):
    choices = {"train": ["train.csv", "training.csv"],
               "val": ["val.csv", "valid.csv", "validation.csv"],
               "test": ["test.csv", "testing.csv"]}
    selected = {}
    for split, options in choices.items():
        found = [root/n for n in options if (root/n).is_file()]
        if len(found) != 1:
            return None
        selected[split] = found[0]
    rows = []
    for split, path in selected.items():
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            path_keys = [n for n in ("path", "image_path", "file", "filename", "image", "id") if n in fields]
            label_keys = [n for n in ("label", "breed", "class", "target") if n in fields]
            if len(path_keys) != 1 or len(label_keys) != 1:
                return None
            for r in reader:
                relative = Path(r[path_keys[0]].replace("\\", "/"))
                prefixes = [root, root/path.stem, root/split, root/"images", root/f"{split}_images"]
                paths = set()
                for prefix in prefixes:
                    candidate = prefix/relative
                    variants = [candidate] if candidate.suffix else [candidate.with_suffix(e) for e in (".jpg", ".jpeg", ".png")]
                    paths.update(p.resolve() for p in variants if p.is_file())
                if len(paths) != 1:
                    raise ValueError(f"Cannot unambiguously locate image {relative} from {path}")
                rows.append({"path":str(paths.pop()),"label":r[label_keys[0]],"split":split})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=["path","label","split"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def find_layout(base, work, explicit_manifest=None):
    base, work = Path(base).resolve(), Path(work).resolve()
    if explicit_manifest:
        return base, Path(explicit_manifest).resolve(), False
    queue = deque([(base,0)])
    raw = []
    while queue:
        root, depth = queue.popleft()
        subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        directories = {p.name for p in subdirs}
        for csv_path in sorted(root.glob("*.csv")):
            if {"path","label","split"} <= set(csv_fields(csv_path)):
                return root, csv_path, False
        manifest = split_csv_manifest(root, work/"detected_manifest.csv")
        if manifest:
            return root, manifest, False
        if {"train","test"} <= directories and directories & {"val","valid","validation"}:
            return root, None, False
        if (root/"labels.csv").is_file() and (root/"train").is_dir():
            if {"id","breed"} <= set(csv_fields(root/"labels.csv")):
                # If split metadata exists, never silently replace its split.
                other_csv = [p for p in root.glob("*.csv") if p.name != "labels.csv" and
                             any(s in p.stem.lower() for s in ("train","val","test","split"))]
                if other_csv:
                    raise ValueError("Split metadata is present but not recognized; supply --manifest path,label,split CSV")
                raw.append(root)
        if depth < 3:
            queue.extend((p,depth+1) for p in subdirs)
    if len(raw) == 1:
        print("Unsplit labels.csv dataset detected: creating and recording a new seeded exact-count stratified split.",flush=True)
        return raw[0], None, True
    raise ValueError("Cannot identify dataset layout. Supported: train/val/test breed folders; "
                     "path,label,split CSV; train/val/test CSVs; or labels.csv + train/<id>.jpg. "
                     "Use --manifest for other labelled image layouts.")


def locate_dataset(argument=None):
    if argument:
        path = Path(argument).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        return path
    preferred = Path("/root/autodl-tmp/data.zip")
    if preferred.is_file():
        return preferred
    candidates = []
    for root in (Path("/root/autodl-tmp"), Path("/root/autodl-fs")):
        if not root.is_dir():
            continue
        for directory, subdirs, files in os.walk(root):
            depth = len(Path(directory).relative_to(root).parts)
            subdirs[:] = [n for n in subdirs if not n.startswith(".") and "venv" not in n.lower()
                          and n not in {"results","runs","extracted","node_modules"}]
            if depth >= 3:
                subdirs[:] = []
            candidates.extend(Path(directory)/n for n in files if n.lower().endswith(".zip"))
    if len(candidates) != 1:
        raise ValueError("Provide the dataset ZIP/directory as the script argument. "
                         f"Found {len(candidates)} candidate archives; default is /root/autodl-tmp/data.zip")
    return candidates[0].resolve()
