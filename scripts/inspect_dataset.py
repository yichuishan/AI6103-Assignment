"""Read-only inspection of a dataset directory or ZIP, without extraction."""
import argparse
from collections import Counter
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    print("DATASET_PATH:", path)
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            names = [entry.filename for entry in entries]
            print("ZIP_FILES:", len(entries))
            print("UNCOMPRESSED_GIB:", round(sum(e.file_size for e in entries)/1024**3, 3))
            print("COMPRESSED_GIB:", round(sum(e.compress_size for e in entries)/1024**3, 3))
    elif path.is_dir():
        names = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]
        print("FILES:", len(names))
    else:
        raise SystemExit("Path is neither a readable ZIP nor a dataset directory")
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    print("IMAGES:", sum(Path(n).suffix.lower() in image_suffixes for n in names))
    print("FIRST_LEVEL_COUNTS:", dict(Counter(n.replace('\\', '/').split('/')[0] for n in names)))
    print("METADATA_FILES:", [n for n in names if Path(n).suffix.lower() in {".csv", ".json", ".txt"}][:30])
    print("EXAMPLE_PATHS:")
    for name in sorted(names)[:20]:
        print(name)


if __name__ == "__main__":
    main()
