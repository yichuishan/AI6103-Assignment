import copy
from pathlib import Path

import pytest

from ai6103.data import clean_rows, validate_rows


def test_cleaning_resolves_all_group_types_independent_of_manifest_order(tmp_path):
    rows = []
    def image(name, split, label, content):
        path = tmp_path / name
        path.write_bytes(content)
        rows.append({"path": str(path), "split": split, "label": label})

    # A three-way duplicate keeps only test, even when training appears first.
    image("train-a", "train", "a", b"three-way")
    image("val-a", "val", "a", b"three-way")
    image("test-a", "test", "a", b"three-way")
    # Within-split duplication keeps the lexically first path.
    image("train-z", "train", "a", b"within")
    image("train-b", "train", "a", b"within")
    # A majority label does not rescue a conflicting group.
    image("conflict1", "train", "a", b"conflict")
    image("conflict2", "val", "a", b"conflict")
    image("conflict3", "test", "b", b"conflict")
    # Unique examples keep both classes represented in all splits.
    for split in ("train", "val", "test"):
        for label in ("a", "b"):
            image(f"{split}-unique-{label}", split, label, f"{split}-{label}".encode())
    before = {}
    validate_rows(rows, strict=False, duplicate_policy="report", audit=before)
    assert before["total_duplicate_groups"] == 3
    assert before["images_in_duplicate_groups"] == 8
    assert before["within_split_duplicate_groups"] == 1
    original_bytes = {r["path"]: Path(r["path"]).read_bytes() for r in rows}
    kept, summary = clean_rows(rows)
    assert clean_rows(list(reversed(rows))) == (kept, summary)
    assert summary["total_removed"] == 6
    assert summary["removed_conflict_images"] == 3
    assert summary["removed_duplicate_images"] == 3
    assert summary["removed_counts"] == {"train": 3, "val": 2, "test": 1}
    names = {Path(r["path"]).name for r in kept}
    assert "test-a" in names and "train-b" in names
    assert not {"train-a", "val-a", "train-z", "conflict1", "conflict2", "conflict3"} & names
    after = {}
    validate_rows(kept, strict=False, audit=after)
    assert after["total_duplicate_groups"] == after["conflicting_label_groups"] == 0
    assert clean_rows(kept)[1]["total_removed"] == 0
    assert {path: Path(path).read_bytes() for path in original_bytes} == original_bytes


def test_cleaning_does_not_merge_same_names_with_different_bytes(tmp_path):
    rows = []
    for split in ("train", "val", "test"):
        path = tmp_path / split / "same.jpg"
        path.parent.mkdir()
        path.write_bytes(split.encode())
        rows.append({"path": str(path), "split": split, "label": "a"})
    validate_rows(rows, strict=False)
    assert clean_rows(rows)[1]["total_removed"] == 0


def test_training_class_loss_aborts_preparation(tmp_path, monkeypatch):
    from ai6103 import data
    rows = []
    for name, split, label, content in (
        ("ta", "train", "a", b"conflict"),
        ("tb", "train", "b", b"train-b"),
        ("vb", "val", "b", b"conflict"),
        ("va", "val", "a", b"val-a"),
        ("xa", "test", "a", b"test-a"),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        rows.append({"path": str(path), "split": split, "label": label})
    monkeypatch.setattr(data, "discover", lambda *args: (copy.deepcopy(rows), "fixture"))
    with pytest.raises(ValueError, match="entire training class"):
        data.prepare(tmp_path, tmp_path / "prepared.json", strict=False)
    assert not (tmp_path / "prepared.json").exists()


def test_formal_protocol_is_unchanged_and_rejects_uncleaned_preparation(tmp_path):
    from ai6103.engine import lr_experiments, run_pipeline
    formal = lr_experiments()
    assert [e.lr for e in formal] == [0.5, 0.005, 0.001]
    assert all(e.batch_size == 128 and e.epochs == 15 for e in formal)
    assert max(e.lr for e in lr_experiments(smoke=True)) < 0.001
    with pytest.raises(ValueError, match="old preprocessing policy"):
        run_pipeline({"assignment_data": True}, tmp_path / "old-runs")
    assert not (tmp_path / "old-runs").exists()
