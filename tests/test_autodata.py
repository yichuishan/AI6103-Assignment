from pathlib import Path
import stat
import zipfile

import pytest

from ai6103.autodata import extract_zip, find_layout


def test_zip_extraction_and_repeatability(tmp_path):
    source=tmp_path/"data.zip"
    with zipfile.ZipFile(source,"w") as z:
        z.writestr("nested/train/breed/a.jpg",b"fixture-image")
    before=source.read_bytes()
    root,digest=extract_zip(source,tmp_path/"work")
    assert (root/"nested/train/breed/a.jpg").read_bytes()==b"fixture-image"
    assert extract_zip(source,tmp_path/"work")== (root,digest)
    assert source.read_bytes()==before


@pytest.mark.parametrize("name",["../escape.jpg","/escape.jpg","C:/escape.jpg",r"..\escape.jpg"])
def test_zip_path_traversal_rejected(tmp_path,name):
    source=tmp_path/"data.zip"
    with zipfile.ZipFile(source,"w") as z:
        z.writestr(name,b"untrusted")
    with pytest.raises(ValueError,match="Unsafe ZIP path"):
        extract_zip(source,tmp_path/"work")
    assert not (tmp_path/"escape.jpg").exists()


def test_zip_symlinks_rejected(tmp_path):
    source=tmp_path/"data.zip"
    info=zipfile.ZipInfo("link")
    info.create_system=3
    info.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(source,"w") as z:
        z.writestr(info,"/etc/passwd")
    with pytest.raises(ValueError,match="symlinks"):
        extract_zip(source,tmp_path/"work")


def test_nested_imagefolder_layout(tmp_path):
    root=tmp_path/"outer/dataset"
    for name in ("train","validation","test"):
        (root/name/"breed").mkdir(parents=True)
    assert find_layout(tmp_path,tmp_path/"work")== (root,None,False)


def test_flat_split_csv_layout(tmp_path):
    for split in ("train","val","test"):
        (tmp_path/split).mkdir()
        (tmp_path/split/f"{split}.jpg").write_bytes(b"fixture")
        (tmp_path/f"{split}.csv").write_text(f"filename,breed\n{split}.jpg,breed_a\n")
    root,manifest,resplit=find_layout(tmp_path,tmp_path/"work")
    assert root==tmp_path and not resplit
    assert "breed_a" in manifest.read_text()
    assert manifest.read_text().count("breed_a")==3


def test_preserve_unknown_split_metadata(tmp_path):
    (tmp_path/"train").mkdir()
    (tmp_path/"labels.csv").write_text("id,breed\nimage,breed_a\n")
    (tmp_path/"test_labels.csv").write_text("unknown,format\nx,y\n")
    with pytest.raises(ValueError,match="Split metadata"):
        find_layout(tmp_path,tmp_path/"work")
