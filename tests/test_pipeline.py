import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torchvision import transforms as T

from ai6103 import data, engine


@pytest.fixture
def prepared(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        for i in range(n):
            path = tmp_path / f"{split}_{i}.png"
            Image.fromarray(rng.integers(0,256,(16,16,3),dtype=np.uint8)).save(path)
            rows.append({"path": str(path), "label": str(i%2), "split": split})
    classes, counts = data.validate_rows(rows, strict=False)
    return {"rows": rows, "classes": classes, "counts": counts,
            "statistics": data.statistics(rows), "fingerprint": "fixture", "assignment_data": False}


def test_statistics_only_training_and_population_std(prepared):
    stats = data.statistics(prepared["rows"])
    pixels = []
    for r in prepared["rows"]:
        if r["split"] == "train":
            with Image.open(r["path"]) as image:
                pixels.append(np.asarray(T.Resize((100,100),antialias=True)(image), dtype=np.float64).reshape(-1,3)/255)
        else:
            Path(r["path"]).unlink()  # No val/test image is opened to compute stats.
    pixels = np.concatenate(pixels)
    actual = data.statistics(prepared["rows"])
    assert actual == stats
    np.testing.assert_allclose(actual["mean"], pixels.mean(0), atol=1e-12)
    np.testing.assert_allclose(actual["std"], pixels.std(0), atol=1e-12)


def test_disjoint_and_class_validation(prepared):
    rows = copy.deepcopy(prepared["rows"])
    rows[-1]["path"] = rows[0]["path"]
    with pytest.raises(ValueError, match="Repeated image"):
        data.validate_rows(rows, strict=False)
    rows = copy.deepcopy(prepared["rows"])
    rows[-1]["label"] = "unseen"
    with pytest.raises(ValueError, match="Label not in training"):
        data.validate_rows(rows, strict=False)


def test_duplicate_content_across_splits(prepared, tmp_path):
    rows = copy.deepcopy(prepared["rows"])
    destination = tmp_path / "duplicate.png"
    destination.write_bytes(Path(rows[0]["path"]).read_bytes())
    rows[-1]["path"] = str(destination)
    with pytest.raises(ValueError, match="Identical image bytes"):
        data.validate_rows(rows, strict=False)


def test_provided_duplicates_are_audited_without_changing_membership(prepared):
    rows=copy.deepcopy(prepared["rows"])
    original_bytes=Path(rows[0]["path"]).read_bytes()
    for index in (4,6):
        Path(rows[index]["path"]).write_bytes(original_bytes)
    original_membership=[(r["path"],r["label"],r["split"]) for r in rows]
    audit={}
    classes,counts=data.validate_rows(rows,strict=False,duplicate_policy="report",audit=audit)
    assert counts==prepared["counts"] and classes==prepared["classes"]
    assert [(r["path"],r["label"],r["split"]) for r in rows]==original_membership
    assert audit["cross_split_duplicate_groups"]==1
    assert audit["images_in_cross_split_groups"]==3
    assert audit["overlap_images"]=={"train_val":1,"train_test":1,"val_test":1}
    assert all(Path(rows[i]["path"]).read_bytes()==original_bytes for i in (0,4,6))
    with pytest.raises(ValueError,match="Identical image bytes across splits"):
        data.validate_rows(rows,strict=False)


@pytest.mark.parametrize("duplicate_index",[1,5])
def test_supplied_conflicting_labels_are_preserved_and_audited(prepared,duplicate_index):
    rows=copy.deepcopy(prepared["rows"])
    Path(rows[duplicate_index]["path"]).write_bytes(Path(rows[0]["path"]).read_bytes())
    original=[(r["path"],r["split"],r["label"]) for r in rows]
    audit={}
    data.validate_rows(rows,strict=False,duplicate_policy="report",audit=audit)
    assert original==[(r["path"],r["split"],r["label"]) for r in rows]
    assert audit["conflicting_label_groups"]==1
    assert audit["images_in_conflicting_label_groups"]==2
    assert audit["training_images_in_conflicting_label_groups"]==(2 if duplicate_index==1 else 1)
    assert audit["conflicts"][0]["labels"]==["0","1"]
    with pytest.raises(ValueError,match="conflicting labels"):
        data.validate_rows(rows,strict=False)


def test_prepare_retains_duplicate_files_and_records_audit(tmp_path):
    rng=np.random.default_rng(88)
    for split in ("train","val","test"):
        for label in ("a","b"):
            path=tmp_path/"dataset"/split/label/"image.png"
            path.parent.mkdir(parents=True)
            Image.fromarray(rng.integers(0,256,(10,10,3),dtype=np.uint8)).save(path)
    train=tmp_path/"dataset/train/a/image.png"
    val=tmp_path/"dataset/val/a/image.png"
    val.write_bytes(train.read_bytes())
    output=tmp_path/"prepared.json"
    prepared=data.prepare(tmp_path/"dataset",output,strict=False)
    assert prepared["counts"]=={"train":2,"val":2,"test":2}
    assert prepared["data_quality_audit"]["overlap_images"]["train_val"]==1
    assert len(prepared["rows"])==6 and val.read_bytes()==train.read_bytes()
    assert data.load_json(output)["data_quality_audit"]==prepared["data_quality_audit"]


def test_prepare_keeps_original_conflicting_training_annotations(tmp_path):
    rng=np.random.default_rng(89)
    for split in ("train","val","test"):
        for breed in ("japanese_spaniel","papillon"):
            path=tmp_path/"dataset"/split/breed/"image.png"
            path.parent.mkdir(parents=True)
            Image.fromarray(rng.integers(0,256,(10,10,3),dtype=np.uint8)).save(path)
    first=tmp_path/"dataset/train/japanese_spaniel/image.png"
    second=tmp_path/"dataset/train/papillon/image.png"
    second.write_bytes(first.read_bytes())
    prepared=data.prepare(tmp_path/"dataset",tmp_path/"prepared.json",strict=False)
    assert len(prepared["rows"])==6
    assert prepared["data_quality_audit"]["conflicting_label_groups"]==1
    assert prepared["data_quality_audit"]["cross_split_duplicate_groups"]==0
    assert prepared["data_quality_audit"]["conflicts"][0]["labels"]==["japanese_spaniel","papillon"]
    assert {r["label"] for r in prepared["rows"] if r["split"]=="train"}=={"japanese_spaniel","papillon"}


def test_assignment_counts_enforced(prepared):
    with pytest.raises(ValueError, match="Expected"):
        data.validate_rows(prepared["rows"], strict=True)


def test_transforms(prepared):
    ops = data.transform(prepared["statistics"], augment=True).transforms
    assert ops[0].size == (100,100)
    assert ops[1].p == .5
    assert ops[2].padding == 12 and ops[2].fill == 0
    assert ops[2].padding_mode == "constant"
    image, label = data.DogDataset(prepared,"train",True)[0]
    assert image.shape == (3,100,100) and torch.isfinite(image).all()


@pytest.mark.parametrize("layout",["{breed}/images/{file}","images/{breed}/{file}"])
def test_nested_image_wrappers_do_not_become_breed_labels(tmp_path,layout):
    for split in ("train","val","test"):
        for breed in ("breed_a","breed_b"):
            path=tmp_path/split/layout.format(breed=breed,file="fixture.jpg")
            path.parent.mkdir(parents=True)
            path.write_bytes(b"path-only fixture")
    rows,_=data.discover(tmp_path)
    assert {r["label"] for r in rows}=={"breed_a","breed_b"}
    assert len(rows)==6


def test_mixup_convex_targets_and_pixels():
    engine.seed_all(23)
    x = torch.arange(24,dtype=torch.float32).reshape(4,6)
    y = torch.arange(4)
    mixed, ya, yb, lam = engine.mix_batch(x,y,.2)
    assert 0 <= lam <= 1
    assert torch.equal(ya,y)
    assert torch.allclose(mixed, lam*x+(1-lam)*x[yb])
    logits = torch.randn(4,4,requires_grad=True)
    weighted = lam*nn.functional.cross_entropy(logits,ya)+(1-lam)*nn.functional.cross_entropy(logits,yb)
    targets = lam*nn.functional.one_hot(ya,4)+(1-lam)*nn.functional.one_hot(yb,4)
    assert torch.allclose(weighted,nn.functional.cross_entropy(logits,targets))


def test_cosine_reaches_zero_at_horizon():
    p = nn.Parameter(torch.tensor(0.))
    opt = torch.optim.SGD([p],lr=.005,momentum=.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
    for epoch in range(300):
        assert opt.param_groups[0]["lr"] == pytest.approx(.005*.5*(1+np.cos(np.pi*epoch/300)))
        opt.step()
        scheduler.step()
    assert opt.param_groups[0]["lr"] == 0


def tiny_model(classes):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(3,classes))


def test_resume_matches_uninterrupted(prepared, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(engine,"make_model",tiny_model)
    exp = engine.Experiment("resume","schedule",.005,3,"cosine",mixup_alpha=.2,batch_size=2)
    full = engine.run_experiment(prepared,exp,tmp_path/"full",smoke=True)
    paused = engine.run_experiment(prepared,exp,tmp_path/"resumed",smoke=True,stop_after=1)
    assert paused["epochs_completed"] == 1
    resumed = engine.run_experiment(prepared,exp,tmp_path/"resumed",smoke=True)
    for key in ["train_loss","val_loss","train_accuracy","val_accuracy","optimization_loss"]:
        assert resumed["final"][key] == pytest.approx(full["final"][key],abs=1e-10)
    assert resumed["final"]["lr_next"] == 0
    for key, value in engine.checkpoint_load(tmp_path/"full/resume/latest.pt")["model"].items():
        assert torch.equal(value, engine.checkpoint_load(tmp_path/"resumed/resume/latest.pt")["model"][key])


def test_pipeline_never_opens_test_and_cached_test(prepared,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(engine,"make_model",tiny_model)
    original_loader = engine.loader
    splits=[]
    def recording(*args,**kwargs):
        splits.append(args[1])
        return original_loader(*args,**kwargs)
    monkeypatch.setattr(engine,"loader",recording)
    engine.run_pipeline(prepared,tmp_path,smoke=True)
    assert "test" not in splits
    assert len(list(tmp_path.glob("*/summary.json"))) == 8
    selection = data.load_json(tmp_path/"selection.json")
    assert selection["smoke"]
    first=engine.test_once(prepared,tmp_path)
    assert splits.count("test") == 1 and first["metrics"]["n"] == 2
    assert engine.test_once(prepared,tmp_path) == first
    assert splits.count("test") == 1


def test_full_pipeline_rejects_synthetic(prepared,tmp_path):
    with pytest.raises(ValueError, match="Non-assignment"):
        engine.run_pipeline(prepared,tmp_path)


def test_real_efficientnet_backward():
    torch.set_num_threads(2)
    engine.seed_all(42)
    model=engine.make_model(120)
    assert model.classifier[-1].out_features == 120
    model.train()
    logits=model(torch.randn(2,3,100,100))
    loss=nn.functional.cross_entropy(logits,torch.tensor([0,119]))
    loss.backward()
    assert torch.isfinite(loss) and model.features[0][0].weight.grad is not None
    assert torch.isfinite(model.features[0][0].weight.grad).all()
