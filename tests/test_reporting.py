import json
import hashlib
from pathlib import Path

import pytest

from ai6103.data import load_json, save_json
from ai6103.reporting import generate
from ai6103.reporting import EXPECTED
from ai6103.reporting import loss_number


def test_report_missing_data_remains_incomplete(tmp_path):
    output = tmp_path / "report"
    generate(tmp_path/"missing.json",tmp_path/"runs",output)
    assert load_json(output/"status.json")["report_complete"] is False
    assert "INCOMPLETE" in (output/"status.tex").read_text()
    assert "have not yet been measured" in (output/"statistics.tex").read_text()
    assert "not been evaluated" in (output/"test.tex").read_text()


def test_report_rejects_smoke_statistics(tmp_path):
    prepared = tmp_path/"prepared.json"
    save_json(prepared,{"assignment_data":False})
    with pytest.raises(ValueError,match="synthetic data"):
        generate(prepared,tmp_path/"runs",tmp_path/"out")


def test_report_rejects_smoke_experiment(tmp_path):
    save_json(tmp_path/"runs/lr_0.5/summary.json",{"config":{"smoke":True}})
    with pytest.raises(ValueError,match="smoke-test"):
        generate(tmp_path/"missing.json",tmp_path/"runs",tmp_path/"out")


def test_theoretical_figure_names(tmp_path):
    from ai6103.visualize import theoretical_figures
    theoretical_figures(tmp_path)
    for name in ("beta_alpha_0.2", "cosine_schedule"):
        for extension in ("pdf","png"):
            assert (tmp_path/f"{name}.{extension}").stat().st_size > 1000


def test_smoke_curves_cannot_overwrite_assignment_figures(tmp_path, monkeypatch):
    from ai6103.visualize import curves
    monkeypatch.chdir(tmp_path)
    save_json(tmp_path/"runs/lr_0.5/summary.json",{
        "status":"complete", "config":{"smoke":True,"experiment":{"name":"lr_0.5"}}})
    with pytest.raises(ValueError,match="separate output"):
        curves(tmp_path/"runs",Path("artifacts/figures"))


@pytest.fixture
def complete_report_fixture(tmp_path):
    """Artificial artifact fixture for report logic, never an empirical run."""
    prepared=tmp_path/"prepared.json"
    save_json(prepared,{"assignment_data":True,"fingerprint":"fixture",
                        "statistics":{"mean":[.1,.2,.3],"std":[.2,.2,.2]},
                        "split_provenance":"ARTIFICIAL UNIT TEST FIXTURE"})
    root=tmp_path/"runs"
    for i,name in enumerate(EXPECTED):
        save_json(root/name/"summary.json",{
            "status":"complete","config":{"smoke":False,"data_fingerprint":"fixture","precision":"bf16",
            "experiment":{"name":name,"lr":.005,"schedule":"cosine","weight_decay":.0001}},
            "final":{"epoch":15 if i<3 else 300,"train_loss":2+i*.01,"val_loss":3+i*.01,
                     "train_accuracy":.6+i*.01,"val_accuracy":.4+i*.01},
            "runtime":{"torch":"2.6.0","torchvision":"0.21.0","gpu":"FIXTURE GPU"}})
    save_json(root/"mixup_baseline.json",{"name":"decay_0.0001"})
    save_json(root/"selection.json",{"name":"mixup","data_fingerprint":"fixture"})
    save_json(root/"test_result.json",{
        "name":"mixup","smoke":False,"selection_sha256":hashlib.sha256((root/"selection.json").read_bytes()).hexdigest(),
        "metrics":{"n":1534,"accuracy":.45,"loss":3.2}})
    return prepared,root


def test_complete_report_updates_abstract_and_limitations(complete_report_fixture,tmp_path):
    prepared,root=complete_report_fixture
    output=tmp_path/"report"
    generate(prepared,root,output,author="Student Name")
    assert load_json(output/"status.json")["report_complete"]
    assert "45.00" in (output/"abstract.tex").read_text()
    assert "INCOMPLETE" not in (output/"status.tex").read_text()
    assert "bf16" in (output/"limitations.tex").read_text()
    assert "Student Name" == (output/"author.tex").read_text()
    assert "percentage points" in (output/"mixup_observations.tex").read_text()


def test_report_rejects_test_selection_mismatch(complete_report_fixture,tmp_path):
    prepared,root=complete_report_fixture
    selection=load_json(root/"selection.json")
    selection["name"]="schedule_cosine"
    save_json(root/"selection.json",selection)
    with pytest.raises(ValueError,match="frozen validation selection"):
        generate(prepared,root,tmp_path/"report")


def test_large_losses_use_scientific_notation_in_table():
    assert "10^{28}" in loss_number(1.23e28)
    assert loss_number(2.5)=="2.5000"


def test_small_nonzero_losses_are_not_displayed_as_zero():
    assert loss_number(1.373700594768536e-5)==r"$1.374\times 10^{-5}$"
    assert loss_number(0)=="0.0000"


def test_report_discloses_supplied_split_overlap(complete_report_fixture,tmp_path):
    prepared_path,root=complete_report_fixture
    prepared=load_json(prepared_path)
    prepared["data_quality_audit"]={"cross_split_duplicate_groups":2,
            "overlap_images":{"train_val":1,"train_test":2,"val_test":0},"groups":[]}
    save_json(prepared_path,prepared)
    output=tmp_path/"report"
    generate(prepared_path,root,output)
    assert "limits the independence" in (output/"abstract.tex").read_text()
    assert "1 validation images duplicate training bytes" in (output/"statistics.tex").read_text()
    assert "2 test images duplicate training bytes" in (output/"limitations.tex").read_text()
    assert "not an independent unseen-image estimate" in (output/"test.tex").read_text()
    assert load_json(output/"data_quality_audit.json")==prepared["data_quality_audit"]


def test_training_label_noise_is_not_misreported_as_test_leakage(complete_report_fixture,tmp_path):
    prepared_path,root=complete_report_fixture
    prepared=load_json(prepared_path)
    prepared["data_quality_audit"]={"cross_split_duplicate_groups":0,
            "overlap_images":{"train_val":0,"train_test":0,"val_test":0},"groups":[],
            "conflicting_label_groups":1,"images_in_conflicting_label_groups":2,
            "training_images_in_conflicting_label_groups":2}
    save_json(prepared_path,prepared)
    output=tmp_path/"report"
    generate(prepared_path,root,output)
    assert "label noise" in (output/"abstract.tex").read_text()
    assert "limits the independence" not in (output/"abstract.tex").read_text()
    assert "inconsistent labels" in (output/"statistics.tex").read_text()
    assert "annotation conflicts" in (output/"test.tex").read_text()
    assert "not an independent unseen-image estimate" not in (output/"test.tex").read_text()
    assert "2 training images" in (output/"limitations.tex").read_text()


def test_cleaned_report_uses_retained_counts_and_no_obsolete_leakage_claim(complete_report_fixture,tmp_path):
    prepared_path,root=complete_report_fixture
    prepared=load_json(prepared_path)
    # Artificial cleaning totals for report logic, not measurements of the dataset.
    prepared.update({
        "source_counts":{"train":7155,"val":1533,"test":1534},
        "counts":{"train":7140,"val":1530,"test":1531},
        "data_quality_audit":{"total_duplicate_groups":15,"cross_split_duplicate_groups":10,
                              "conflicting_label_groups":6},
        "cleaning":{"removed_conflict_images":12,"removed_duplicate_images":9,
                    "after":{"cross_split_duplicate_groups":0,"conflicting_label_groups":0}},
    })
    save_json(prepared_path,prepared)
    test=load_json(root/"test_result.json")
    test["metrics"]["n"]=1531
    save_json(root/"test_result.json",test)
    output=tmp_path/"cleaned-report"
    generate(prepared_path,root,output)
    summary=(output/"data_summary.tex").read_text()
    assert r"\newcommand{\TrainingCount}{7140}" in summary
    assert "7,140, 1,530, and 1,531" in summary
    statistics=(output/"statistics.tex").read_text()
    assert "12 conflicting-label records and 9 duplicate copies" in statistics
    assert "No byte-identical images remain" in statistics
    assert "1531 images" in (output/"test.tex").read_text()
    assert "annotation conflicts" not in (output/"test.tex").read_text()
    assert "limits the independence" not in (output/"abstract.tex").read_text()
    assert "preserved" not in statistics
