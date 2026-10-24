import fcntl
from pathlib import Path
import subprocess
import sys
import json

import pytest

from scripts.autodl_runner import check_input_binding, parse_args


def test_second_launcher_cannot_start_duplicate_workflow(tmp_path):
    lock=(tmp_path/".pipeline.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        script=Path(__file__).resolve().parents[1]/"scripts/autodl_runner.py"
        result=subprocess.run([sys.executable,str(script),"--smoke","--output",str(tmp_path)],
                              capture_output=True,text=True,timeout=15)
        assert result.returncode!=0
        assert "already running" in result.stderr
        assert not (tmp_path/"status.json").exists()
    finally:
        lock.close()


def test_failed_preparation_can_accept_corrected_label_manifest(tmp_path):
    previous={"root":"dataset","manifest":None}
    (tmp_path/"input.json").write_text(json.dumps(previous))
    corrected={"root":"dataset","manifest":"detected_manifest.csv"}
    check_input_binding(tmp_path,corrected)
    backups=list(tmp_path.glob("input-before-failed-preparation-*.json"))
    assert len(backups)==1 and json.loads(backups[0].read_text())==previous
    assert json.loads((tmp_path/"input.json").read_text())==previous


@pytest.mark.parametrize("artifact",["prepared.json","runs/lr_0.5/latest.pt"])
def test_corrected_binding_does_not_reuse_prepared_or_trained_run(tmp_path,artifact):
    (tmp_path/"input.json").write_text(json.dumps({"manifest":None}))
    target=tmp_path/artifact
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text("existing data")
    with pytest.raises(ValueError,match="another data/configuration"):
        check_input_binding(tmp_path,{"manifest":"different.csv"})


def test_cleaned_workflow_uses_a_separate_default_output():
    assert parse_args([]).output.name == "results_cleaned"


def test_cleaning_policy_change_cannot_resume_original_run(tmp_path):
    previous={"seed":42,"manifest":"dataset.csv"}
    (tmp_path/"input.json").write_text(json.dumps(previous))
    (tmp_path/"prepared.json").write_text("original uncleaned data")
    with pytest.raises(ValueError,match="another data/configuration"):
        check_input_binding(tmp_path,{**previous,"cleaning_policy":"new-policy"})
