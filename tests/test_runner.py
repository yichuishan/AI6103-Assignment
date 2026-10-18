import fcntl
from pathlib import Path
import subprocess
import sys


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
