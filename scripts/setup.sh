#!/usr/bin/env bash
set -euo pipefail
cd /workspace/AI6103-Assignment
VENV_DIR=/workspace/.venvs/ai6103
python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url "${AI6103_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-deps -e .
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" -m pytest -q
