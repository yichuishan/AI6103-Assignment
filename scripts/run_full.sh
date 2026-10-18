#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${AI6103_PYTHON:-python}"
PREPARED="${AI6103_PREPARED:-data/prepared.json}"
RUNS="${AI6103_RUNS:-runs/full}"
DEVICE="${AI6103_DEVICE:-cuda}"
"$PYTHON_BIN" -m ai6103.cli run --prepared "$PREPARED" --runs "$RUNS" --device "$DEVICE" --workers 0
# The training command must finish all eight conditions before test evaluation.
"$PYTHON_BIN" -m ai6103.cli test --prepared "$PREPARED" --runs "$RUNS" --device "$DEVICE"
"$PYTHON_BIN" -m ai6103.cli plot --runs "$RUNS" --output artifacts/figures
"$PYTHON_BIN" -m ai6103.cli report --prepared "$PREPARED" --runs "$RUNS"
bash scripts/build_report.sh
