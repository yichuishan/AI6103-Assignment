#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${AI6103_PYTHON:-python}" -u scripts/autodl_runner.py "$@"
