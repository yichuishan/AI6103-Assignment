#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${1:-$SCRIPT_DIR/../report}"
trap 'tail -n 35 build-pass*.log build-bibtex.log 2>/dev/null || true' ERR
pdflatex -interaction=nonstopmode -halt-on-error report.tex > build-pass1.log
bibtex report > build-bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error report.tex > build-pass2.log
pdflatex -interaction=nonstopmode -halt-on-error report.tex > build-pass3.log
pdfinfo report.pdf | sed -n '/^Pages:/p'
pages=$(pdfinfo report.pdf | awk '/^Pages:/ {print $2}')
body_pages=$(python -c 'import re; from pathlib import Path; s=Path("report.aux").read_text(); m=re.search(r"\\newlabel\{body-end\}\{\{[^}]*\}\{(\d+)\}",s); print(m.group(1) if m else 999)')
if [ "$body_pages" -gt 6 ]; then
  echo "Report body exceeds six pages. Edit prose, never shrink the style." >&2
  exit 1
fi
echo "Built report/report.pdf (AAAI 2024 style, $body_pages body pages, $pages total pages)."
