#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PAPER="$ROOT/paper"
FIG="$ROOT/figures"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ -f "$ROOT/results/phase2t_latency_final.csv" && "$ROOT/results/phase2t_latency_final.csv" -nt "$FIG/paper_fig2_latency_waterfall.png" ]]; then
  echo "ERROR: latency CSV is newer than Figure 2."
  echo "Run: python3 29_paper_figures.py"
  exit 1
fi

cp "$PAPER/main.tex" "$PAPER/references.bib" "$TMP"/
python3 - "$TMP/main.tex" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace("../figures/", "")
p.write_text(s)
PY
cp "$FIG"/paper_fig1_multiseed_advantage.png "$TMP"/
cp "$FIG"/paper_fig2_latency_waterfall.png "$TMP"/
cp "$FIG"/paper_fig3_zeroday_generalization.png "$TMP"/
cp "$FIG"/paper_fig4_scale_comparison.png "$TMP"/

cd "$TMP"

pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

if grep -Eiq 'undefined references|Citation .* undefined|Reference .* undefined|There were undefined references' main.log; then
  echo "ERROR: unresolved citation/reference warning in LaTeX log."
  exit 1
fi

if grep -Eiq 'LaTeX Error|Emergency stop|Fatal error occurred' main.log; then
  echo "ERROR: LaTeX reported a fatal error."
  exit 1
fi

cp main.pdf "$ROOT/arxiv_preview.pdf"
rm -f "$ROOT/arxiv_submission.zip"
zip -j "$ROOT/arxiv_submission.zip" \
  main.tex \
  references.bib \
  paper_fig1_multiseed_advantage.png \
  paper_fig2_latency_waterfall.png \
  paper_fig3_zeroday_generalization.png \
  paper_fig4_scale_comparison.png >/dev/null

echo "Created:"
echo "  $ROOT/arxiv_preview.pdf"
echo "  $ROOT/arxiv_submission.zip"
echo
echo "Upload arxiv_submission.zip to arXiv and verify arXiv's generated PDF before submitting."
