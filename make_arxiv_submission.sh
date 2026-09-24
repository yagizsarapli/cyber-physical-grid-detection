#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/arxiv"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp "$SRC/main.tex" "$SRC/references.bib" "$SRC"/paper_fig*.png "$TMP"/
cd "$TMP"

pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

if grep -Eiq 'undefined references|Citation .* undefined|Reference .* undefined|There were undefined references' main.log; then
  echo "ERROR: unresolved citation/reference warning in LaTeX log."
  exit 1
fi

cp main.pdf "$ROOT/arxiv_preview.pdf"
rm -f "$ROOT/arxiv_submission.zip"
zip -j "$ROOT/arxiv_submission.zip" \
  "$SRC/main.tex" \
  "$SRC/references.bib" \
  "$SRC"/paper_fig*.png >/dev/null

echo "Created:"
echo "  $ROOT/arxiv_preview.pdf"
echo "  $ROOT/arxiv_submission.zip"
echo
echo "Upload arxiv_submission.zip to arXiv and verify arXiv's generated PDF before submitting."
