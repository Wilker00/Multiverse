# Paper Build Notes

This folder contains a submission-style LaTeX source for the camera-ready draft.

## Files

- `paper/main.tex`: main manuscript source.

## Build (local TeX install)

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper/main.tex
pdflatex -interaction=nonstopmode -halt-on-error paper/main.tex
```

Two passes are recommended for stable table/cross-reference layout.

## Build (Overleaf)

1. Create a new Overleaf project.
2. Upload `paper/main.tex`.
3. Compile with pdfLaTeX.

## Source of Truth

The content is aligned to:

- `docs/PAPER_CAMERA_READY.md`
- artifact JSON files under `models/paper/paper_readiness/latest/` and `models/validation/`
