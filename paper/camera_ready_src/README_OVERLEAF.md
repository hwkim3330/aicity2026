# Overleaf upload instructions

## Recommended workflow
1. In Overleaf choose **New Project -> Upload Project**.
2. Upload `Korea_Drive_ECCV2026_Camera_Ready_v12_Overleaf.zip`.
3. Confirm the main document is `main.tex`.
4. Compiler: **pdfLaTeX**.
5. Recompile. Bibliography is BibTeX (`main.bib`, `splncs04`).

## Format basis
The AI City Challenge 2026 workshop page directs workshop-paper authors to follow the **ECCV 2026 author guidelines**, and accepted papers are planned for Springer publication. The manuscript therefore uses the ECCV 2026 / Springer LNCS structure:
- `\documentclass[runningheads]{llncs}`
- final-mode `\usepackage{eccv}`
- `eccvabbrv`
- `hyperref` without review-only page back-references
- Springer `splncs04` bibliography style

Official references:
- https://www.aicitychallenge.org/
- https://eccv.ecva.net/Conferences/2026/SubmissionPolicies
- https://github.com/paolo-favaro/paper-template

## Current paper metadata
- Authors: Hyunwoo Kim, Yooseung Wang, Pusik Park
- Corresponding author: Pusik Park
- Institution: Korea Electronics Technology Institute (KETI)
- Email: `{hwkim3, yswang, pusik.park}@keti.re.kr`

## Files
- `main.tex`: manuscript
- `main.bib`: bibliography
- `fig_system_overview.pdf`: system figure
- `fig_fetv_fields.pdf`: FETV field-score figure
- `fig_psi_temporal.pdf`: PSI temporal figure
- `eccv.sty`, `eccvabbrv.sty`, `llncs.cls`, `splncs04.bst`: LaTeX support files
- `build.sh`: local build helper

Do not modify page dimensions, margins, base font sizes, or other layout parameters in the ECCV/LNCS template.
