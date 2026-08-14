# v12 revision summary

This revision applies the internal review comments received on 12-13 August 2026 while preserving the accepted paper's official results and evidence boundaries.

## Authorship and affiliation
- Author order: **Hyunwoo Kim, Yooseung Wang, Pusik Park**.
- Pusik Park is marked as the corresponding author.
- A single KETI affiliation is used, so redundant `1 1 1` institution superscripts were removed.
- Same-domain email addresses are grouped as `{hwkim3, yswang, pusik.park}@keti.re.kr`.
- Running author is `H. Kim et al.` for three authors.

## Paper narrative
- Removed response-to-reviewer / revision-meta wording from the paper body.
- Replaced contribution-style overclaiming with a task-wise **key findings** summary.
- Added concise TAR/FETV/PSI-VQA findings in the Introduction, grounded in the archived official artifacts and controlled analyses.
- Renamed **Task Setting** to **Task Description**.
- Converted the short TAR/FETV/PSI task subsections in Section 3 to paragraph-level headings.
- Converted the short result subsections in Section 5 to paragraph-level headings.
- Renamed **Controlled Diagnostics** to **Controlled Analyses**.
- Rephrased the temporal-baseline discussion as ordinary paper prose rather than a response to reviewers.
- Reframed the former post-deadline section as **Design Implications and Exploratory Prototypes** and states only that those prototypes are outside the official submissions.

## Evidence and figures
- Retained four figures: shared system overview, FETV field-score analysis, PSI temporal-baseline analysis, and a worked PSI answer-contract failure case.
- Official results remain TAR 0.4256, FETV 0.4634, and PSI-VQA 57.04.
- Controlled PSI MCQ prompt study remains generic 8/24, shipped routed 3/24, red-box-aware 9/24.
- Temporal-baseline analysis remains video VLM 0.4617 held-out mIoU versus metadata-only baselines up to 0.5566 under held-out refitting.
- QLoRA is described as an attempted but uncontrolled development experiment; no fine-tuned checkpoint is claimed to have produced an official submission.

## Acknowledgements
Both requested funding sources are included:
1. IITP / Ministry of Science and ICT, No. RS-2025-02283230.
2. KEIT / Ministry of Trade, Industry and Resources (MOTIR), No. RS-2024-00404601.

## Build status
- pdfLaTeX + BibTeX build succeeds.
- 13 PDF pages total.
- No undefined citations or references in the final pass.
- Four figures render without clipping or overlap.
