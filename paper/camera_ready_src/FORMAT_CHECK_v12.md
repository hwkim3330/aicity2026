# ECCV 2026 / AI City Challenge format check

Checked on 13 August 2026 against the public ECCV 2026 author policies/template and the AI City Challenge 2026 workshop notice.

- AI City Challenge 2026 explicitly directs workshop-paper authors to follow the ECCV 2026 author guidelines.
- ECCV 2026 requires Springer LNCS style and the ECCV 2026 author kit.
- Final-mode source uses `\documentclass[runningheads]{llncs}` and `\usepackage{eccv}`.
- The paper does not alter page dimensions, margins, or font size.
- Three authors share one institution; redundant institution superscripts are removed.
- More than two authors use `H. Kim et al.` in the running head, matching the ECCV example convention.
- Same-domain emails are grouped in the institute block, matching the ECCV example convention.
- Paper length: 13 pages total including references; below ECCV's general 14-page main-text allowance (references may extend beyond that allowance).
- `hyperref` is enabled without review-only `pagebackref`.
- Four figures are included and rendered.
- Final build has no undefined citations or references.
- Visual render inspection found no clipped text, overlap, black boxes, or broken glyphs.

Note: AI City Challenge 2026 does not publish a separate workshop-specific LaTeX layout on its public notice; it directs authors to the ECCV 2026 author guidelines. Accordingly, this package uses the ECCV/Springer LNCS paper format rather than a custom AI City layout.
