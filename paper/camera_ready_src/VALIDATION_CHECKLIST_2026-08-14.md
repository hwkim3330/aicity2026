# KoreaDrive camera-ready validation checklist

Evidence baseline: repository commit `d100b24` (2026-08-14) and the portal exports documented in `ABLATIONS.md`, `REPRODUCE.md`, and `paper/FINAL_CHECK.md`.

Status legend: **DONE** is reflected in the manuscript; **SIGN-OFF** requires an author decision; **KEEP BOUNDED** is a known limitation whose wording must not be strengthened.

## P0 — before submission

| Status | Item | Evidence / current treatment | Required action |
|---|---|---|---|
| **SIGN-OFF** | Permission to publish the FETV frame | `frames/README.md`: the FETV repository exposes public example frames but provides no LICENSE or explicit reuse grant. The paper uses one downscaled, prediction-annotated frame from public clip `019_004.mp4` with FETV/FishEye8K attribution. | Corresponding author must approve publication or remove Fig. 4. This is the only unresolved image-rights decision. |
| **DONE** | Page-limit interpretation | Verified PDF: manuscript and acknowledgements end on p.14; pages 15–16 contain references only. Both the AI City submission page and ECCV 2026 policy exclude references from the 14-page limit. | Do not add non-reference content after p.14. |
| **SIGN-OFF** | Funding text | Acknowledgements retain the IITP and KEIT project names and grant numbers. | Project lead checks agency names, ministry name, grant numbers, and required wording. |
| **DONE** | Figure 1 architecture | Benchmark input connects directly to one shared frozen Qwen3-VL-8B-Instruct backbone. A deterministic selector visibly branches to TAR/FETV/PSI-VQA, with nested frame/prompt/output boxes. | Do not reintroduce logos, rounded cards, scores, or `+`-style detail lists. |
| **DONE** | Prompt-program claim boundary | Controlled PSI prompt result is 3/24 routed, 8/24 generic, 9/24 box-aware. No aggregate prompt-only improvement is claimed. Yellow in Fig. 1 is explicitly descriptive. | Do not add a statement that the shipped prompt program improved leaderboard performance. |
| **DONE** | PSI denominator boundary | Contract-failure rates use all available generations (7/47 routed, 6/24 generic, 4/24 box-aware); Table 5 accuracy uses only the 24-item intersection common to all conditions. | Preserve the distinction between per-condition availability and paired analysis. |
| **DONE** | Ambiguous `official pipeline` wording | Manuscript uses `submitted pipeline`, `leaderboard-submission pipeline`, and `scored submission artifacts`; `official` is reserved for organizer-defined results/metrics where appropriate. | Preserve this terminology. |
| **KEEP BOUNDED** | Organizer Qwen baseline | The board exposes the label `Qwen3-VL-8B-Instruct` and score 0.3143 but not exact revision, precision, frames, prompts, or decoding. The paper calls it an external same-model-label reference. | Upgrade to a controlled same-checkpoint claim only if the organizer provides the missing configuration. |
| **KEEP BOUNDED** | FETV v6→v7 result | Score 0.4238→0.4584; ten non-temporal fields are byte-identical; description contributes +0.0282 and date/time +0.0065. Two components changed jointly. | Keep as a historical submission-artifact association, not a one-factor ablation or prompt attribution. |
| **DONE** | Frame-redistribution wording | Fig. 5 now says specifically that **PSI-VQA frames** are not redistributed; it no longer conflicts with the FETV frame figure. | Preserve the benchmark-specific wording. |
| **DONE** | Git/frame provenance | Paper footnote cites commit `d100b24`; the packaged annotated image matches the Git asset added at `1552b38` and is listed in `SHA256SUMS.txt`. | If Git changes again, update the commit in `main.tex` and rerun hashes/build. |
| **SIGN-OFF** | Final leaderboard refresh | Current values: TAR 0.4256 (24/27), FETV 0.4634 (3/8), PSI-VQA 57.04 (5/7); Cosmos3-Super 0.5729; organizer Qwen3-VL-8B label 0.3143. | Refetch portal exports immediately before submission and compare every cited rank/score. |
| **DONE** | Build integrity | pdfLaTeX/BibTeX build completes with no undefined references, missing citations, overfull boxes, or underfull boxes. Title, running head, and PDF metadata are aligned. | Recompile once on Overleaf and inspect its log. |

## P1 — known provenance and reproducibility limits

| Status | Item | Evidence / required wording |
|---|---|---|
| **KEEP BOUNDED** | FETV v11 exact reproduction | Current command reproduces a bit-stable single pass and matches `v6_fewshot` on 1742/2600 fields (67.0%), not the complete v11 chain. v9/v10 generating code is absent; v10→v11 descriptions are exactly recovered. |
| **KEEP BOUNDED** | PSI/TAR sampled voting | Original sampling seeds were never recorded. PSI BCQ (55 rows) and TAR BCQ cannot be reproduced byte-for-byte; current deterministic controls apply only to reruns. |
| **KEEP BOUNDED** | Recovered Qwen revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` was reconstructed from Hub history and cache evidence, not persisted by the original run. |
| **KEEP BOUNDED** | Runtime and VRAM | 18.7 s load, 469.3 s inference, and 16.78 GiB are reproduction-run measurements, not challenge-run logs. |
| **KEEP BOUNDED** | QLoRA | 49/77 exists without a matched frozen control. It documents development provenance only and cannot support a fine-tuning gain. |
| **KEEP BOUNDED** | Small prompt/frame studies | PSI prompt comparison has 24 paired items; TAR frame comparison is 16/25 vs 15/25. Directions are reportable; broad effect claims are not. |
| **KEEP BOUNDED** | Temporal-prior optimism | Shipped constants score 0.5724 in-sample vs 0.5566 held out; the 0.0158 difference is reported as fitting optimism. |
| **KEEP BOUNDED** | OpenQA local scorer mismatch | Held-out local estimate 0.7816 exceeds organizer score 0.6019; the manuscript treats this as scorer sensitivity, not expected leaderboard performance. |
| **SIGN-OFF** | Dataset/model-use statement | Manuscript states no private data or private models. Confirm every benchmark/source-dataset access term, especially FETV image publication, before signing. |

## Artifact identity

| Artifact | Records | Expected SHA256 |
|---|---:|---|
| `submission_qwen3vl8b_v9.csv` | 960 | `243a5e8b67310428096cfc760ddeedaf5bc9d280729ad73f4c940eb3da759f6f` |
| `fetv_submission_v11.json` | 200 | `39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835` |
| `psi_vqa_submission_v7.csv` | 328 | `a3829a36f591907bb8838098b1cc61feb907fec1cc6215f6098094aafaafb110` |

## Final mechanical check

1. Upload the ZIP as a new Overleaf project and set `main.tex` / pdfLaTeX.
2. Confirm 16 PDF pages, with references beginning on p.15.
3. Inspect pp.1–2 (abstract/introduction), p.5 (Fig. 1), pp.12–13 (diagnostic figures), and p.14 (conclusion/acknowledgements).
4. Search the PDF for `official pipeline`, `Qwen2.5`, `0.5748`, `Capability Audit`, and stale commit hashes; expected count is zero.
5. Archive the submitted ZIP and compiled PDF with their SHA256 values.
