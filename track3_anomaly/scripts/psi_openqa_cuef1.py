#!/usr/bin/env python3
"""Local reimplementation of the PSI-VQA Open QA (PSI-T2) Cue-F1 metric.

Official definition (aicitychallenge.org/2026-track3/):
  "Cue-level F1 using sentence-transformer (all-MiniLM-L6-v2) semantic
   matching at cosine threshold 0.55. Prediction cues are matched against
   GT cues; precision and recall are computed at the cue level and
   averaged into F1. Where the GT is 'None', predicting 'None' scores 1.0
   and predicting cues scores 0.0."

Cue parsing is not published; we support two variants to test robustness:
  - "bullets": only lines starting with a bullet marker are cues
  - "lines":   every non-empty line is a cue (bullet markers stripped)
"""
import re

import numpy as np

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def parse_cues(text, variant="lines"):
    text = (text or "").strip()
    if not text or text.lower().rstrip(".") == "none":
        return []
    cues = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        is_bullet = bool(re.match(r"^[-*•]\s+|^\d+[.)]\s+", line))
        if variant == "bullets" and not is_bullet:
            continue
        line = re.sub(r"^[-*•]\s+|^\d+[.)]\s+", "", line).strip()
        if line:
            cues.append(line)
    if variant == "bullets" and not cues:
        # no bullet lines at all -> treat whole text as one cue
        cues = [text]
    return cues


def item_prf(pred_cues, gt_cues, pred_is_none, gt_is_none, sim=None):
    """sim: precomputed cosine matrix [len(pred), len(gt)]."""
    if gt_is_none:
        return (1.0, 1.0, 1.0) if pred_is_none else (0.0, 0.0, 0.0)
    if pred_is_none or not pred_cues:
        return (0.0, 0.0, 0.0)
    matched_pred = (sim.max(axis=1) >= 0.55).sum()
    matched_gt = (sim.max(axis=0) >= 0.55).sum()
    p = matched_pred / len(pred_cues)
    r = matched_gt / len(gt_cues)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return (p, r, f1)


def score_all(preds, gts, variant="lines"):
    """preds, gts: parallel lists of raw answer strings.
    Returns (mean_f1, mean_p, mean_r, per_item_f1_list)."""
    m = _model()
    parsed = []
    to_embed = []
    for pr, gt in zip(preds, gts):
        pc = parse_cues(pr, variant)
        gc = parse_cues(gt, variant)
        p_none = len(pc) == 0
        g_none = (gt or "").strip().lower().rstrip(".") == "none"
        parsed.append((pc, gc, p_none, g_none))
        to_embed.extend(pc)
        to_embed.extend(gc)
    if to_embed:
        embs = m.encode(to_embed, batch_size=256, show_progress_bar=False,
                        normalize_embeddings=True)
    ps, rs, f1s = [], [], []
    i = 0
    for pc, gc, p_none, g_none in parsed:
        pe = embs[i:i + len(pc)]; i += len(pc)
        ge = embs[i:i + len(gc)]; i += len(gc)
        sim = pe @ ge.T if (len(pc) and len(gc)) else None
        p, r, f1 = item_prf(pc, gc, p_none, g_none, sim)
        ps.append(p); rs.append(r); f1s.append(f1)
    return float(np.mean(f1s)), float(np.mean(ps)), float(np.mean(rs)), f1s
