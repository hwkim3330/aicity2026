#!/usr/bin/env python3
"""Video-grouped cross-validation of the PSI-VQA temporal-localization prior.

Every parameterized baseline is re-fitted from scratch on each split's
TRAINING half only and scored on that split's held-out half. No parameter is
ever selected using held-out data. The one exception is deliberately labelled:
``shipped_prior_full_fit`` replays the exact constants that were shipped in
``psi_vqa_submission_v6/v7.csv``; those constants were fitted once on all 227
training windows, so their per-split numbers are reported as NOT held out.

Baselines
---------
random               uniform random interval, averaged over ``--random-draws``
center               centered window, width ratio fitted on train
global_mean          mean start/end ratio over train windows
duration_prior       duration-stratified [lo*dur, hi*dur] prior; threshold and
                     both strata fitted jointly on train by grid search
vlm                  model predictions, if ``--vlm-preds`` is supplied
vlm_prior_blend      endpoint average of vlm and duration_prior (needs --vlm-preds)
shipped_prior_full_fit   the shipped constants (NOT held out; see above)

Usage
-----
    python3 temporal_prior_baselines.py \
        --gt   ../data/psi_vqa/train/temporal_localization.json \
        --videos ../data/psi_vqa/train/videos/temporal \
        --out-csv  ../results/temporal_prior_cv.csv \
        --out-json ../results/temporal_prior_summary.json
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np

# Constants shipped in psi_vqa_submission_v6.csv / v7.csv, fitted once on the
# full 227-window training set. Kept here only so the CV can quantify how much
# that full-set fit flatters itself relative to a properly held-out refit.
SHIPPED_THRESHOLD_S = 10.0
SHIPPED_SHORT = (0.30, 0.70)
SHIPPED_LONG = (0.20, 0.62)

RATIO_GRID = np.round(np.arange(0.00, 1.001, 0.02), 3)
THRESHOLD_GRID = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
WIDTH_GRID = np.round(np.arange(0.05, 1.001, 0.01), 3)

_TS = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_ts(text):
    """'MM:SS.ss' or 'SS.ss' -> seconds."""
    m = _TS.match(text.strip())
    if not m:
        raise ValueError(f"unparseable timestamp: {text!r}")
    minutes = float(m.group(1) or 0.0)
    return minutes * 60.0 + float(m.group(2))


def parse_window(answer):
    obj = json.loads(answer)
    return parse_ts(obj["start"]), parse_ts(obj["end"])


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def load_durations(items, video_root, cache_path):
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    missing = []
    for it in items:
        vid = it["video_id"]
        if vid in cache:
            continue
        rel = vid.split("PSI/", 1)[-1]
        rel = rel.split("temporal/", 1)[-1]
        path = os.path.join(video_root, rel)
        if not os.path.exists(path):
            missing.append(vid)
            continue
        cache[vid] = ffprobe_duration(path)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump(cache, open(cache_path, "w"), indent=1, sort_keys=True)
    return cache, missing


def iou(pred, gt):
    ps, pe = pred
    gs, ge = gt
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def miou(preds, gts):
    return float(np.mean([iou(p, g) for p, g in zip(preds, gts)]))


def ratio_windows(durs, lo, hi):
    return [(lo * d, max(hi * d, lo * d + 1e-6)) for d in durs]


# --------------------------------------------------------------------------
# fitters: each takes the TRAINING half only and returns a params dict
# --------------------------------------------------------------------------

def fit_center(durs, gts):
    best, best_w = -1.0, None
    for w in WIDTH_GRID:
        lo, hi = (1.0 - w) / 2.0, (1.0 + w) / 2.0
        s = miou(ratio_windows(durs, lo, hi), gts)
        if s > best:
            best, best_w = s, float(w)
    return {"width_ratio": best_w, "train_miou": best}


def fit_global_mean(durs, gts):
    lo = float(np.mean([g[0] / d for d, g in zip(durs, gts)]))
    hi = float(np.mean([g[1] / d for d, g in zip(durs, gts)]))
    return {"lo": lo, "hi": hi,
            "train_miou": miou(ratio_windows(durs, lo, hi), gts)}


def _best_ratio_pair(durs, gts):
    """Exhaustive (lo, hi) grid search on one stratum."""
    if not durs:
        return None, None, 0.0
    best, best_pair = -1.0, (0.0, 1.0)
    for lo in RATIO_GRID:
        for hi in RATIO_GRID:
            if hi <= lo:
                continue
            s = miou(ratio_windows(durs, lo, hi), gts)
            if s > best:
                best, best_pair = s, (float(lo), float(hi))
    return best_pair[0], best_pair[1], best


def fit_duration_prior(durs, gts):
    best = {"train_miou": -1.0}
    for thr in THRESHOLD_GRID:
        sd = [d for d in durs if d < thr]
        sg = [g for d, g in zip(durs, gts) if d < thr]
        ld = [d for d in durs if d >= thr]
        lg = [g for d, g in zip(durs, gts) if d >= thr]
        s_lo, s_hi, _ = _best_ratio_pair(sd, sg)
        l_lo, l_hi, _ = _best_ratio_pair(ld, lg)
        preds = []
        for d in durs:
            lo, hi = (s_lo, s_hi) if d < thr else (l_lo, l_hi)
            if lo is None:                      # empty stratum on this split
                lo, hi = (l_lo, l_hi) if d < thr else (s_lo, s_hi)
            preds.append((lo * d, max(hi * d, lo * d + 1e-6)))
        score = miou(preds, gts)
        if score > best["train_miou"]:
            best = {"threshold_s": thr, "short": [s_lo, s_hi],
                    "long": [l_lo, l_hi], "train_miou": score,
                    "n_short": len(sd), "n_long": len(ld)}
    return best


def apply_duration_prior(durs, params):
    thr = params["threshold_s"]
    out = []
    for d in durs:
        lo, hi = params["short"] if d < thr else params["long"]
        if lo is None:
            lo, hi = params["long"] if d < thr else params["short"]
        out.append((lo * d, max(hi * d, lo * d + 1e-6)))
    return out


def apply_shipped(durs):
    out = []
    for d in durs:
        lo, hi = SHIPPED_SHORT if d < SHIPPED_THRESHOLD_S else SHIPPED_LONG
        # the shipped script rounded to whole seconds and forced end > start
        s, e = round(lo * d), round(hi * d)
        if e <= s:
            e = s + 1
        out.append((float(s), float(e)))
    return out


def random_windows(durs, rng):
    out = []
    for d in durs:
        a, b = sorted(rng.uniform(0.0, d, size=2))
        if b - a < 1e-6:
            b = min(d, a + 1.0)
        out.append((float(a), float(b)))
    return out


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="../data/psi_vqa/train/temporal_localization.json")
    ap.add_argument("--videos", default="../data/psi_vqa/train/videos/temporal")
    ap.add_argument("--duration-cache", default="../results/psi_temporal_durations.json")
    ap.add_argument("--vlm-preds", default=None,
                    help="JSONL with {item_index, prediction} model answers on the "
                         "training split; enables the vlm and vlm_prior_blend rows")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-draws", type=int, default=20)
    ap.add_argument("--out-csv", default="../results/temporal_prior_cv.csv")
    ap.add_argument("--out-json", default="../results/temporal_prior_summary.json")
    args = ap.parse_args()

    items = json.load(open(args.gt))["items"]
    durations, missing = load_durations(items, args.videos, args.duration_cache)
    if missing:
        print(f"[warn] {len(missing)} videos missing; those items are dropped",
              file=sys.stderr)
    items = [it for it in items if it["video_id"] in durations]
    if not items:
        sys.exit("no items with a resolvable duration; check --videos")

    records = []
    for it in items:
        gs, ge = parse_window(it["answer"])
        records.append({"item_index": it["item_index"],
                        "video_id": it["video_id"],
                        "dur": durations[it["video_id"]],
                        "gt": (gs, ge)})

    vlm = {}
    if args.vlm_preds:
        with open(args.vlm_preds) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                try:
                    vlm[r["item_index"]] = parse_window(r["prediction"])
                except Exception:
                    continue          # unparseable model output -> treated as absent
        print(f"[info] {len(vlm)} parseable VLM predictions loaded", file=sys.stderr)

    # video-grouped folds (PSI temporal is 1 item per video, but group anyway
    # so the protocol stays correct if that ever changes)
    videos = sorted({r["video_id"] for r in records})
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(videos))
    folds = [set(videos[i] for i in order[k::args.folds]) for k in range(args.folds)]

    cols = ["random", "center", "global_mean", "duration_prior"]
    if vlm:
        cols += ["vlm", "vlm_prior_blend"]
    cols += ["shipped_prior_full_fit"]

    rows, fitted = [], []
    for k, held in enumerate(folds):
        tr = [r for r in records if r["video_id"] not in held]
        ho = [r for r in records if r["video_id"] in held]
        tr_d = [r["dur"] for r in tr]
        tr_g = [r["gt"] for r in tr]
        ho_d = [r["dur"] for r in ho]
        ho_g = [r["gt"] for r in ho]

        p_center = fit_center(tr_d, tr_g)
        p_mean = fit_global_mean(tr_d, tr_g)
        p_prior = fit_duration_prior(tr_d, tr_g)

        rnd_rng = np.random.RandomState(args.seed * 1000 + k)
        rnd = float(np.mean([miou(random_windows(ho_d, rnd_rng), ho_g)
                             for _ in range(args.random_draws)]))

        prior_ho = apply_duration_prior(ho_d, p_prior)
        row = {
            "split": k,
            "train_videos": len(tr),
            "test_videos": len(ho),
            "random": rnd,
            "center": miou(ratio_windows(ho_d, (1 - p_center["width_ratio"]) / 2,
                                         (1 + p_center["width_ratio"]) / 2), ho_g),
            "global_mean": miou(ratio_windows(ho_d, p_mean["lo"], p_mean["hi"]), ho_g),
            "duration_prior": miou(prior_ho, ho_g),
            "shipped_prior_full_fit": miou(apply_shipped(ho_d), ho_g),
        }
        if vlm:
            paired = [(vlm[r["item_index"]], r["gt"], pr)
                      for r, pr in zip(ho, prior_ho) if r["item_index"] in vlm]
            if paired:
                row["vlm"] = miou([p[0] for p in paired], [p[1] for p in paired])
                blend = [((p[0][0] + p[2][0]) / 2, (p[0][1] + p[2][1]) / 2)
                         for p in paired]
                row["vlm_prior_blend"] = miou(blend, [p[1] for p in paired])
                row["vlm_n"] = len(paired)
            else:
                row["vlm"] = row["vlm_prior_blend"] = float("nan")
                row["vlm_n"] = 0
        rows.append(row)
        fitted.append({"split": k, "center": p_center, "global_mean": p_mean,
                       "duration_prior": p_prior})

    summary = {}
    for c in cols:
        vals = [r[c] for r in rows if not (isinstance(r[c], float) and np.isnan(r[c]))]
        summary[c] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "train_videos", "test_videos"] + cols)
        for r in rows:
            w.writerow([r["split"], r["train_videos"], r["test_videos"]]
                       + [f"{r[c]:.4f}" for c in cols])
        w.writerow(["mean", "", ""] + [f"{summary[c]['mean']:.4f}" for c in cols])
        w.writerow(["std", "", ""] + [f"{summary[c]['std']:.4f}" for c in cols])

    json.dump({
        "protocol": {
            "splits": args.folds,
            "grouping": "video_id (video-grouped, no video spans train and test)",
            "seed": args.seed,
            "random_draws_per_split": args.random_draws,
            "objective": "maximize mean IoU on the split's training half",
            "search_space": {
                "ratio_grid": [float(RATIO_GRID[0]), float(RATIO_GRID[-1]), 0.02],
                "threshold_grid_s": THRESHOLD_GRID,
                "center_width_grid": [float(WIDTH_GRID[0]), float(WIDTH_GRID[-1]), 0.01],
            },
            "refit_per_split": True,
            "note": ("random / center / global_mean / duration_prior are refitted "
                     "on each split's training half only. "
                     "shipped_prior_full_fit replays constants fitted once on all "
                     "training windows and is therefore NOT a held-out number."),
        },
        "n_items": len(records),
        "n_videos": len(videos),
        "missing_videos": len(missing),
        "vlm_predictions_loaded": len(vlm),
        "fitted_params_per_split": fitted,
        "per_split": rows,
        "summary": summary,
    }, open(args.out_json, "w"), indent=1)

    width = max(len(c) for c in cols)
    print(f"{len(records)} items / {len(videos)} videos, {args.folds} video-grouped splits")
    for c in cols:
        flag = "  (NOT held out)" if c == "shipped_prior_full_fit" else ""
        print(f"  {c:<{width}}  mIoU {summary[c]['mean']:.4f} "
              f"± {summary[c]['std']:.4f}{flag}")
    print(f"wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
