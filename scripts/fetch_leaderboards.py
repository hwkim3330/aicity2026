#!/usr/bin/env python3
"""Fetch the final AI City Challenge 2026 leaderboards straight from the
evaluation system and write them to leaderboards/.

The endpoint is public — no login, no cookie:

    https://eval.aicitychallenge.org/aicity2026/submission/leaderboard/stats/{dtype}/{track}

`dtype` matters. The portal keeps two boards per track:

  public   submissions marked Public. This is the board the challenge reports
           and the board our paper's ranks come from.
  general  every submission, including General-type ones. More teams, and our
           rank is different on it.

Reporting a rank without saying which board it came from is meaningless, so
both are stored.

    python3 scripts/fetch_leaderboards.py
    python3 scripts/fetch_leaderboards.py --tracks 7 8 --out-dir /tmp/lb
"""
import argparse
import json
import os
import urllib.request

BASE = "https://eval.aicitychallenge.org/aicity2026/submission/leaderboard/stats"

TRACKS = {
    3: ("track3_tar", "Traffic Anomaly Reasoning (TAR), in-domain"),
    7: ("track7_fetv", "Fisheye Traffic Violation Understanding (FETV), Track 3 OOD Test Set 1"),
    8: ("track8_psi_vqa", "PSI-VQA Pedestrian Situated Intent, Track 3 OOD Test Set 2"),
}
OUR_TEAM_ID = 277


def score_key(row):
    for k in ("final", "final_score", "mean"):
        if k in row:
            return k
    raise KeyError(f"no score key in {sorted(row)}")


def fetch(dtype, track, timeout=30):
    url = f"{BASE}/{dtype}/{track}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", type=int, default=sorted(TRACKS))
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "leaderboards"))
    ap.add_argument("--snapshot-date", default=None,
                    help="ISO date recorded in the output; omit to leave null")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for track in args.tracks:
        slug, desc = TRACKS[track]
        boards = {}
        for dtype in ("public", "general"):
            rows = fetch(dtype, track)
            key = score_key(rows[0])
            ours = next((r for r in rows if r.get("teamId") == OUR_TEAM_ID), None)
            boards[dtype] = {
                "team_count": len(rows),
                "our_rank": ours["rank"] if ours else None,
                "our_score": ours[key] if ours else None,
                "score_key": key,
                "teams": [{"rank": r["rank"], "team_id": r.get("teamId"),
                           "team": r["teamName"], "score": r[key],
                           "components": r.get("score", {}),
                           "is_baseline": r.get("isBaseline", False)}
                          for r in rows],
            }
        doc = {
            "evaluation": desc,
            "portal_track_tab": track,
            "phase": "final full test set",
            "source": f"{BASE}/<dtype>/{track}",
            "exported_from_portal": True,
            "snapshot_date": args.snapshot_date,
            "our_team_id": OUR_TEAM_ID,
            "board_note": ("'public' is the board the challenge reports and the "
                           "one the paper's ranks come from; 'general' includes "
                           "General-type submissions and gives a different rank."),
            "boards": boards,
        }
        path = os.path.join(args.out_dir, f"{slug}_final.json")
        json.dump(doc, open(path, "w"), indent=1)
        pub, gen = boards["public"], boards["general"]
        print(f"track {track}: public rank {pub['our_rank']}/{pub['team_count']}, "
              f"general rank {gen['our_rank']}/{gen['team_count']}  -> {path}")


if __name__ == "__main__":
    main()
