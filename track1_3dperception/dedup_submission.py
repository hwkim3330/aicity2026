"""dedup_submission.py -- per-frame 3D spatial NMS over a track1 submission file.

The distance-based MTMC fusion under-merges: the same physical object seen by
many cameras often survives as several global tracks, so a single person can
emit 5-20 near-identical lines per frame. That both bloats the file past the
50MB submission cap and directly hurts DetA (every duplicate is a false
positive against a single GT box).

Greedy per-(scene, frame, class) clustering: sort candidates by their global
track's total length (longest = most stable first), keep a candidate only if
it is farther than --radius meters from every already-kept candidate of the
same class in that frame.

Usage:
    python3 dedup_submission.py --inp track1_test.txt --out track1_test_dedup.txt --radius 0.75
"""
import argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.75)
    args = ap.parse_args()

    track_len = defaultdict(int)
    with open(args.inp) as f:
        for line in f:
            p = line.split()
            track_len[(p[0], p[2])] += 1

    by_frame = defaultdict(list)  # (scene, frame) -> [(class, obj, x, y, line)]
    order = []
    with open(args.inp) as f:
        for line in f:
            p = line.split()
            scene, cls, obj, frame = p[0], p[1], p[2], p[3]
            x, y = float(p[4]), float(p[5])
            key = (scene, frame)
            if key not in by_frame:
                order.append(key)
            by_frame[key].append((cls, obj, x, y, line))

    r2 = args.radius ** 2
    kept_lines = []
    n_in = n_kept = 0
    for key in order:
        scene = key[0]
        cands = by_frame[key]
        n_in += len(cands)
        cands.sort(key=lambda c: -track_len[(scene, c[1])])
        kept = defaultdict(list)  # class -> [(x, y)]
        for cls, obj, x, y, line in cands:
            if any((x - kx) ** 2 + (y - ky) ** 2 < r2 for kx, ky in kept[cls]):
                continue
            kept[cls].append((x, y))
            kept_lines.append(line)
            n_kept += 1

    with open(args.out, "w") as f:
        f.writelines(kept_lines)
    print(f"{n_in} -> {n_kept} lines ({100 * n_kept / max(n_in, 1):.1f}% kept) -> {args.out}")


if __name__ == "__main__":
    main()
