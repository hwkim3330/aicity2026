"""Sanity-check answer.txt against the expected submission format:
one line per query (same order as query_index.txt), top-10 unique gallery
IDs that actually exist in the gallery.

Usage:
    python3 validate_submission.py --answer answer.txt
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", default="answer.txt")
    ap.add_argument("--query-index", default="data/query_index.txt")
    ap.add_argument("--gallery-dir", default="data/gallery")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    with open(args.query_index) as f:
        queries = [line.strip() for line in f if line.strip()]

    gallery_ids = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(args.gallery_dir, "*.jpg"))
    }

    with open(args.answer) as f:
        lines = [line.strip() for line in f if line.strip()]

    errors = []
    if len(lines) != len(queries):
        errors.append(f"line count {len(lines)} != query count {len(queries)}")

    for i, line in enumerate(lines):
        ids = line.split()
        if len(ids) != args.topk:
            errors.append(f"line {i}: {len(ids)} ids, expected {args.topk}")
        if len(set(ids)) != len(ids):
            errors.append(f"line {i}: duplicate ids")
        unknown = [g for g in ids if g not in gallery_ids]
        if unknown:
            errors.append(f"line {i}: unknown gallery ids {unknown[:3]}")

    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors[:20]:
            print(" -", e)
        raise SystemExit(1)

    print(f"OK: {len(lines)} rows x top-{args.topk}, all ids valid")


if __name__ == "__main__":
    main()
