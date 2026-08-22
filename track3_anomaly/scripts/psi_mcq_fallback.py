#!/usr/bin/env python3
"""Recover an answer from a truncated chain of thought instead of guessing "A".

`inference.py` parses the letter three ways -- a `final answer:` marker, an
`answer is|option X` phrase, and a line starting `A)` -- and when all three miss,
`make_submission.py` writes the literal string "A". Measured on 321 labelled
train MCQ items at the shipped 16-frame budget:

    parse failures            38 / 321  = 11.8%
    all 38 cut mid-sentence   38 / 38
    "A" happens to be right   10 / 38   = 26.3%   (chance is 25%)

So one item in eight is decided by a coin flip, and the coin is not even paying
for itself. The generations are not empty or refusing -- they are the model
reasoning past its token budget, and the reasoning is on topic. One of them loops
("not moving toward the camera. not moving away from the camera.") but most read
as a description that has already eliminated options.

This scores each option against the surviving text instead. The prompt asks the
model to describe the pedestrian and then eliminate options, so the text contains
both the description and, usually, explicit negations of the wrong options. Two
signals fall out of that:

  overlap   content-word F1 between the option and the text, which rewards an
            option the description actually supports
  negation  whether the option's distinctive words appear inside an explicit
            "the pedestrian is not ..." clause, which is the model telling us to
            drop that option even though it never reached a final letter

Deliberately not a model call: this reads text the pipeline already paid for, so
it costs nothing at inference and cannot change any item the three existing
parsers already resolve.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "and", "or", "but", "not", "no", "they", "their",
    "them", "it", "its", "this", "that", "these", "those", "with", "for",
    "from", "by", "as", "into", "toward", "towards", "pedestrian", "video",
    "clip", "camera", "option", "answer", "because", "which", "who", "may",
    "might", "would", "could", "does", "do", "did", "has", "have", "had",
}
NEG = re.compile(
    r"(?:is|are|was|were|does|do|is not|are not)\s+not\s+([^.;]{0,120})|"
    r"\bnot\s+(?:described as|shown|depicted)\s+([^.;]{0,120})",
    re.IGNORECASE)


def content(text: str) -> Counter:
    return Counter(w for w in re.findall(r"[a-z]+", text.lower())
                   if w not in STOP and len(w) > 2)


def f1(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    ov = sum((a & b).values())
    if not ov:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


def negated_spans(text: str) -> Counter:
    out: Counter = Counter()
    for m in NEG.finditer(text):
        span = m.group(1) or m.group(2) or ""
        out.update(content(span))
    return out


def choose(text: str, options: dict[str, str], neg_weight: float = 1.0
           ) -> tuple[str, dict[str, float]]:
    """Best option letter given the surviving generation."""
    body = content(text)
    neg = negated_spans(text)
    scores = {}
    for letter, opt in options.items():
        c = content(opt)
        support = f1(c, body)
        # words of this option that the text explicitly negated, as a fraction
        # of the option's own content
        denied = sum((c & neg).values()) / max(sum(c.values()), 1)
        scores[letter] = support - neg_weight * denied
    best = max(scores, key=lambda k: (scores[k], k))
    return best, scores


OPT = re.compile(r"^\s*\(?([A-D])\)?[.):]\s*(.+?)\s*$", re.MULTILINE)


def options_from_question(question: str) -> dict[str, str]:
    found = dict(OPT.findall(question))
    return {k: v for k, v in found.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="CV output with raw generations and gt")
    ap.add_argument("--neg-weight", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl)]
    unresolved = [r for r in rows if not r.get("pred")]
    print(f"{len(rows)} items, {len(unresolved)} unresolved by the shipped parsers")

    hit_a = sum(1 for r in unresolved if r["gt"] == "A")
    ok = 0
    no_opts = 0
    for r in unresolved:
        opts = options_from_question(r["question"])
        if len(opts) < 2:
            no_opts += 1
            continue
        pick, _ = choose(r.get("raw") or "", opts, args.neg_weight)
        ok += pick == r["gt"]

    n = len(unresolved)
    base = sum(1 for r in rows if r.get("ok"))
    print(f"  options not parseable from the question: {no_opts}")
    print(f'  fallback "A"      {hit_a}/{n} = {hit_a / n:.3f}')
    print(f"  text-based choice {ok}/{n} = {ok / n:.3f}")
    print(f"\n  overall {base}/{len(rows)} = {base / len(rows):.4f}"
          f"  ->  {(base - hit_a + ok)}/{len(rows)} = "
          f"{(base - hit_a + ok) / len(rows):.4f}")


if __name__ == "__main__":
    main()
