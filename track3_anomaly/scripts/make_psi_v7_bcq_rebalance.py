#!/usr/bin/env python3
"""Write psi_vqa_submission_v7.csv = v6 with vote-margin-based BCQ
Yes->No rebalancing applied. All non-flipped bytes stay identical to v6
(line-level substitution; no CSV re-serialization, since open_qa rows
contain multi-line quoted fields that a rewrite could re-quote).

Flip rule (validated on train via bcq_rebalance_cv.py): among the test
items whose SHIPPED v6 prediction is Yes, order by fresh 5-sample
yes_count ascending (tie-break: greedy sample == No first, then
item_index for determinism), flip the weakest N_FLIP to No.
"""
import json
import sys

V6 = "../submissions/psi_vqa_submission_v6.csv"
V7 = "../submissions/psi_vqa_submission_v7.csv"
TEST_VOTES = "../submissions/bcq_votes_test.jsonl"
BCQ_Q = "../data/psi_vqa/test_public/bcq_questions.json"

N_FLIP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    bcq_ids = {it["item_index"] for it in json.load(open(BCQ_Q))["items"]}

    votes = {}
    with open(TEST_VOTES) as f:
        for line in f:
            r = json.loads(line)
            vs = [v for v in r["votes"] if v in ("Yes", "No")]
            votes[r["item_index"]] = {
                "yes_count": sum(v == "Yes" for v in vs),
                "greedy": r["votes"][0] if r["votes"] else None,
            }
    assert bcq_ids <= set(votes), "missing test vote tallies"

    raw = open(V6, "rb").read().decode("utf-8")
    lines = raw.split("\n")

    shipped = {}
    for ln in lines:
        parts = ln.split(",")
        if len(parts) == 2 and parts[0] in bcq_ids:
            shipped[parts[0]] = parts[1]
    assert len(shipped) == 55, f"found {len(shipped)} BCQ rows, expected 55"

    yes_ids = [i for i, p in shipped.items() if p == "Yes"]
    yes_ids.sort(key=lambda i: (votes[i]["yes_count"],
                                0 if votes[i]["greedy"] == "No" else 1,
                                i))
    flip = set(yes_ids[:N_FLIP])
    print(f"shipped Yes={len(yes_ids)} No={55-len(yes_ids)}; flipping {len(flip)}:")
    for i in yes_ids[:N_FLIP]:
        print(f"  {i} yes_count={votes[i]['yes_count']} greedy={votes[i]['greedy']}")
    new_yes = len(yes_ids) - len(flip)
    print(f"new balance: Yes={new_yes} No={55-new_yes} "
          f"(Yes rate {new_yes/55:.3f})")

    out_lines = []
    n_flipped = 0
    for ln in lines:
        if ln.split(",")[0] in flip and ln == ln.split(",")[0] + ",Yes":
            out_lines.append(ln.split(",")[0] + ",No")
            n_flipped += 1
        else:
            out_lines.append(ln)
    assert n_flipped == len(flip), (n_flipped, len(flip))
    open(V7, "wb").write("\n".join(out_lines).encode("utf-8"))
    print(f"wrote {V7} ({n_flipped} lines changed)")


if __name__ == "__main__":
    main()
