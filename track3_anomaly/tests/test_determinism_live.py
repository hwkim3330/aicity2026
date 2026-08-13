"""Live check that seeding actually reaches the sampler.

Loads the real backbone, so it needs a GPU and the weights -- this is not a unit
test. Run it after changing anything in the decode path:

    python3 track3_anomaly/tests/test_determinism_live.py <any.mp4>

Any clip works; the point is the RNG, not the content. Generate one with:

    ffmpeg -f lavfi -i "testsrc2=size=640x360:rate=10:duration=4" -pix_fmt yuv420p clip.mp4

Exits 0 on PASS. Recorded result, RTX 3090, Qwen3-VL-8B-Instruct @ 0c351dd:
same seed twice gave sha 5f6bf78990911223 both times, seed 4321 gave
de206dfcca17a06f.

Exercises the one thing that was broken: `bcq`, the sampled 5-vote path. Same
seed twice must give identical votes; a different seed must give different ones,
otherwise the seed is not reaching the sampler and the test proves nothing.
"""
import hashlib
import os
import sys

REPO = "/home/kim/aicity2026"
sys.path.insert(0, os.path.join(REPO, "track3_anomaly", "scripts"))
sys.path.insert(0, os.path.join(REPO, "shared", "scripts"))

CLIP = sys.argv[1]
QUESTION = "Does a collision occur in the video?\nAnswer with only Yes or No."

import inference  # noqa: E402

print(f"MODEL_ID       = {inference.MODEL_ID}")
print(f"MODEL_REVISION = {inference.MODEL_REVISION}")
sys.stdout.flush()

backend = inference.QwenVLBackend(quant="bf16", verbose=True)
print(f"seed in use    = {backend.seed}")

# The vote is a majority over 5 draws, so a single letter could match by luck.
# Capture every raw generation instead and hash the whole sequence.
raw = []
orig = backend._generate_once


def spy(inputs, mnt, do_sample, seed=None):
    out = orig(inputs, mnt, do_sample, seed=seed)
    raw.append((do_sample, seed, out))
    return out


backend._generate_once = spy


def run(label, seed):
    raw.clear()
    backend.seed = seed
    ans = backend.answer(CLIP, "bcq", QUESTION, samples=5)
    blob = "\n".join(f"{d}|{s}|{t}" for d, s, t in raw)
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    print(f"\n[{label}] seed={seed} answer={ans!r} draws={len(raw)} sha={h}")
    for d, s, t in raw:
        print(f"    sample do_sample={d} seed={s} -> {t[:64]!r}")
    return h, ans


a, ans_a = run("run A", 1234)
b, ans_b = run("run B", 1234)
c, ans_c = run("run C", 4321)

print("\n" + "=" * 62)
print(f"same seed reproduces : {a == b}   ({a} vs {b})")
print(f"different seed differs: {a != c}   ({a} vs {c})")
ok = (a == b) and (a != c)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
