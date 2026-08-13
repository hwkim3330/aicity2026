"""Pin every source of run-to-run variation the inference paths expose.

The official 2026-07-10 submissions were produced without any of this. The
greedy paths (Track 2 captioning, Track 7 FETV) were deterministic in their
decode policy but still float over kernel reduction order; the Track 3 TAR and
Track 8 PSI BCQ/MCQ paths draw five unseeded samples at temperature 0.7 and
majority-vote, so they differed on every run. `REPRODUCE.md` records this.

Calling `pin()` cannot make those official numbers reproducible after the fact
-- the seeds that produced them were never recorded and are unrecoverable. What
it does is make every *subsequent* run reproducible, so a reviewer re-running
the pipeline twice gets identical bytes and any difference they see is a real
difference rather than sampling noise.

Two levels, because they cost different amounts:

  pin(seed)                   seeds python/numpy/torch and turns off the cuDNN
                              autotuner. Cheap. Enough for the sampled voting
                              paths, which is where the actual variance was.

  pin(seed, strict=True)      additionally demands bit-reproducible kernels.
                              Slower, and raises if any op in the graph has no
                              deterministic implementation.

`strict` is off by default deliberately. Turning it on changes which kernels run
and therefore the numbers themselves, so making it the default would silently
diverge from what the repository's recorded results were produced with.
"""

from __future__ import annotations

import os
import random

SEED_ENV = "AICITY_SEED"
DEFAULT_SEED = 1234


def resolve_seed(explicit: int | None = None) -> int:
    """Explicit argument beats $AICITY_SEED beats the default."""
    if explicit is not None:
        return int(explicit)
    return int(os.environ.get(SEED_ENV, DEFAULT_SEED))


def pin(seed: int | None = None, strict: bool = False, verbose: bool = True) -> int:
    """Seed every RNG in play and fix the cuDNN/cuBLAS knobs. Returns the seed.

    `AICITY_NO_PIN=1` skips everything and leaves torch at its defaults. That
    exists to test whether pinning is itself what makes a run differ from the
    2026-07-10 artifacts, which were produced with no pinning at all — asserting
    it does not would be guessing.
    """
    seed = resolve_seed(seed)

    if os.environ.get("AICITY_NO_PIN") == "1":
        if verbose:
            import sys
            print("[determinism] AICITY_NO_PIN=1 — torch defaults, nothing pinned",
                  file=sys.stderr, flush=True)
        return seed

    # Set before torch touches CUDA: cuBLAS reads this when it creates its
    # workspace, and a later change is ignored for the rest of the process.
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # benchmark=True lets cuDNN pick an algorithm per input shape by timing it,
    # so the same input can take different kernels on different runs. The clips
    # here vary in resolution and frame count, which is exactly the case where
    # the autotuner reshuffles.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    if verbose:
        import sys
        print(f"[determinism] seed={seed} cudnn.deterministic=True "
              f"cudnn.benchmark=False strict={strict}", file=sys.stderr, flush=True)
    return seed


def add_args(ap) -> None:
    """Attach --seed/--strict-determinism to an argparse parser."""
    ap.add_argument("--seed", type=int, default=None,
                    help=f"RNG seed for sampled decoding (default ${SEED_ENV} or {DEFAULT_SEED})")
    ap.add_argument("--strict-determinism", action="store_true",
                    help="also demand bit-reproducible CUDA kernels; slower, and "
                         "changes results relative to the recorded runs")
