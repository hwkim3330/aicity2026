# Runtime and memory

The paper claims the system runs on a single 24 GB consumer GPU but reports no
runtime or memory figure. These are measured numbers.

**Official-run runtime: not recorded.** None of the challenge-period runs logged
wall-clock time or peak memory. Everything below is a **reproduction-run
measurement**, taken on 2026-08-03 at commit `6e4507b` on the hardware in the
table. Reproduction runs are the same code and configuration as the official
runs, but they are not the official runs.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB, driver 580.173.02 |
| CPU / OS | Linux 6.8 (Ubuntu 24.04), Python 3.12.3 |
| Framework | PyTorch 2.10.0+cu128, Transformers 5.13.0 |
| GPU contention | none — the GPU was otherwise idle (≈0.9 GB baseline) |

## Measured

| Workload | Model | Precision | Items | Model load | Inference wall | sec/item | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|
| PSI temporal, training split | Qwen3-VL-8B-Instruct | bf16 | 227 | 18.7 s | 469.3 s | 2.07 | 16.78 GiB |
| PSI temporal, 3-item smoke | Qwen3-VL-8B-Instruct | bf16 | 3 | 28.7 s | 8.1 s | 2.69 | 16.78 GiB |
| Temporal-prior CV (all baselines) | none — CPU only | — | 227 | — | 23.9 s | 0.11 | 0 |

Peak VRAM is `torch.cuda.max_memory_allocated()`. Of the 16.78 GiB peak,
16.33 GiB is reached at model load; video decode and generation add ≈0.45 GiB.
Both PSI rows report the same peak because it is dominated by the weights.

Sources: `track3_anomaly/results/psi_temporal_vlm_train_stats.json`.

## Estimates for the full official runs

Not measured directly. Scaling the measured 2.07 s/item is only valid for
single-call greedy tasks with comparable clip lengths and token budgets:

| Workload | Items | Basis | Estimate |
|---|---:|---|---|
| FETV v11 | 200 | one structured JSON call per clip, larger token budget than temporal | ≳ 7 min plus load |
| PSI-VQA full | 328 | 55 BCQ items cost ≈5× (self-consistency); rest single-call | ≳ 15 min plus load |
| TAR full | 960 | 80 clips × ~12 questions; earlier smoke test measured ≈4.3 s/item at 4-bit with two other GPU processes resident | ≈ 60–80 min |

The TAR figure is the one number carried over from a challenge-period log
(`track3_anomaly/README.md`, smoke test of 36 items in 155 s). It was measured
under GPU contention and at 4-bit, so it is not comparable to the bf16 rows
above.

## Reproducing these measurements

```bash
cd track3_anomaly/analysis
python3 psi_temporal_vlm_eval.py --limit 0   # writes ../results/psi_temporal_vlm_train_stats.json
```

The script records model-load time, per-item latency, wall clock, and
`torch.cuda.max_memory_allocated()` directly. For an external cross-check:

```bash
/usr/bin/time -v python3 psi_temporal_vlm_eval.py --limit 20
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```
