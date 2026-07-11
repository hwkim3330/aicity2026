# Track 6 — logdump run progress log

## 2026-07-11 05:20-05:40 KST (agent session start)
Deadline: 2026-07-11 20:59 KST. ~15.5h runway at start.

### Orientation findings
- `logdump_client.py selftest` PASSED offline (27 chunks, md5 verified, byte-exact round trip).
- Found + fixed a real bug in `scripts/logdump_client.py try_reassemble()`: crashed with
  `AttributeError: 'list' object has no attribute 'get'` when the dumped payload is a raw
  JSON *list* (i.e. a plain submission.json dumped via `logdump.py <file>` standalone path)
  instead of the compact dict format. Now guarded with isinstance check.
  Verified: standalone file dump at chunk_chars=900 round-trips byte-identical (cmp OK).
- `hafnia experiment ls` confirms `rfdetr-main-v4` (69dd1a4f-9128-4e39-9977-a26b6345ea71)
  EXPERIMENT_SUCCEEDED. Full detail via REST:
  - command: `python scripts/train_rfdetr.py --epochs 20 --resolution 560 --model-size base
    --batch-size 2 --name rfdetr_main && python scripts/benchmark.py --weights
    runs/rfdetr_main/checkpoint_best_total.pth --model-type rfdetr --rfdetr-model-size base
    --rfdetr-resolution 560 --split test --out submission.json`
    => the chained `train && benchmark --model-type rfdetr` pattern is PROVEN in-cloud.
  - Lite tier, 21.46 h total, 1765 credits. dataset_recipe=6456e182-1cf2-4939-925c-6d8263c762ee,
    environment id f4b57178-2a9a-46b1-874e-80f9d9654657.
- Fetched ALL of v4's logs (runs/v4_logs.json, 8893 entries): **pagination past the 1000-entry
  cap WORKS** via `before=<oldest created_at>` time-window walking (logdump_client fetch does
  this automatically). Complete coverage from build start to "MLflow run ended".
  - v4 had NO @HAFDUMP@ lines (logdump.py written 07-10 17:34, after v4's package) — no shortcut,
    fresh run required.
  - Benchmark stats from v4 logs: test split = 14,925 images; 248,660 detections @ conf 0.15;
    benchmark inference took ~72 min on Lite T4.
  - Training: ~60.7 min/epoch (incl. per-epoch val) on Lite, batch 2, res 560.
  - Val EMA mAP curve: ep0 0.518, ep3 0.614, ep5 0.634, ep7 0.650, ep10 0.661, ep12 0.668,
    ep15 0.681, ep18 0.691 (best, became checkpoint_best_total). Diminishing returns ~ep10+.
  - Max observed log line length: 408 chars (no evidence either way on long-line truncation —
    hence dual-chunk-size dump strategy + inline probe below).

### Plan (decided ~05:40 KST)
- 10 epochs (not 20): 10×61min ≈ 10.2h train + ~25min build + 72min benchmark + dumps
  → done ~17:45 KST, leaves ~3h margin for fetch/verify/portal upload. 12 epochs would leave
  <1.5h margin — too tight. Expected val mAP ~0.66 (vs v4's 0.691).
- Single experiment (1-concurrent-max makes a separate probe run cost 45 min of train time);
  instead chain log_probe INLINE at the start (`|| true` so probe failure can't kill the run)
  and dump the submission TWICE at the end: compact payload @ chunk 3800 (safe if lines up to
  ~4KB survive) AND raw submission.json @ chunk 900 (safe under a 1024-char truncation limit).
  Pagination is proven, so chunk COUNT is not a constraint; line truncation is the only risk.
- Probe results are fetchable ~15 min after training starts → early abort possible if even
  900-char lines get mangled (has never been observed; 408-char lines were intact).

## 2026-07-11 18:47 KST — SUCCESS
Main experiment (92290564-795b-4e82-8936-919b6a134174, rfdetr-main-v5-0946, 7 epochs)
reached EXPERIMENT_SUCCEEDED at 18:46:13 KST. Logs fetched (770 entries via
time-window pagination), reassembled via logdump_client.py: 451/451 chunks
recovered, md5 verified (enc + raw), 275,159 detections across 14,814 images
(of 14,925 test images) reconstructed successfully.

Output: submissions/track6_rfdetr_v5_7ep.json (COCO-style list: image_id,
file_path, category_id [0-9], category_name, bbox [x,y,w,h], score [0.15-0.983]).

This is the FIRST TIME this challenge season that a real trained-model
prediction file has been extracted from Hafnia -- all prior successful
training runs (main-yolo11m-40ep, rfdetr-main-v4) had their checkpoints/
predictions permanently lost to the no-artifact-retrieval problem.

Ready for manual upload to https://eval.aicitychallenge.org/aicity2026/.
