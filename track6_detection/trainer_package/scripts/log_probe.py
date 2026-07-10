#!/usr/bin/env python3
"""
log_probe.py -- CHEAP (~3 min) Hafnia experiment that characterizes the log
pipeline BEFORE committing to another ~21 h train+benchmark run.

Launch this exactly like benchmark.py (same trainer package, tiny/any
dataset, any instance) and then run locally:

    python scripts/logdump_client.py fetch <experiment_id> -o probe_logs.json
    python scripts/logdump_client.py probe-report probe_logs.json

It answers, empirically:
  1. LINE SIZE   -- largest single print() that survives intact (ladder of
                    0.5/1/2/4/6/8/16/32/64/128/250 KB lines, each with
                    length+crc32 so truncation is detected exactly). Printed
                    EARLY and again LATE, on stdout and stderr, so the answer
                    survives whichever end of the stream the entry cap keeps.
  2. ENTRY CAP   -- 1500 numbered burst lines + 300 throttled lines push the
                    total well past 1000 entries, so the fetch step learns
                    whether pagination works and, if not, WHICH ~1000 entries
                    are kept (newest vs oldest) and whether bursts get dropped.
  3. REAL PATH   -- a synthetic ~3000-detection submission is pushed through
                    the ACTUAL logdump.dump_bytes() production code path
                    (tag "probe") and must reassemble locally to the same md5.

Total lines printed: ~1900. Runtime: ~2-3 min + 60 s flush sleep.
"""
import base64
import binascii
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import logdump  # noqa: E402

PMARK = "@HAFPROBE@"
END = "@END@"
SIZES = [500, 1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000, 250000]
BURST_LINES = 1500
THROTTLE_LINES = 300
THROTTLE_SLEEP = 0.025


def crc(s: str) -> str:
    return format(binascii.crc32(s.encode("ascii")) & 0xFFFFFFFF, "08x")


def deterministic_payload(n: int) -> str:
    """Repeating base64 alphabet -- content recoverable by position, so a
    truncated line still tells us exactly how many chars survived."""
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return (alpha * (n // len(alpha) + 1))[:n]


def size_ladder(phase: str):
    for stream_name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        for size in SIZES:
            p = deterministic_payload(size)
            print(f"{PMARK}|SIZE|{phase}|{stream_name}|{size}|{len(p)}:{crc(p)}|{p}|{END}",
                  file=stream, flush=True)
            time.sleep(0.05)


def main():
    from hafnia.experiment import HafniaLogger  # keeps the job from being killed at ~6 min
    logger = HafniaLogger(project_name="track6-log-probe")
    logger.log_configuration({"probe": "log pipeline characterization",
                              "sizes": SIZES, "burst": BURST_LINES})

    t0 = time.time()
    print(f"{PMARK}|START|sizes={SIZES}|burst={BURST_LINES}|throttle={THROTTLE_LINES}|{END}",
          flush=True)

    # 1. size ladder, EARLY (lost if the cap keeps only the newest ~1000)
    size_ladder("EARLY")

    # 2. burst: 1500 rapid small lines -> tests entry cap, ordering, rate drops
    for i in range(1, BURST_LINES + 1):
        filler = deterministic_payload(80)
        print(f"{PMARK}|BURST|{i}/{BURST_LINES}|{filler}|{END}", flush=True)

    # 3. throttled: 300 lines at 25 ms -- does pacing prevent drops?
    for i in range(1, THROTTLE_LINES + 1):
        filler = deterministic_payload(80)
        print(f"{PMARK}|THROT|{i}/{THROTTLE_LINES}|{filler}|{END}", flush=True)
        time.sleep(THROTTLE_SLEEP)

    # 4. size ladder, LATE (kept if the cap keeps the newest entries)
    size_ladder("LATE")

    # 5. production-path rehearsal: synthetic detections through the real
    #    dump_bytes() at the same default settings benchmark.py will use.
    random.seed(7)
    fused, id_of = {}, {}
    for i in range(150):
        stem = f"probeimg{i:04d}" + format(random.getrandbits(64), "016x")
        id_of[stem] = {"file_path": f"data/{stem}.jpg", "sample_index": i}
        fused[stem] = [{
            "category_id": random.randrange(10),
            "bbox": [round(random.uniform(0, 1500), 2), round(random.uniform(0, 900), 2),
                     round(random.uniform(8, 400), 2), round(random.uniform(8, 300), 2)],
            "score": round(random.uniform(0.15, 0.99), 4),
        } for _ in range(20)]
    payload = logdump.compact_payload(fused, id_of,
                                      [f"class_{c}" for c in range(10)])
    n = logdump.dump_bytes(payload, tag="probe",
                           chunk_chars=logdump.DEFAULT_CHUNK_CHARS)
    print(f"{PMARK}|REALDUMP|chunks={n}|raw_bytes={len(payload)}|{END}", flush=True)

    print(f"{PMARK}|DONE|elapsed={time.time() - t0:.1f}s|{END}", flush=True)
    logger.log_metric(name="probe_lines_printed", value=1.0, step=0)
    time.sleep(60)  # let the log collector flush before the container dies
    logger.end_run()


if __name__ == "__main__":
    main()
