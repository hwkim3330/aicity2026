"""Run the freeze-frame baseline over every real WTS_TRACK5_TEST clip and
write the submission directory."""
import glob
import os

from baseline_forecast import forecast
from schema import load_clip
from write_submission import write_frames

TEST_ROOT = "wts_data/WTS_TRACK5_TEST"
OUT_ROOT = "submission"


def main():
    clip_dirs = sorted(glob.glob(os.path.join(TEST_ROOT, "*")))
    print(f"[run_test_all] {len(clip_dirs)} clips found", file=__import__("sys").stderr)
    n_ok = 0
    n_err = 0
    for clip_dir in clip_dirs:
        if not os.path.isdir(clip_dir):
            continue
        try:
            clip = load_clip(clip_dir)
            frames = forecast(clip)
            out_dir = write_frames(clip.clip_id, frames, out_root=OUT_ROOT)
            n_ok += 1
            print(f"[{clip.clip_id}] {len(clip.input_frame_paths)} in -> "
                  f"{len(frames)} generated -> {out_dir}")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERROR] {clip_dir}: {e}")
    print(f"[run_test_all] done: {n_ok} ok, {n_err} errors")


if __name__ == "__main__":
    main()
