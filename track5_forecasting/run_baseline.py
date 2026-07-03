"""Smoke-test the full pipeline: dummy clip -> freeze-frame baseline -> submission files."""
from baseline_forecast import forecast
from make_dummy_sample import main as make_dummy_sample
from write_submission import write_frames


def main():
    clip = make_dummy_sample()
    frames = forecast(clip)
    out_dir = write_frames(clip.clip_id, frames)
    print(f"wrote {len(frames)} frames to {out_dir}")


if __name__ == "__main__":
    main()
