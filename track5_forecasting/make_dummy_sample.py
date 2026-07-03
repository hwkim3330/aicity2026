"""Generate one fake ClipRequest so the pipeline is testable before real data arrives."""
import os

from PIL import Image

from schema import ClipRequest

OUT_DIR = "dummy_data"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init_path = os.path.join(OUT_DIR, "init_frame.png")
    Image.new("RGB", (640, 360), color=(100, 140, 180)).save(init_path)

    clip = ClipRequest(
        clip_id="dummy_clip_0",
        caption_1="A car approaches a pedestrian crossing at moderate speed.",
        caption_2="The pedestrian steps into the crosswalk as the car brakes.",
        init_frame_path=init_path,
        num_frames=16,
        width=640,
        height=360,
    )
    print(clip)
    return clip


if __name__ == "__main__":
    main()
