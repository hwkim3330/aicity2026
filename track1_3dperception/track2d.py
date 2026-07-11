"""
track2d.py -- simple self-contained per-camera 2D multi-object tracker
(IoU association via `lap` Hungarian matching + a lightweight constant-velocity
Kalman filter per track). ByteTrack-style two-stage association (high-conf then
low-conf) is used to reduce ID switches, but this is a minimal baseline, not a
faithful ByteTrack re-implementation.

Input: detect.py's cache JSON (cache/detections/<scene>__<camera>.json)
Output: cache/tracks2d/<scene>__<camera>.json
{
  "camera": ..., "scene": ...,
  "frames": {"<frame_idx>": [{"track_id":int, "bbox":[x1,y1,x2,y2], "conf":float,
                               "target_class": str}, ...]}
}
"""
import argparse
import json
import os

import numpy as np
import lap


def iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = np.array(boxes_a)
    b = np.array(boxes_b)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 1e-6, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 1e-6, None)
    union = area_a + area_b - inter
    return inter / union


class KalmanBox:
    """Constant-velocity Kalman filter over [cx, cy, w, h, vx, vy]."""

    def __init__(self, bbox):
        cx, cy, w, h = self._to_cwh(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0])
        self.P = np.eye(6) * 10.0
        self.Q = np.eye(6) * 1.0
        self.R = np.eye(4) * 5.0
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=float)

    @staticmethod
    def _to_cwh(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    @staticmethod
    def _to_xyxy(cx, cy, w, h):
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_xyxy(*self.x[:4])

    def update(self, bbox):
        z = np.array(self._to_cwh(bbox))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_bbox(self):
        return self._to_xyxy(*self.x[:4])


class Track:
    _next_id = 1

    def __init__(self, bbox, cls_name, conf):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanBox(bbox)
        self.cls_name = cls_name
        self.conf = conf
        self.age = 0
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return bbox

    def update(self, bbox, cls_name, conf):
        self.kf.update(bbox)
        self.cls_name = cls_name
        self.conf = conf
        self.time_since_update = 0
        self.hits += 1


def greedy_or_hungarian_match(cost, thresh):
    """Solve assignment via lap.lapjv on a cost matrix (1-iou), reject cost>=thresh."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    cost_ = cost.copy()
    cost_[cost_ >= thresh] = thresh + 1.0
    _, x, y = lap.lapjv(cost_, extend_cost=True, cost_limit=thresh)
    matches = []
    unmatched_a = []
    unmatched_b = []
    for i, j in enumerate(x):
        if j >= 0:
            matches.append((i, j))
        else:
            unmatched_a.append(i)
    for j, i in enumerate(y):
        if i < 0:
            unmatched_b.append(j)
    return matches, unmatched_a, unmatched_b


def track_camera(scene, camera, det_path, out_dir="cache/tracks2d",
                  iou_thresh=0.15, max_age=90, min_hits=1, conf_high=0.4):
    with open(det_path) as f:
        data = json.load(f)

    frame_keys = sorted(data["frames"].keys(), key=lambda k: int(k))
    tracks = []  # active tracks
    out_frames = {}

    for fk in frame_keys:
        dets = data["frames"][fk]
        # predict step
        pred_boxes = [t.predict() for t in tracks]

        # ByteTrack-style two-stage association -- the docstring claimed this
        # was already implemented but the actual matching below used to be a
        # single pass over every detection regardless of confidence. Splitting
        # it matters: average track length in this data was ~35 frames
        # (~1.2s), and low-confidence detections during partial occlusion
        # were being ignored entirely instead of used to keep a track alive,
        # so a track died and respawned with a new id every time detection
        # confidence dipped, not just during full occlusion gaps.
        high_idx = [i for i, d in enumerate(dets) if d["conf"] >= conf_high]
        low_idx = [i for i, d in enumerate(dets) if d["conf"] < conf_high]
        high_boxes = [dets[i]["bbox"] for i in high_idx]
        low_boxes = [dets[i]["bbox"] for i in low_idx]

        # stage 1: high-confidence detections vs all active tracks
        cost1 = 1.0 - iou_matrix(pred_boxes, high_boxes)
        matches1, unmatched_tracks1, unmatched_high = greedy_or_hungarian_match(cost1, 1.0 - iou_thresh)
        for ti, hi in matches1:
            di = high_idx[hi]
            tracks[ti].update(dets[di]["bbox"], dets[di]["target_class"], dets[di]["conf"])

        # stage 2: low-confidence detections vs tracks still unmatched after
        # stage 1 (a looser threshold, matching ByteTrack's own design --
        # low-confidence boxes are noisier, so demand less IoU to accept a
        # match, but never let them spawn a brand new track).
        remaining_tracks = [tracks[ti] for ti in unmatched_tracks1]
        remaining_pred_boxes = [pred_boxes[ti] for ti in unmatched_tracks1]
        cost2 = 1.0 - iou_matrix(remaining_pred_boxes, low_boxes)
        matches2, unmatched_tracks2, _ = greedy_or_hungarian_match(cost2, 1.0 - iou_thresh * 0.5)
        for ri, li in matches2:
            ti = unmatched_tracks1[ri]
            di = low_idx[li]
            tracks[ti].update(dets[di]["bbox"], dets[di]["target_class"], dets[di]["conf"])

        # spawn new tracks only from unmatched HIGH-confidence detections --
        # spawning from low-confidence ones just seeds noisy, short-lived
        # tracks that inflate the id count without representing real objects.
        for hi in unmatched_high:
            di = high_idx[hi]
            tracks.append(Track(dets[di]["bbox"], dets[di]["target_class"], dets[di]["conf"]))

        # drop stale tracks
        tracks = [t for t in tracks if t.time_since_update <= max_age]

        frame_out = []
        for t in tracks:
            if t.hits >= min_hits and t.time_since_update == 0:
                frame_out.append({
                    "track_id": t.id,
                    "bbox": [round(float(v), 2) for v in t.kf.get_bbox()],
                    "conf": round(float(t.conf), 4),
                    "target_class": t.cls_name,
                })
        out_frames[fk] = frame_out

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}__{camera}.json")
    with open(out_path, "w") as f:
        json.dump({"camera": camera, "scene": scene, "frames": out_frames}, f)
    n_tracks = len({d["track_id"] for fr in out_frames.values() for d in fr})
    print(f"[track2d] {scene}/{camera}: {len(frame_keys)} frames, {n_tracks} unique tracks")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--det-dir", default="cache/detections")
    ap.add_argument("--out-dir", default="cache/tracks2d")
    ap.add_argument("--iou-thresh", type=float, default=0.15)
    ap.add_argument("--max-age", type=int, default=90)
    args = ap.parse_args()

    if args.cameras is None:
        # infer from det-dir contents for this scene
        cams = []
        for fn in os.listdir(args.det_dir):
            if fn.startswith(args.scene + "__") and fn.endswith(".json"):
                cams.append(fn[len(args.scene) + 2:-5])
        args.cameras = sorted(cams)

    for cam in args.cameras:
        det_path = os.path.join(args.det_dir, f"{args.scene}__{cam}.json")
        track_camera(args.scene, cam, det_path, args.out_dir, args.iou_thresh, args.max_age)


if __name__ == "__main__":
    main()
