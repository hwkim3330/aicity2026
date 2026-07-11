"""
common.py -- shared utilities for the Track1 MTMC baseline pipeline.

Covers:
  - calibration.json loading + per-camera projection helpers
  - class name <-> class_id mapping (per track1.txt submission spec)
  - scene / camera discovery helpers under data/MTMC_Tracking_2026/{train,test}

Calibration convention (VERIFIED empirically against ground_truth.json, see README.md):
  - `cameraMatrix` is a 3x4 matrix P such that for a world point (X,Y,Z):
        [u,v,w]^T = P @ [X,Y,Z,1]^T ;  pixel = (u/w, v/w)
    i.e. full 3D world -> image homogeneous projection (world->pixel), NOT pixel->world.
  - `homography` is exactly `cameraMatrix` with the Z-column (column index 2) removed,
    i.e. the same projection restricted to the ground plane Z=0:
        [u,v,w]^T = H @ [X,Y,1]^T
    We verified: homography[:, :2] == cameraMatrix[:, [0,1]] and
                 homography[:, 2]  == cameraMatrix[:, 3]  (columns 0,1,3 of P).
  - ground_truth.json "3d location" is the 3D bounding box CENTER (not ground-contact
    point): for a sample PalletTruck, location.z ~= scale.h/2, and projecting the bottom
    of the box (z = location.z - h/2 ~= 0) with cameraMatrix lands very close to the
    bottom-center of the reported 2D bbox, and matches the homography(z=0) projection.
"""
import json
import os
import glob
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MTMC_Tracking_2026")

# ---------------------------------------------------------------------------
# Class mapping (per task spec / track1.txt submission format)
# ---------------------------------------------------------------------------
CLASS_NAME_TO_ID = {
    "Person": 0,
    "Forklift": 1,
    "NovaCarter": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "PalletTruck": 6,
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# Average object footprint sizes (meters), computed from ground_truth.json
# "3d bounding box scale" = [width, length, height]. Filled in via
# scripts/inspect_gt_stats (see README); used as fallback size priors in project3d.py.
# Values below were measured from Warehouse_000..003 ground_truth.json.
CLASS_SIZE_PRIOR = {
    # Person/Forklift/PalletTruck measured directly from ground_truth.json across
    # Warehouse_000-006 (only classes actually present in the downloaded scenes).
    "Person": (0.59, 0.61, 1.84),
    "Forklift": (1.14, 2.06, 2.58),
    "PalletTruck": (0.75, 1.75, 1.62),
    # Not observed in any locally available scene -- placeholders (documented limitation).
    "NovaCarter": (0.6, 0.6, 0.4),
    "Transporter": (0.8, 1.2, 1.2),
    "FourierGR1T2": (0.6, 0.6, 1.7),
    "AgilityDigit": (0.6, 0.6, 1.8),
}

# Weak heuristic mapping from COCO (YOLO pretrained) class names to our target classes.
# COCO has no warehouse-robot classes; only "person" is a reliable proxy.
# Vehicle-ish COCO classes are optionally mapped to Forklift/PalletTruck as a *weak* proxy
# (documented limitation -- these will have poor precision/recall).
COCO_TO_TARGET = {
    "person": "Person",
    "truck": "Forklift",       # weak proxy
    "car": "PalletTruck",      # weak proxy
    "motorcycle": "NovaCarter",  # weak proxy (small ground robot silhouette)
}


# ---------------------------------------------------------------------------
# Scene / camera discovery
# ---------------------------------------------------------------------------
def list_scenes(split="train"):
    """Return sorted list of scene names (e.g. Warehouse_000) under data/<split>/."""
    base = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(base):
        return []
    return sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("Warehouse_")
    )


def scene_dir(scene, split="train"):
    return os.path.join(DATA_ROOT, split, scene)


def list_cameras(scene, split="train"):
    """Return sorted list of camera ids (e.g. Camera_0000) that have both a video file
    and a calibration.json entry for this scene."""
    sdir = scene_dir(scene, split)
    vids = glob.glob(os.path.join(sdir, "videos", "Camera_*.mp4"))
    cam_ids = sorted(os.path.splitext(os.path.basename(v))[0] for v in vids)
    cal = load_calibration(scene, split)
    cal_ids = set(cal.keys())
    return [c for c in cam_ids if c in cal_ids]


def video_path(scene, camera, split="train"):
    return os.path.join(scene_dir(scene, split), "videos", f"{camera}.mp4")


def gt_path(scene, split="train"):
    return os.path.join(scene_dir(scene, split), "ground_truth.json")


def has_ground_truth(scene, split="train"):
    return os.path.isfile(gt_path(scene, split))


def load_ground_truth(scene, split="train"):
    p = gt_path(scene, split)
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def load_calibration(scene, split="train"):
    """Return dict: camera_id -> sensor dict (raw calibration.json entry)."""
    p = os.path.join(scene_dir(scene, split), "calibration.json")
    with open(p) as f:
        cal = json.load(f)
    return {s["id"]: s for s in cal["sensors"] if s.get("type") == "camera"}


class CameraModel:
    """Wraps a single camera's calibration with projection helpers."""

    # A real warehouse floor plan doesn't span more than this many meters in
    # any direction -- used to detect scenes whose raw ground-plane
    # projection needs the scaleFactor/translationToGlobalCoordinates
    # correction (see class docstring below for why this is scene-dependent).
    PLAUSIBLE_COORD_BOUND = 300.0

    def __init__(self, sensor):
        self.id = sensor["id"]
        self.K = np.array(sensor["intrinsicMatrix"], dtype=np.float64)
        self.Rt = np.array(sensor["extrinsicMatrix"], dtype=np.float64)  # 3x4
        self.P = np.array(sensor["cameraMatrix"], dtype=np.float64)      # 3x4, world->pixel
        self.H = np.array(sensor["homography"], dtype=np.float64)        # 3x3, ground(z=0)->pixel
        self.H_inv = np.linalg.inv(self.H)
        attrs = {a["name"]: a["value"] for a in sensor.get("attributes", [])}
        self.width = int(float(attrs.get("frameWidth", 1920)))
        self.height = int(float(attrs.get("frameHeight", 1080)))
        self.fps = float(attrs.get("fps", 30))

        # scaleFactor/translationToGlobalCoordinates: the dataset's calibration
        # export is inconsistent across scene batches. For some scenes
        # (verified via ground_truth.json: Warehouse_000 group, scaleFactor
        # ~9.556; Warehouse_010 group, ~14.421) the homography already
        # outputs global meters directly, and applying this correction would
        # BREAK a working projection. For others (Warehouse_026: every camera's
        # image-center projects to thousands of meters; Warehouse_027: some
        # cameras do, others don't) the raw output needs
        # X/scaleFactor + translation.x to land in a plausible range. There is
        # no reliable static rule (same scaleFactor value doesn't even
        # predict it consistently within Warehouse_027), so detect per-camera
        # by checking whether the raw image-center projection is plausible for
        # a real building.
        self.scale_factor = float(sensor.get("scaleFactor", 1.0))
        tr = sensor.get("translationToGlobalCoordinates", {"x": 0.0, "y": 0.0})
        self.translation = (float(tr.get("x", 0.0)), float(tr.get("y", 0.0)))
        cx, cy = self.width / 2.0, self.height / 2.0
        p = self.H_inv @ np.array([cx, cy, 1.0])
        raw_x, raw_y = p[0] / p[2], p[1] / p[2]
        self.needs_scale_correction = (
            abs(raw_x) > self.PLAUSIBLE_COORD_BOUND or abs(raw_y) > self.PLAUSIBLE_COORD_BOUND
        )

    def project_point3d(self, X, Y, Z):
        """World (X,Y,Z) meters -> pixel (u,v). Full 3D projection via cameraMatrix."""
        p = self.P @ np.array([X, Y, Z, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def pixel_to_ground(self, u, v):
        """Pixel (u,v) -> world ground-plane (X,Y) at Z=0, via inverse homography.

        Returns None when the projection can't be trusted: either the pixel
        is near this camera's vanishing line (the homogeneous denominator
        p[2] is near zero, so a tiny pixel error blows up into a huge world
        offset -- verified empirically, e.g. Warehouse_023 Camera_0016), or
        the result is still implausible for a real building even after the
        scaleFactor correction (a genuinely degenerate projection, not just
        an uncorrected one). Callers must handle None (skip the detection),
        not treat it as (0, 0) or any other silent default -- that's exactly
        the bug that let ~74-90% of Warehouse_026's detections through as
        garbage coordinates in the millions.
        """
        p = self.H_inv @ np.array([u, v, 1.0])
        if abs(p[2]) < 1e-6:
            return None
        X, Y = p[0] / p[2], p[1] / p[2]
        if self.needs_scale_correction:
            X = X / self.scale_factor + self.translation[0]
            Y = Y / self.scale_factor + self.translation[1]
        if abs(X) > self.PLAUSIBLE_COORD_BOUND or abs(Y) > self.PLAUSIBLE_COORD_BOUND:
            return None
        return X, Y

    def ground_to_pixel(self, X, Y):
        """World ground-plane (X,Y,Z=0) -> pixel (u,v), via homography."""
        p = self.H @ np.array([X, Y, 1.0])
        return p[0] / p[2], p[1] / p[2]


def load_camera_models(scene, split="train"):
    cal = load_calibration(scene, split)
    return {cam_id: CameraModel(sensor) for cam_id, sensor in cal.items()}
