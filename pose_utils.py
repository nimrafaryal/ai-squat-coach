"""
Shared helpers for the AI Gym Coach: model loading, smoothing,
near-side selection, joint angles and drawing.
Every step from Step 2 onward imports this file.
"""
import os
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = "models/pose_landmarker_full.task"

# One colour per person (BGR): green, purple, orange, cyan
PERSON_COLOURS = [(80, 200, 120), (220, 90, 180), (40, 150, 255), (220, 200, 40)]

# MediaPipe landmark indices for each side of the body
SIDES = {
    "left":  {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23,
              "knee": 25, "ankle": 27, "heel": 29, "foot": 31},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24,
              "knee": 26, "ankle": 28, "heel": 30, "foot": 32},
}
# Bones drawn for the near side only
SIDE_BONES = [("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
              ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"),
              ("heel", "foot"), ("ankle", "foot")]


# ---------------------------------------------------------------- model
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading pose model (about 9 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker(max_people=2):
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max_people,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def landmarks_to_array(landmarks, w, h):
    """33 landmarks -> array of shape (33, 3): pixel x, pixel y, visibility."""
    return np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks],
                    dtype=np.float32)


# ---------------------------------------------------------------- smoothing
class Smoother:
    """Exponential moving average on landmark positions, to stop jitter.
    alpha near 1 = follows quickly, near 0 = very smooth but laggy."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = None

    def __call__(self, pts):
        if self.state is None:
            self.state = pts.copy()
        else:
            self.state[:, :2] = self.alpha * pts[:, :2] + (1 - self.alpha) * self.state[:, :2]
            self.state[:, 2] = pts[:, 2]
        return self.state

    def reset(self):
        self.state = None


# ---------------------------------------------------------------- near side
class SideChooser:
    """Picks the body side facing the camera. The far side is hidden in a
    side-view video, so its points are guesses (that caused the 'ghost leg').
    A margin stops the choice flipping back and forth every frame."""

    KEY_POINTS = ("shoulder", "hip", "knee", "ankle")

    def __init__(self, margin=0.1):
        self.margin = margin
        self.side = None

    def __call__(self, pts):
        score = {s: np.mean([pts[SIDES[s][k], 2] for k in self.KEY_POINTS])
                 for s in SIDES}
        best = max(score, key=score.get)
        if self.side is None or score[best] > score[self.side] + self.margin:
            self.side = best
        return self.side


# ---------------------------------------------------------------- angles
def angle(a, b, c):
    """Angle at point b (degrees) formed by points a-b-c."""
    ba, bc = a[:2] - b[:2], c[:2] - b[:2]
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def joint(pts, side, name):
    return pts[SIDES[side][name]]


# ---------------------------------------------------------------- drawing
def draw_side_skeleton(frame, pts, side, colour, min_visibility=0.3):
    """Draw only the near-side bones of one person."""
    w = frame.shape[1]
    t = max(2, w // 300)
    idx = SIDES[side]
    visible = lambda n: pts[idx[n], 2] >= min_visibility
    p = lambda n: (int(pts[idx[n], 0]), int(pts[idx[n], 1]))

    for a, b in SIDE_BONES:
        if visible(a) and visible(b):
            cv2.line(frame, p(a), p(b), (255, 255, 255), t + 3, cv2.LINE_AA)
            cv2.line(frame, p(a), p(b), colour, t + 1, cv2.LINE_AA)
    for n in idx:
        if visible(n):
            cv2.circle(frame, p(n), t + 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p(n), t + 2, colour, -1, cv2.LINE_AA)


def draw_label(frame, text, centre, colour, scale=None, pad=10):
    """Filled tag with white text, centred on `centre`."""
    if scale is None:
        scale = max(0.6, frame.shape[1] / 1600)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = int(centre[0] - tw / 2), int(centre[1] + th / 2)
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), colour, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
