"""
Step 1 - Read a workout video, detect the body pose on every frame,
draw the skeleton, and save a new video.

Usage:
    python step1_pose_video.py videos/squat.mp4
Output:
    output/squat_pose.mp4
"""
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = "models/pose_landmarker_full.task"

# Pairs of landmark indices to connect with lines (MediaPipe's 33-point body).
# Face points are skipped to keep the drawing clean.
CONNECTIONS = [
    (11, 12),                      # shoulders
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (27, 31),  # left leg + foot
    (24, 26), (26, 28), (28, 32),  # right leg + foot
]
BODY_POINTS = sorted({i for pair in CONNECTIONS for i in pair})


def ensure_model():
    """Download the pose model the first time the script runs."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading pose model (about 9 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model saved to", MODEL_PATH)


MAX_PEOPLE = 2   # how many people to track; raise it for group workouts

# One colour per person (BGR): green, purple, orange, cyan
PERSON_COLOURS = [(80, 200, 120), (220, 90, 180), (40, 150, 255), (220, 200, 40)]


def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def hip_x(landmarks):
    """Horizontal position of the hips, used to sort people left to right."""
    return (landmarks[23].x + landmarks[24].x) / 2


def draw_skeleton(frame, landmarks, colour, label, min_visibility=0.5):
    """Draw lines, joints and a name tag for one detected person."""
    h, w = frame.shape[:2]
    pts = {}
    for i in BODY_POINTS:
        lm = landmarks[i]
        if lm.visibility >= min_visibility:
            pts[i] = (int(lm.x * w), int(lm.y * h))

    thickness = max(2, w // 300)
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), thickness + 2, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], colour, thickness, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(frame, p, thickness + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, thickness + 1, colour, -1, cv2.LINE_AA)

    # Name tag above the head (nose = landmark 0)
    nose = landmarks[0]
    x, y = int(nose.x * w), max(30, int(nose.y * h) - 40)
    scale = max(0.6, w / 1200)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (x - tw // 2 - 8, y - th - 8), (x + tw // 2 + 8, y + 8), colour, -1)
    cv2.putText(frame, label, (x - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)


def main(video_path):
    ensure_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"output/{name}_pose.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    landmarker = create_landmarker()
    frame_idx, detected = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Sort left to right so "Person 1" is always the one on the left,
        # instead of swapping labels when the model's order changes.
        people = sorted(result.pose_landmarks, key=hip_x)
        for i, landmarks in enumerate(people):
            draw_skeleton(frame, landmarks, PERSON_COLOURS[i % len(PERSON_COLOURS)],
                          f"Person {i + 1}")
        detected = max(detected, len(people))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  processed {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()
    landmarker.close()
    print(f"\nDone. Up to {detected} people found across {frame_idx} frames.")
    print("Saved:", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python step1_pose_video.py path/to/video.mp4")
    main(sys.argv[1])
