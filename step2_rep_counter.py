"""
Step 2 - Count squat reps for every person in a video.

- Uses only the body side facing the camera (no more ghost legs)
- Smooths joints to stop jitter
- Measures the knee angle and counts reps with a state machine
- Draws a HUD card per person: rep count, phase, depth bar

Usage:
    python step2_rep_counter.py videos/squat.mp4
Output:
    output/squat_reps.mp4      the annotated video
    output/squat_angles.csv    knee angle per person per frame (for tuning)
"""
import csv
import os
import sys

import cv2
import mediapipe as mp

from pose_utils import (PERSON_COLOURS, SideChooser, Smoother, angle,
                        create_landmarker, draw_label, draw_side_skeleton,
                        joint, landmarks_to_array)

MAX_PEOPLE = 2

# Knee-angle thresholds (degrees). 180 = standing straight.
UP_ANGLE = 160     # above this = standing, a rep can finish
DOWN_ANGLE = 100   # below this = deep enough, the rep is "loaded"
# The 60-degree gap between them is the hysteresis: jitter can't double-count.


class SquatCounter:
    """UP -> DOWN -> UP = one rep. Remembers each rep's depth and duration."""

    def __init__(self):
        self.stage = "UP"
        self.reps = []            # list of dicts, one per finished rep
        self.min_angle = 180.0
        self.rep_start = None     # frame when the person left the top

    def update(self, knee, frame_idx, fps):
        if self.stage == "UP":
            if knee < UP_ANGLE and self.rep_start is None:
                self.rep_start = frame_idx
            if knee >= UP_ANGLE:
                self.rep_start = None          # back at the top, not a rep yet
                self.min_angle = 180.0
            if knee < DOWN_ANGLE:
                self.stage = "DOWN"
        else:  # DOWN
            if knee >= UP_ANGLE:
                start = self.rep_start if self.rep_start is not None else frame_idx
                self.reps.append({
                    "rep": len(self.reps) + 1,
                    "depth_deg": round(self.min_angle, 1),
                    "duration_s": round((frame_idx - start) / fps, 2),
                })
                self.stage = "UP"
                self.rep_start = None
                self.min_angle = 180.0
                return True                     # a rep just finished
        self.min_angle = min(self.min_angle, knee)
        return False

    @property
    def count(self):
        return len(self.reps)


class Person:
    def __init__(self, index):
        self.index = index
        self.colour = PERSON_COLOURS[index % len(PERSON_COLOURS)]
        self.smoother = Smoother(alpha=0.5)
        self.side_chooser = SideChooser()
        self.counter = SquatCounter()
        self.knee = None
        self.flash = 0            # frames left to flash the card after a rep


def draw_card(frame, person, x, y, card_w):
    """HUD card: name, big rep number, phase, and a knee-depth bar."""
    h_card = int(card_w * 0.55)
    c = person.colour
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + card_w, y + h_card), (30, 30, 30), -1)
    alpha = 0.85 if person.flash else 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 10, y + h_card), c, -1)
    if person.flash:
        cv2.rectangle(frame, (x, y), (x + card_w, y + h_card), c, 4)

    s = card_w / 380
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"PERSON {person.index + 1}", (x + 28, y + int(40 * s)),
                font, 0.75 * s, c, 2, cv2.LINE_AA)
    cv2.putText(frame, str(person.counter.count), (x + 24, y + int(150 * s)),
                font, 3.2 * s, (255, 255, 255), int(6 * s), cv2.LINE_AA)
    cv2.putText(frame, "REPS", (x + int(150 * s), y + int(150 * s)),
                font, 0.8 * s, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, person.counter.stage, (x + int(260 * s), y + int(40 * s)),
                font, 0.75 * s, (255, 255, 255), 2, cv2.LINE_AA)

    # Depth bar: fills as the knee bends from 180 down to 70 degrees
    bx, by = x + 28, y + int(175 * s)
    bw, bh = card_w - 56, int(14 * s)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    if person.knee is not None:
        fill = min(1.0, max(0.0, (180 - person.knee) / (180 - 70)))
        deep = person.knee <= DOWN_ANGLE
        cv2.rectangle(frame, (bx, by), (bx + int(bw * fill), by + bh),
                      (80, 200, 120) if deep else (40, 170, 255), -1)
    target = bx + int(bw * (180 - DOWN_ANGLE) / (180 - 70))
    cv2.line(frame, (target, by - 6), (target, by + bh + 6), (255, 255, 255), 2)


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"output/{name}_reps.mp4"
    csv_path = f"output/{name}_angles.csv"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv_file = open(csv_path, "w", newline="")
    log = csv.writer(csv_file)
    log.writerow(["frame", "time_s", "person", "side", "knee_deg", "stage", "reps"])

    landmarker = create_landmarker(MAX_PEOPLE)
    people = [Person(i) for i in range(MAX_PEOPLE)]
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(frame_idx * 1000 / fps))

        # Sort left to right so each person keeps the same index
        detections = sorted(result.pose_landmarks,
                            key=lambda lms: (lms[23].x + lms[24].x) / 2)

        for i, person in enumerate(people):
            if i >= len(detections):
                person.smoother.reset()
                person.knee = None
                continue
            pts = person.smoother(landmarks_to_array(detections[i], w, h))
            side = person.side_chooser(pts)
            knee = angle(joint(pts, side, "hip"), joint(pts, side, "knee"),
                         joint(pts, side, "ankle"))
            person.knee = knee
            if person.counter.update(knee, frame_idx, fps):
                person.flash = int(fps * 0.4)

            draw_side_skeleton(frame, pts, side, person.colour)
            k = joint(pts, side, "knee")
            draw_label(frame, f"{int(knee)} deg", (k[0] + 70, k[1]), person.colour)
            head = pts[0]
            draw_label(frame, f"Person {i + 1}", (head[0], head[1] - 60), person.colour)

            log.writerow([frame_idx, round(frame_idx / fps, 2), i + 1, side,
                          round(knee, 1), person.counter.stage, person.counter.count])

        # HUD cards: one per person, spread across the top
        card_w = int(w * 0.2)
        for i, person in enumerate(people):
            x = 30 if i == 0 else w - card_w - 30 if i == len(people) - 1 \
                else int((w - card_w) * i / (len(people) - 1))
            draw_card(frame, person, x, 30, card_w)
            person.flash = max(0, person.flash - 1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  processed {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()
    csv_file.close()
    landmarker.close()

    print(f"\nSaved video: {out_path}")
    print(f"Saved angles: {csv_path}\n")
    for p in people:
        print(f"Person {p.index + 1}: {p.counter.count} reps")
        for r in p.counter.reps:
            print(f"   rep {r['rep']}: depth {r['depth_deg']} deg, {r['duration_s']} s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python step2_rep_counter.py path/to/video.mp4")
    main(sys.argv[1])
