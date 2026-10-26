"""
Push-up form rules, live red/green limbs, and a score per rep.
Same outputs as step3_form_score.py, so the voice coach (Step 4) and the
dashboard (Step 5) work on push-ups too.

Film from the SIDE: the elbow, body line and head are only measurable there.

Rules, 100 points per rep:
  Depth       35 pts   elbow angle at the bottom: full marks at 90 deg or less
  Body line   30 pts   hips in line with shoulders and ankles (no sagging, no piking)
  Tempo       15 pts   lowering should take at least half a second
  Lockout     10 pts   arms (almost) straight at the top: 160 deg or more
  Head        10 pts   head in line with the body, not dropping to the floor

Usage:
    python pushup_form_score.py videos/pushups.mp4          (up to 2 people)
    python pushup_form_score.py videos/pushups.mp4 1        (only 1 person)
Output:
    output/pushups_form.mp4       annotated video
    output/pushups_skeleton.mp4   skeleton only (Step 5 builds the dashboard on it)
    output/pushups_reps.json      reps, scores, faults, timings and the elbow angle per frame
"""
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from pose_utils import (PERSON_COLOURS, SIDES, SideChooser, Smoother, angle,
                        create_landmarker, draw_label, joint, landmarks_to_array)
from step3_form_score import (BAD, GOOD, MIN_SEEN_SECONDS, NEUTRAL, ViewDetector,
                              bone, dot, draw_card, draw_feedback, open_writer, penalty)

MAX_PEOPLE = 2
UP_ANGLE, DOWN_ANGLE = 150, 100    # elbow: above 150 = top, below 100 = rep is deep enough
SHALLOW_ANGLE = 140                # a dip below this that never reaches 100 = half rep

# Scoring: (points, value where losing starts, value where all points are lost)
RULES = {
    "depth":   (35, 90.0, 115.0),  # elbow angle at the bottom, degrees
    "body":    (30, 0.05, 0.10),   # hips off the shoulder-ankle line, fraction of body length
    "tempo":   (15, 0.5, 0.2),     # lowering time, seconds (less = worse)
    "lockout": (10, 160.0, 140.0), # elbow angle at the top, degrees (less = worse)
    "head":    (10, 30.0, 50.0),   # head bent down from the body line, degrees
}
LOWER_IS_WORSE = {"tempo", "lockout"}

NO_REP_SCORE = 70
NO_REP_BODY = 0.09          # hips this far off the line = no rep

LIVE_BODY_LIMIT = 0.07      # body line turns red live past this
LIVE_HEAD_LIMIT = 40.0      # neck turns red live past this
LIVE_HOLD_FRAMES = 5        # a fault must last this many frames before the coach shouts
BODY_PERCENTILE = 75        # body line is judged on its typical value during the rep,
                            # so a one-frame wobble doesn't cost points
WORKING_ANGLE = 140         # body line and head are judged only below this elbow angle

# Ignore detections that aren't a real person lying in a plank
MIN_BODY_VISIBILITY = 0.6
MIN_BODY_LENGTH = 0.25      # shoulder-to-ankle length, as a fraction of the frame's longer side
PLANK_KNEE = 140            # knees straighter than this...
PLANK_TILT = 40             # ...and body within this many degrees of horizontal = in a plank

EARS = {"left": 7, "right": 8}

CUES = {
    "depth": "Chest closer to the floor",
    "sag": "Hips up, squeeze your glutes",
    "pike": "Hips down, straight line",
    "tempo": "Slow down on the way down",
    "lockout": "Lock out your arms at the top",
    "head": "Keep your head in line",
    None: "Great rep",
}


# ---------------------------------------------------------------- measurements
def body_offset(pts, side):
    """How far the hips sit off the shoulder-ankle line, as a fraction of body length.
    Positive = hips sagging below the line, negative = hips piked up."""
    sh, hip, an = (joint(pts, side, n)[:2] for n in ("shoulder", "hip", "ankle"))
    length = np.linalg.norm(an - sh) + 1e-6
    if abs(an[0] - sh[0]) < 1e-3:
        return 0.0
    line_y = sh[1] + (an[1] - sh[1]) * (hip[0] - sh[0]) / (an[0] - sh[0])
    return float((hip[1] - line_y) / length)       # image y points down: + = below


def head_drop(pts, side):
    """Angle between the hip->shoulder direction and the shoulder->ear direction.
    About 0-25 deg when the head is in line, much more when it drops to the floor."""
    sh, hip = joint(pts, side, "shoulder")[:2], joint(pts, side, "hip")[:2]
    ear = pts[EARS[side], :2]
    u, v = sh - hip, ear - sh
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def in_plank(pts, side):
    """Straight legs and a roughly horizontal body. Stops reps being counted
    while someone gets into position or stands up."""
    knee = angle(joint(pts, side, "hip"), joint(pts, side, "knee"), joint(pts, side, "ankle"))
    d = joint(pts, side, "ankle")[:2] - joint(pts, side, "shoulder")[:2]
    tilt = np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6))
    return knee > PLANK_KNEE and tilt < PLANK_TILT


def is_real_person(pts, frame_w, frame_h):
    """Filters out false detections. Uses body LENGTH, because a push-up body
    is horizontal and would fail a height check."""
    best = max(np.mean([pts[SIDES[s][n], 2] for n in ("shoulder", "hip", "knee")])
               for s in SIDES)
    if best < MIN_BODY_VISIBILITY:
        return False
    side = "left" if pts[11, 2] + pts[23, 2] >= pts[12, 2] + pts[24, 2] else "right"
    length = np.linalg.norm(joint(pts, side, "ankle")[:2] - joint(pts, side, "shoulder")[:2])
    return length >= MIN_BODY_LENGTH * max(frame_w, frame_h)


# ---------------------------------------------------------------- the coach
class PushupCoach:
    """Counts push-ups from the elbow angle and scores each one."""

    def __init__(self, fps):
        self.fps = fps
        self.stage = "UP"
        self.reps = []
        self.top_angle = 0.0            # straightest elbow while at the top
        self._reset_rep()

    def _reset_rep(self):
        self.rep_start = None
        self.min_elbow, self.bottom_frame = 180.0, None
        self.offsets = []
        self.max_head = 0.0
        self.live = {}
        self.streak = {"sag": 0, "pike": 0, "head": 0}

    def abandon(self):
        """Person left the plank mid-rep: forget the rep without counting it."""
        self.stage = "UP"
        self.top_angle = 0.0
        self._reset_rep()

    def update(self, elbow, offset, head, frame_idx):
        """Returns the finished rep's dict when a rep completes, else None."""
        if self.stage == "UP" and elbow >= UP_ANGLE:
            half = None
            if self.rep_start is not None and self.min_elbow < SHALLOW_ANGLE:
                half = self._score(frame_idx)          # went down a bit, came back up
                self.reps.append(half)
                self.top_angle = 0.0
            self._reset_rep()
            self.top_angle = max(self.top_angle, elbow)
            return half

        if self.rep_start is None:
            self.rep_start = frame_idx
        if elbow < self.min_elbow:
            self.min_elbow, self.bottom_frame = elbow, frame_idx
        if elbow < WORKING_ANGLE:
            self.offsets.append(offset)
            self.max_head = max(self.max_head, head)
            for fault, on in (("sag", offset > LIVE_BODY_LIMIT),
                              ("pike", -offset > LIVE_BODY_LIMIT),
                              ("head", head > LIVE_HEAD_LIMIT)):
                self.streak[fault] = self.streak[fault] + 1 if on else 0
                if self.streak[fault] == LIVE_HOLD_FRAMES:
                    self.live.setdefault(fault, frame_idx - LIVE_HOLD_FRAMES + 1)

        if self.stage == "UP" and elbow < DOWN_ANGLE:
            self.stage = "DOWN"
        elif self.stage == "DOWN" and elbow >= UP_ANGLE:
            self.stage = "UP"
            rep = self._score(frame_idx)
            self.reps.append(rep)
            self._reset_rep()
            self.top_angle = elbow
            return rep
        return None

    def _score(self, frame_idx):
        descent = (self.bottom_frame - self.rep_start) / self.fps
        offs = np.array(self.offsets or [0.0])
        sag = max(0.0, float(np.percentile(offs, BODY_PERCENTILE)))
        pike = max(0.0, float(np.percentile(-offs, BODY_PERCENTILE)))
        hips = max(sag, pike)
        measured = {"depth": self.min_elbow, "body": hips, "tempo": descent,
                    "lockout": self.top_angle or UP_ANGLE, "head": self.max_head}
        lost = {k: penalty(measured[k], RULES[k], k in LOWER_IS_WORSE) for k in RULES}
        score = int(round(100 - sum(lost.values())))

        # name the body fault by its direction
        body_fault = "sag" if sag >= pike else "pike"
        worst = max(lost, key=lost.get)
        worst = worst if lost[worst] >= 3 else None
        if worst == "body":
            worst = body_fault

        broken = []
        if hips > NO_REP_BODY:
            broken.append(body_fault)
        if self.min_elbow >= DOWN_ANGLE:
            broken.append("depth")
        valid = not broken and score >= NO_REP_SCORE
        if broken:
            worst = broken[0]

        return {
            "rep": len(self.reps) + 1,
            "start_s": round(self.rep_start / self.fps, 2),
            "bottom_s": round(self.bottom_frame / self.fps, 2),
            "time_s": round(frame_idx / self.fps, 2),
            "live_faults": [{"fault": k, "time_s": round(f / self.fps, 2)}
                            for k, f in sorted(self.live.items(), key=lambda kv: kv[1])],
            "view": "side",
            "score": score,
            "valid": valid,
            "measured": {k: round(v, 3) for k, v in measured.items()},
            "points_lost": {k: round(v, 1) for k, v in lost.items()},
            "main_fault": worst,
            "cue": CUES[worst] if valid else f"No rep. {CUES[worst or 'depth']}",
        }

    @property
    def good_reps(self):
        return sum(r["valid"] for r in self.reps)

    @property
    def average(self):
        return int(round(np.mean([r["score"] for r in self.reps]))) if self.reps else None


class Person:
    def __init__(self, index, fps):
        self.index = index
        self.colour = PERSON_COLOURS[index % len(PERSON_COLOURS)]
        self.smoother = Smoother(alpha=0.5)
        self.side_chooser = SideChooser()
        self.view_detector = ViewDetector()
        self.coach = PushupCoach(fps)
        self.knee = None          # the card's depth bar reads this; for push-ups it's the elbow
        self.view = "side"
        self.seen = 0
        self.front_frames = 0
        self.horizontal = 0
        self.series = []
        self.flash = 0
        self.feedback, self.feedback_frames = None, 0


# ---------------------------------------------------------------- drawing
def draw_pushup_skeleton(frame, pts, side, person, offset, head):
    """Near-side skeleton: arms green at full depth, body line red when the hips
    sag or pike, neck red when the head drops."""
    t = max(2, frame.shape[1] // 300)
    j = lambda n: joint(pts, side, n)
    arms = GOOD if person.knee is not None and person.knee <= 90 else NEUTRAL
    body = BAD if abs(offset) > LIVE_BODY_LIMIT else GOOD
    neck = BAD if head > LIVE_HEAD_LIMIT else person.colour

    bone(frame, pts[EARS[side]], j("shoulder"), neck, t)
    bone(frame, j("shoulder"), j("elbow"), arms, t)
    bone(frame, j("elbow"), j("wrist"), arms, t)
    bone(frame, j("shoulder"), j("hip"), body, t)
    bone(frame, j("hip"), j("knee"), body, t)
    bone(frame, j("knee"), j("ankle"), body, t)
    bone(frame, j("ankle"), j("foot"), person.colour, t)
    for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"):
        dot(frame, j(n), person.colour, t)
    dot(frame, pts[EARS[side]], person.colour, t)


# ---------------------------------------------------------------- main
def main(video_path, max_people=MAX_PEOPLE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_path, json_path = f"output/{name}_form.mp4", f"output/{name}_reps.json"
    writer, out_path = open_writer(out_path, fps, w, h)
    clean_writer, _ = open_writer(f"output/{name}_skeleton.mp4", fps, w, h)

    landmarker = create_landmarker(max_people)
    people = [Person(i, fps) for i in range(max_people)]
    frame_idx = 0
    card_w = int(w * 0.2) if w >= h else int(w * (0.42 if max_people > 1 else 0.5))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(frame_idx * 1000 / fps))

        bodies = [landmarks_to_array(lms, w, h) for lms in result.pose_landmarks]
        bodies = [b for b in bodies if is_real_person(b, w, h)]
        bodies.sort(key=lambda b: (b[23, 0] + b[24, 0]) / 2)   # left to right

        labels = []
        for i, person in enumerate(people):
            if i >= len(bodies):
                person.smoother.reset()
                person.knee = None
                continue
            pts = person.smoother(bodies[i])
            person.seen += 1
            if person.view_detector(pts) == "front":
                person.front_frames += 1
            side = person.side_chooser(pts)
            d = joint(pts, side, "ankle")[:2] - joint(pts, side, "shoulder")[:2]
            person.horizontal += np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6)) < PLANK_TILT

            elbow = angle(joint(pts, side, "shoulder"), joint(pts, side, "elbow"),
                          joint(pts, side, "wrist"))
            offset, head = body_offset(pts, side), head_drop(pts, side)
            person.knee = elbow

            rep = None
            if in_plank(pts, side):
                rep = person.coach.update(elbow, offset, head, frame_idx)
            else:
                person.coach.abandon()

            draw_pushup_skeleton(frame, pts, side, person, offset, head)
            e = joint(pts, side, "elbow")
            labels.append((f"{int(elbow)} deg", (e[0], e[1] + 70), person.colour))
            if rep:
                person.flash = int(fps * 0.4)
                person.feedback, person.feedback_frames = rep, int(fps * 1.5)
            person.series.append([round(frame_idx / fps, 3), round(elbow, 1)])
            if len(bodies) > 1:            # name tags only matter with two people
                head_pt = pts[0]
                labels.append((f"Person {i + 1}", (head_pt[0], head_pt[1] - 70), person.colour))

        clean_writer.write(frame)
        for text, pos, colour in labels:
            draw_label(frame, text, pos, colour)

        shown = [p for p in people if p.seen >= fps * MIN_SEEN_SECONDS]
        for slot, person in enumerate(shown):
            x = 30 if slot == 0 else w - card_w - 30
            draw_card(frame, person, x, 30, card_w)
            draw_feedback(frame, person, x, 30 + int(250 * card_w / 380) + 8, card_w)
            person.flash = max(0, person.flash - 1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  processed {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()
    clean_writer.release()
    landmarker.close()

    real = [p for p in people if p.seen >= fps * MIN_SEEN_SECONDS]
    cues = sorted(({"time_s": r["time_s"], "person": p.index + 1, "rep": r["rep"],
                    "score": r["score"], "valid": r["valid"], "text": r["cue"]}
                   for p in real for r in p.coach.reps), key=lambda c: c["time_s"])
    report = {
        "exercise": "pushup",
        "angle_name": "Elbow angle",
        "angle_refs": [[160, "arms locked"], [90, "chest down"]],
        "video": video_path, "fps": fps, "duration_s": round(frame_idx / fps, 2),
        "people": [{"person": p.index + 1, "colour_bgr": list(p.colour),
                    "reps": p.coach.reps, "average_score": p.coach.average,
                    "series": p.series,
                    "horizontal_share": round(p.horizontal / max(p.seen, 1), 2)} for p in real],
        "cues": cues,
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"\nSaved video: {os.path.abspath(out_path)} ({size_mb:.1f} MB)")
    print(f"Saved report: {os.path.abspath(json_path)}\n")
    if not any(p.coach.reps for p in real) and real and \
            all(p.horizontal < p.seen / 4 for p in real):
        print("No push-ups found, and the person is upright most of the time.\n"
              f"This looks like a squat video. Try:  python run_coach.py {video_path}\n")
    for p in real:
        if p.front_frames > p.seen / 2:
            print(f"Tip: Person {p.index + 1} looks filmed from the front. "
                  "Push-ups are judged best from the side.")
        print(f"Person {p.index + 1}: {p.coach.good_reps} good push-ups out of "
              f"{len(p.coach.reps)} attempts, average score {p.coach.average}")
        for r in p.coach.reps:
            tag = "GOOD   " if r["valid"] else "NO REP "
            m = r["measured"]
            print(f"   {tag} rep {r['rep']}: {r['score']}/100  [elbow {m['depth']:.0f}, "
                  f"hips {m['body']:.3f}, head {m['head']:.0f}, top {m['lockout']:.0f}, "
                  f"lowering {m['tempo']:.2f}s]  -> {r['cue']}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python pushup_form_score.py path/to/video.mp4 [max_people]")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else MAX_PEOPLE)
