"""
Step 3 - Squat form rules, live red/green limbs, and a score per rep.
Works from the SIDE or the FRONT; the camera view is detected automatically.

Side view, 100 points per rep:
  Depth        35 pts   knee angle at the bottom: full marks at 90 deg or deeper
  Torso lean   25 pts   chest should stay up: lean past 45 deg from vertical loses points
  Heels down   15 pts   heels should stay on the floor
  Tempo        25 pts   lowering should take at least 1 second

Front view, 100 points per rep:
  Depth        40 pts   how far the hips drop (hips level with knees = full marks)
  Knees        35 pts   knees must not cave inward (knee gap vs ankle gap)
  Tempo        25 pts   lowering should take at least 1 second

Usage:
    python step3_form_score.py videos/squat.mp4          (up to 2 people)
    python step3_form_score.py videos/squat.mp4 1        (only 1 person)
Output:
    output/squat_form.mp4      annotated video
    output/squat_reps.json     every rep's score and faults, plus timed coach
                               cues (Step 4 turns these into speech)
"""
import json
import os
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from pose_utils import (PERSON_COLOURS, SIDES, SideChooser, Smoother, angle,
                        create_landmarker, draw_label, joint, landmarks_to_array)

MAX_PEOPLE = 2
UP_ANGLE, DOWN_ANGLE = 160, 100          # rep counting thresholds (degrees)

GOOD, BAD, NEUTRAL = (90, 210, 110), (60, 60, 235), (255, 255, 255)   # BGR
ORANGE = (40, 170, 255)

# Scoring: (points, value where losing starts, value where all points are lost)
SIDE_RULES = {
    "depth": (35, 90.0, 115.0),    # knee angle at the bottom, degrees
    "lean":  (25, 45.0, 65.0),     # max torso lean from vertical, degrees
    "heels": (15, 0.06, 0.18),     # heel rise as a fraction of shin length
    "tempo": (25, 1.0, 0.4),       # lowering time, seconds (less = worse)
}
FRONT_RULES = {
    "depth": (40, 90.0, 115.0),    # depth from hip drop, as an equivalent angle
    "knees": (35, 0.90, 0.65),     # knee gap / ankle gap (less = caving in)
    "tempo": (25, 1.0, 0.4),
}
LOWER_IS_WORSE = {"tempo", "knees"}

# A rep only counts as GOOD if it passes all of these; otherwise it's a "no rep"
NO_REP_SCORE = 70
NO_REP_LEAN = 60.0             # side: leaning further than this
NO_REP_HEELS = 0.15            # side: heels lifting more than this
NO_REP_KNEES = 0.75            # front: knee gap below 75% of ankle gap
SHALLOW_ANGLE = 135            # a dip below this that never hits depth = half squat

LIVE_LEAN_LIMIT = 50.0         # torso turns red live past this
LIVE_HEEL_LIMIT = 0.10         # feet turn red live past this
LIVE_KNEE_LIMIT = 0.80         # legs turn red live below this (front view)

# Ignore detections that aren't a clearly visible full body (leaves, posters...)
MIN_BODY_VISIBILITY = 0.6
MIN_BODY_HEIGHT = 0.25         # shoulders-to-ankles, as a fraction of frame height
MIN_SEEN_SECONDS = 1.0         # a person's card appears only after this long

CUES = {
    "depth": "Go a little deeper",
    "lean": "Keep your chest up",
    "heels": "Keep your heels down",
    "knees": "Push your knees out",
    "tempo": "Slow down on the way down",
    None: "Great rep",
}


def penalty(value, rule, lower_is_worse=False):
    """Points lost, scaled linearly between the start and end of the rule."""
    points, start, end = rule
    if lower_is_worse:
        value, start, end = -value, -start, -end
    return points * float(np.clip((value - start) / (end - start), 0.0, 1.0))


# ---------------------------------------------------------------- measurements
def torso_lean(pts, side):
    """Angle between the hip->shoulder line and straight up, in degrees."""
    v = joint(pts, side, "shoulder")[:2] - joint(pts, side, "hip")[:2]
    cos = -v[1] / (np.linalg.norm(v) + 1e-6)          # image y points down
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def heel_rise(pts, side):
    """How far the heel sits above the toe, relative to shin length."""
    heel, toe = joint(pts, side, "heel"), joint(pts, side, "foot")
    shin = np.linalg.norm(joint(pts, side, "knee")[:2] - joint(pts, side, "ankle")[:2])
    return float((toe[1] - heel[1]) / (shin + 1e-6))


def mid(pts, name):
    return (joint(pts, "left", name)[:2] + joint(pts, "right", name)[:2]) / 2


def shoulder_ratio(pts):
    """Shoulder width / torso length. Wide from the front, narrow from the side."""
    width = abs(joint(pts, "left", "shoulder")[0] - joint(pts, "right", "shoulder")[0])
    torso = np.linalg.norm(mid(pts, "shoulder") - mid(pts, "hip"))
    return float(width / (torso + 1e-6))


def knee_ratio(pts):
    """Knee gap / ankle gap. About 1 or more when knees track over the feet,
    well below 1 when the knees cave in."""
    knees = abs(joint(pts, "left", "knee")[0] - joint(pts, "right", "knee")[0])
    ankles = abs(joint(pts, "left", "ankle")[0] - joint(pts, "right", "ankle")[0])
    return float(knees / (ankles + 1e-6))


def is_real_person(pts, frame_h):
    """Filters out false detections in leaves, posters, reflections..."""
    # Judge by the better-visible side: from the side, the far half is hidden
    best = max(np.mean([pts[SIDES[s][n], 2] for n in ("shoulder", "hip", "knee")])
               for s in SIDES)
    if best < MIN_BODY_VISIBILITY:
        return False
    height = max(pts[27, 1], pts[28, 1]) - min(pts[11, 1], pts[12, 1])
    return height >= MIN_BODY_HEIGHT * frame_h


class ViewDetector:
    """Decides whether the camera sees this person from the front or the side.
    Two thresholds so the view doesn't flicker between them."""

    def __init__(self):
        self.view = None
        self.history = deque(maxlen=15)

    def __call__(self, pts):
        # One side clearly hidden = side view, whatever the shoulder width says
        vis = [np.mean([pts[SIDES[s][n], 2] for n in ("shoulder", "hip", "knee", "ankle")])
               for s in SIDES]
        hidden_side = abs(vis[0] - vis[1]) > 0.3
        self.history.append(0.0 if hidden_side else shoulder_ratio(pts))
        r = float(np.median(self.history))
        if self.view is None:
            self.view = "front" if r > 0.5 else "side"
        elif self.view == "side" and r > 0.6:
            self.view = "front"
        elif self.view == "front" and r < 0.4:
            self.view = "side"
        return self.view


class HipDrop:
    """Front-view depth. The knee angle can't be read from the front (the thigh
    points at the camera), so we measure how far the hips drop instead and
    convert it to an equivalent knee angle: standing = 180, hips level with
    knees (about 55% of standing height) = 90."""

    def __init__(self, fps):
        self.recent = deque(maxlen=int(fps * 4))

    def __call__(self, pts):
        height = float(mid(pts, "ankle")[1] - mid(pts, "hip")[1])
        self.recent.append(height)
        standing = max(self.recent)
        ratio = height / (standing + 1e-6)
        return float(np.clip(180 - 200 * (1 - ratio), 40, 180))


# ---------------------------------------------------------------- the coach
class SquatCoach:
    """Counts reps and scores each one against the rules for its camera view."""

    def __init__(self, fps):
        self.fps = fps
        self.stage = "UP"
        self.reps = []
        self.heel_baseline = []
        self._reset_rep()

    def _reset_rep(self):
        self.rep_start = None
        self.min_knee, self.bottom_frame = 180.0, None
        self.max_lean, self.max_heel = 0.0, 0.0
        self.min_knee_ratio = 9.9
        self.front_frames = self.side_frames = 0
        self.live = {}                   # fault -> first frame it showed up mid-rep

    def baseline(self):
        return float(np.median(self.heel_baseline)) if self.heel_baseline else 0.0

    def update(self, knee, frame_idx, view, lean=0.0, heel=0.0, kratio=None):
        """Returns the finished rep's dict when a rep completes, else None."""
        if self.stage == "UP" and knee >= UP_ANGLE:
            half = None
            if self.rep_start is not None and self.min_knee < SHALLOW_ANGLE:
                half = self._score(frame_idx)          # a half squat
                self.reps.append(half)
            if view == "side":
                self.heel_baseline = (self.heel_baseline + [heel])[-60:]
            self._reset_rep()
            return half

        if self.rep_start is None:
            self.rep_start = frame_idx
        if knee < self.min_knee:
            self.min_knee, self.bottom_frame = knee, frame_idx
        if view == "front":
            self.front_frames += 1
            if kratio is not None and knee < 140:      # only judge knees when bent
                self.min_knee_ratio = min(self.min_knee_ratio, kratio)
                if kratio < LIVE_KNEE_LIMIT:
                    self.live.setdefault("knees", frame_idx)
        else:
            self.side_frames += 1
            self.max_lean = max(self.max_lean, lean)
            self.max_heel = max(self.max_heel, heel - self.baseline())
            if lean > LIVE_LEAN_LIMIT:
                self.live.setdefault("lean", frame_idx)
            if self.stage == "DOWN" and heel - self.baseline() > LIVE_HEEL_LIMIT:
                self.live.setdefault("heels", frame_idx)

        if self.stage == "UP" and knee < DOWN_ANGLE:
            self.stage = "DOWN"
        elif self.stage == "DOWN" and knee >= UP_ANGLE:
            self.stage = "UP"
            rep = self._score(frame_idx)
            self.reps.append(rep)
            self._reset_rep()
            return rep
        return None

    def _score(self, frame_idx):
        view = "front" if self.front_frames > self.side_frames else "side"
        descent = (self.bottom_frame - self.rep_start) / self.fps
        measured = {"depth": self.min_knee, "tempo": descent}
        if view == "side":
            rules = SIDE_RULES
            measured.update(lean=self.max_lean, heels=self.max_heel)
        else:
            rules = FRONT_RULES
            measured["knees"] = self.min_knee_ratio if self.min_knee_ratio < 9 else 1.0

        lost = {k: penalty(measured[k], rules[k], k in LOWER_IS_WORSE) for k in rules}
        worst = max(lost, key=lost.get)
        worst = worst if lost[worst] >= 3 else None
        score = int(round(100 - sum(lost.values())))

        # Hard failures, most serious first: these make it a "no rep"
        broken = []
        if self.min_knee >= DOWN_ANGLE:
            broken.append("depth")
        if view == "front" and measured["knees"] < NO_REP_KNEES:
            broken.append("knees")
        if view == "side" and self.max_lean > NO_REP_LEAN:
            broken.append("lean")
        if view == "side" and self.max_heel > NO_REP_HEELS:
            broken.append("heels")
        valid = not broken and score >= NO_REP_SCORE
        if broken:
            worst = broken[0] if broken[0] != "depth" or len(broken) == 1 else broken[1]

        return {
            "rep": len(self.reps) + 1,
            "start_s": round(self.rep_start / self.fps, 2),
            "bottom_s": round(self.bottom_frame / self.fps, 2),
            "time_s": round(frame_idx / self.fps, 2),
            # moments a fault first appeared during the rep (the voice shouts these)
            "live_faults": [{"fault": k, "time_s": round(f / self.fps, 2)}
                            for k, f in sorted(self.live.items(), key=lambda kv: kv[1])],
            "view": view,
            "score": score,
            "valid": valid,
            "measured": {k: round(v, 2) for k, v in measured.items()},
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
        self.hip_drop = HipDrop(fps)
        self.coach = SquatCoach(fps)
        self.knee = None
        self.view = "side"
        self.seen = 0
        self.series = []          # [time_s, knee_deg] every frame, for the Step 5 chart
        self.flash = 0
        self.feedback, self.feedback_frames = None, 0


# ---------------------------------------------------------------- drawing
def score_colour(score):
    return GOOD if score >= 85 else ORANGE if score >= 65 else BAD


def bone(frame, a, b, colour, t):
    a, b = tuple(int(v) for v in a[:2]), tuple(int(v) for v in b[:2])
    cv2.line(frame, a, b, (255, 255, 255), t + 3, cv2.LINE_AA)
    cv2.line(frame, a, b, colour, t + 1, cv2.LINE_AA)


def dot(frame, p, colour, t):
    p = tuple(int(v) for v in p[:2])
    cv2.circle(frame, p, t + 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p, t + 2, colour, -1, cv2.LINE_AA)


def draw_side_skeleton(frame, pts, side, person, lean, heel_up):
    """Near-side skeleton; each body part coloured by its own rule."""
    t = max(2, frame.shape[1] // 300)
    j = lambda n: joint(pts, side, n)
    legs = GOOD if person.knee is not None and person.knee <= 90 else NEUTRAL
    torso = BAD if lean > LIVE_LEAN_LIMIT else GOOD
    feet = BAD if heel_up else GOOD

    bone(frame, j("shoulder"), j("elbow"), person.colour, t)
    bone(frame, j("elbow"), j("wrist"), person.colour, t)
    bone(frame, j("shoulder"), j("hip"), torso, t)
    bone(frame, j("hip"), j("knee"), legs, t)
    bone(frame, j("knee"), j("ankle"), legs, t)
    for a, b in (("ankle", "heel"), ("heel", "foot"), ("ankle", "foot")):
        bone(frame, j(a), j(b), feet, t)
    for n in SIDES[side]:
        dot(frame, j(n), person.colour, t)


def draw_front_skeleton(frame, pts, person, knees_in):
    """Both sides are visible from the front; legs go red when knees cave in."""
    t = max(2, frame.shape[1] // 300)
    deep = person.knee is not None and person.knee <= 90
    legs = BAD if knees_in else GOOD if deep else NEUTRAL
    L = lambda n: joint(pts, "left", n)
    R = lambda n: joint(pts, "right", n)

    bone(frame, L("shoulder"), R("shoulder"), person.colour, t)
    bone(frame, L("hip"), R("hip"), person.colour, t)
    for j in (L, R):
        bone(frame, j("shoulder"), j("elbow"), person.colour, t)
        bone(frame, j("elbow"), j("wrist"), person.colour, t)
        bone(frame, j("shoulder"), j("hip"), person.colour, t)
        bone(frame, j("hip"), j("knee"), legs, t)
        bone(frame, j("knee"), j("ankle"), legs, t)
        bone(frame, j("ankle"), j("foot"), person.colour, t)
        for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"):
            dot(frame, j(n), person.colour, t)
    if knees_in:
        k = (L("knee")[:2] + R("knee")[:2]) / 2
        draw_label(frame, "KNEES IN", (k[0], k[1] - 50), BAD)


def facing_right(pts, side):
    return joint(pts, side, "foot")[0] > joint(pts, side, "heel")[0]


def draw_card(frame, person, x, y, card_w):
    s = card_w / 380
    h_card = int(250 * s)
    c = person.colour
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + card_w, y + h_card), (30, 30, 30), -1)
    alpha = 0.85 if person.flash else 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + int(10 * s), y + h_card), c, -1)
    if person.flash:
        cv2.rectangle(frame, (x, y), (x + card_w, y + h_card), c, max(2, int(4 * s)))

    f = cv2.FONT_HERSHEY_SIMPLEX
    put = lambda txt, px, py, sc, col, th=2: cv2.putText(
        frame, txt, (x + int(px * s), y + int(py * s)), f, sc * s, col,
        max(1, int(th * s)), cv2.LINE_AA)
    coach = person.coach

    put(f"PERSON {person.index + 1}", 28, 40, 0.8, c)
    put(person.view.upper(), 200, 40, 0.6, (190, 190, 190))
    put(coach.stage, 300, 40, 0.8, (255, 255, 255))
    put(str(coach.good_reps), 24, 140, 3.0, (255, 255, 255), 6)
    put(f"GOOD / {len(coach.reps)}", 28, 170, 0.7, (190, 190, 190))

    last = coach.reps[-1]["score"] if coach.reps else None
    put("LAST REP", 190, 75, 0.6, (190, 190, 190))
    put(str(last) if last is not None else "--", 190, 130, 1.8,
        score_colour(last) if last is not None else (190, 190, 190), 4)
    avg = coach.average
    put(f"AVG {avg}" if avg is not None else "AVG --", 190, 170, 0.7, (230, 230, 230))

    bx, by = x + int(28 * s), y + int(200 * s)
    bw, bh = card_w - int(56 * s), max(6, int(16 * s))
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    if person.knee is not None:
        fill = float(np.clip((180 - person.knee) / (180 - 70), 0, 1))
        cv2.rectangle(frame, (bx, by), (bx + int(bw * fill), by + bh),
                      GOOD if person.knee <= 90 else ORANGE, -1)
    target = bx + int(bw * (180 - 90) / (180 - 70))
    cv2.line(frame, (target, by - 6), (target, by + bh + 6), (255, 255, 255), 2)
    put("DEPTH", 28, 238, 0.55, (190, 190, 190))


def draw_feedback(frame, person, x, y, card_w):
    """Score + cue in a strip under the person's card for ~1.5 s after each rep."""
    if not person.feedback_frames:
        return
    rep = person.feedback
    s = card_w / 380
    h_strip = int(56 * s)
    colour = score_colour(rep["score"]) if rep["valid"] else BAD
    cv2.rectangle(frame, (x, y), (x + card_w, y + h_strip), colour, -1)
    text = f"{rep['score']}  {rep['cue']}" if rep["valid"] else rep["cue"].upper()
    scale = 0.75 * s
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    if tw > card_w - 24:
        scale *= (card_w - 24) / tw
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(frame, text, (x + (card_w - tw) // 2, y + (h_strip + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    person.feedback_frames -= 1


# ---------------------------------------------------------------- main
def open_writer(path, fps, w, h):
    """Open the output video. If the file is locked (usually because it's
    still open in a video player on Windows), save under a new name."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    candidates = [path] + [path.replace(".mp4", f"_{n}.mp4") for n in range(2, 10)]
    for candidate in candidates:
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except PermissionError:
            print(f"  {candidate} is open in another program, trying a new name...")
            continue
        writer = cv2.VideoWriter(candidate, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, candidate
    sys.exit("Could not create the output video. Close any open video players "
             "and check the 'output' folder isn't read-only.")


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
    clean_writer, clean_path = open_writer(f"output/{name}_skeleton.mp4", fps, w, h)

    landmarker = create_landmarker(max_people)
    people = [Person(i, fps) for i in range(max_people)]
    cues = []
    frame_idx = 0
    # Cards are bigger on vertical (phone) videos so they stay readable
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
        bodies = [b for b in bodies if is_real_person(b, h)]
        bodies.sort(key=lambda b: (b[23, 0] + b[24, 0]) / 2)   # left to right

        labels = []           # drawn after the clean copy is taken
        for i, person in enumerate(people):
            if i >= len(bodies):
                person.smoother.reset()
                person.knee = None
                continue
            pts = person.smoother(bodies[i])
            person.seen += 1
            view = person.view = person.view_detector(pts)

            if view == "side":
                side = person.side_chooser(pts)
                knee = angle(joint(pts, side, "hip"), joint(pts, side, "knee"),
                             joint(pts, side, "ankle"))
                lean, heel = torso_lean(pts, side), heel_rise(pts, side)
                person.knee = knee
                rep = person.coach.update(knee, frame_idx, "side", lean=lean, heel=heel)
                heel_up = person.coach.stage == "DOWN" and \
                    heel - person.coach.baseline() > LIVE_HEEL_LIMIT
                draw_side_skeleton(frame, pts, side, person, lean, heel_up)
                k = joint(pts, side, "knee")
                dx = 110 if facing_right(pts, side) else -110
                labels.append((f"{int(knee)} deg", (k[0] + dx, k[1]), person.colour))
            else:
                knee = person.hip_drop(pts)
                kr = knee_ratio(pts)
                person.knee = knee
                rep = person.coach.update(knee, frame_idx, "front", kratio=kr)
                draw_front_skeleton(frame, pts, person, knees_in=knee < 140 and kr < LIVE_KNEE_LIMIT)

            if rep:
                person.flash = int(fps * 0.4)
                person.feedback, person.feedback_frames = rep, int(fps * 1.5)
                cues.append({"time_s": rep["time_s"], "person": i + 1,
                             "rep": rep["rep"], "score": rep["score"],
                             "valid": rep["valid"], "text": rep["cue"]})
            person.series.append([round(frame_idx / fps, 3), round(knee, 1)])
            head = pts[0]
            labels.append((f"Person {i + 1}", (head[0], head[1] - 60), person.colour))

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
    report = {
        "video": video_path, "fps": fps, "duration_s": round(frame_idx / fps, 2),
        "people": [{"person": p.index + 1, "colour_bgr": list(p.colour),
                    "reps": p.coach.reps, "average_score": p.coach.average,
                    "series": p.series} for p in real],
        "cues": [c for c in cues if any(p.index + 1 == c["person"] for p in real)],
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    if size_mb < 0.01:
        print(f"\nWARNING: {out_path} is empty. Close any video player and run again.")
    print(f"\nSaved video: {os.path.abspath(out_path)} ({size_mb:.1f} MB)")
    print(f"Saved report: {os.path.abspath(json_path)}\n")
    for p in real:
        print(f"Person {p.index + 1}: {p.coach.good_reps} good reps out of "
              f"{len(p.coach.reps)} attempts, average score {p.coach.average}")
        for r in p.coach.reps:
            tag = "GOOD   " if r["valid"] else "NO REP "
            m = ", ".join(f"{k} {v}" for k, v in r["measured"].items())
            print(f"   {tag} rep {r['rep']} ({r['view']}): {r['score']}/100  "
                  f"[{m}]  -> {r['cue']}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python step3_form_score.py path/to/video.mp4 [max_people]")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else MAX_PEOPLE)
