"""
Step 5 - Dashboard video: your clip on the left, live analytics on the right.

  - Stat cards: clean reps (with a dot per rep), animated form-score ring,
    and a live depth gauge whose needle follows your knee
  - Glowing knee-angle chart that draws itself, with green/red rep bands
  - Rep chips that fade in as each rep finishes
  - An animated "Set complete" summary card at the end, while the coach talks

Run Steps 3 and 4 first, then:
    python step5_dashboard.py videos/squat.mp4
Output:
    output/squat_dashboard.mp4
"""
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SS = 2                                   # draw at 2x, then shrink = smooth edges

# ---------------------------------------------------------------- layout (1x pixels)
H = 1080
MARGIN, GAP = 36, 28
PANEL_W = 820                 # right panel, portrait videos
WIDE_VIDEO_W = 1100           # video width for landscape videos
WIDE_PANEL_W = 720            # right panel, landscape videos
VIDEO_MAX_W = 900
RADIUS = 26

# ---------------------------------------------------------------- palette (RGB)
BG_TOP, BG_BOTTOM = (10, 14, 26), (17, 24, 41)
CARD = (22, 30, 48)
CARD_EDGE = (44, 56, 82)
TEXT = (238, 242, 250)
MUTED = (128, 140, 165)
FAINT = (58, 70, 96)
GOOD = (124, 230, 120)      # lime green
BAD = (255, 92, 110)        # coral
WARN = (255, 184, 76)       # amber
LINE = (76, 201, 240)       # cyan
PERSON_RGB = [(124, 230, 120), (196, 120, 255), (255, 184, 76), (76, 201, 240)]

STANDING, PARALLEL = 160, 90
ANGLE_TOP, ANGLE_BOTTOM = 185, 35
FAULT_TAG = {"depth": "DEPTH", "lean": "CHEST", "heels": "HEELS", "knees": "KNEES",
             "tempo": "RUSHED", None: "NO REP"}
FOCUS = {"depth": "Go a little deeper", "lean": "Keep your chest up",
         "heels": "Keep your heels down", "knees": "Push your knees out",
         "tempo": "Slow down on the way down"}

# Labels per exercise (the report from Step 3 says which one it is)
EXERCISES = {
    "squat": {"title": "AI SQUAT COACH", "angle": "Knee angle",
              "refs": {STANDING: "standing", PARALLEL: "parallel"},
              "tags": FAULT_TAG, "focus": FOCUS},
    "pushup": {"title": "AI PUSH-UP COACH", "angle": "Elbow angle",
               "refs": {STANDING: "arms locked", PARALLEL: "chest down"},
               "tags": {"depth": "DEPTH", "sag": "HIPS SAG", "pike": "HIPS UP",
                        "head": "HEAD", "tempo": "RUSHED", "lockout": "LOCKOUT",
                        None: "NO REP"},
               "focus": {"depth": "Get your chest lower", "sag": "Keep your hips up",
                         "pike": "Keep a straight body line", "head": "Keep your head in line",
                         "tempo": "Slow down on the way down",
                         "lockout": "Lock out at the top"}},
}

# ---------------------------------------------------------------- fonts
FONT_DIR = "fonts"
POPPINS = "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-{}.ttf"


def get_fonts():
    """Poppins, downloaded once into ./fonts. Falls back to Segoe UI / Arial /
    DejaVu if there's no internet."""
    os.makedirs(FONT_DIR, exist_ok=True)
    found = {}
    for weight in ("Regular", "Medium", "SemiBold", "Bold"):
        path = os.path.join(FONT_DIR, f"Poppins-{weight}.ttf")
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(POPPINS.format(weight), path)
            except Exception:
                pass
        if os.path.exists(path):
            found[weight] = path
    if len(found) == 4:
        return found
    fallbacks = {
        "Regular": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "Medium": ["C:/Windows/Fonts/seguisb.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "SemiBold": ["C:/Windows/Fonts/seguisb.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "Bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    }
    for weight, paths in fallbacks.items():
        if weight not in found:
            found[weight] = next((p for p in paths if os.path.exists(p)), None)
    return found


FONT_PATHS = get_fonts()
_cache = {}


def font(size, weight="Regular"):
    """A font at 1x `size`, loaded at the 2x drawing scale."""
    key = (size, weight)
    if key not in _cache:
        path = FONT_PATHS.get(weight)
        _cache[key] = ImageFont.truetype(path, size * SS) if path else \
            ImageFont.load_default(size * SS)
    return _cache[key]


# ---------------------------------------------------------------- drawing layer
class Layer:
    """A transparent 2x drawing surface over one region of the canvas.
    Every method takes normal 1x canvas coordinates."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def p(self, x, y):
        return ((x - self.x0) * SS, (y - self.y0) * SS)

    def box(self, x0, y0, x1, y1):
        return [*self.p(x0, y0), *self.p(x1, y1)]

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(self.box(x0, y0, x1, y1), r * SS, fill=fill,
                                 outline=outline, width=width * SS)

    def text(self, x, y, s, size, weight="Regular", fill=TEXT, anchor="la"):
        self.d.text(self.p(x, y), s, font=font(size, weight), fill=fill, anchor=anchor)

    def width_of(self, s, size, weight="Regular"):
        return self.d.textlength(s, font=font(size, weight)) / SS

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self.d.ellipse(self.box(cx - r, cy - r, cx + r, cy + r), fill=fill,
                       outline=outline, width=width * SS)

    def arc(self, cx, cy, r, start, end, fill, width):
        self.d.arc(self.box(cx - r, cy - r, cx + r, cy + r), start, end, fill=fill,
                   width=width * SS)

    def line(self, pts, fill, width, joint="curve"):
        self.d.line([self.p(*q) for q in pts], fill=fill, width=width * SS, joint=joint)

    def polygon(self, pts, fill):
        self.d.polygon([self.p(*q) for q in pts], fill=fill)

    def onto(self, canvas):
        """Shrink to 1x (this is the anti-aliasing) and composite onto the canvas."""
        small = self.img.reduce(SS)
        canvas.alpha_composite(small, (self.x0, self.y0))


def with_alpha(rgb, a):
    return (*rgb, int(255 * max(0.0, min(1.0, a))))


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------- dashboard
class Dashboard:
    def __init__(self, report, video_w, video_h):
        self.report = report
        self.ex = EXERCISES.get(report.get("exercise", "squat"), EXERCISES["squat"])
        self.people = report["people"]
        self.many = len(self.people) > 1
        self.duration = max(report["duration_s"], 1.0)
        for k, p in enumerate(self.people):
            p["rgb"] = PERSON_RGB[k % len(PERSON_RGB)]
            p["_times"] = np.array([s[0] for s in p["series"]]) if p["series"] else np.zeros(0)

        y = MARGIN
        self.top = (y, y + 56)
        self.stats = (y + 76, y + 316)
        if video_w > video_h * 1.1:
            # Landscape: big video top-left, stats and chart on the right,
            # rep breakdown full width along the bottom
            self.vw = WIDE_VIDEO_W
            self.vh = int(round(self.vw * video_h / video_w)) // 2 * 2
            if self.vh > H - 2 * MARGIN - GAP - 300:          # very tall-ish: keep room
                self.vh = (H - 2 * MARGIN - GAP - 300) // 2 * 2
                self.vw = int(round(self.vh * video_w / video_h)) // 2 * 2
            self.vx, self.vy = MARGIN, MARGIN
            self.pw = WIDE_PANEL_W
            self.px = MARGIN + WIDE_VIDEO_W + GAP
            self.chart = (y + 336, self.vy + self.vh)
            self.chipbox = (MARGIN, self.vy + self.vh + GAP,
                            self.px + self.pw, H - MARGIN)
        else:
            # Portrait / square: video on the left, everything else stacked on the right
            box_h = H - 2 * MARGIN
            self.vw = min(int(round(box_h * video_w / video_h)), VIDEO_MAX_W) // 2 * 2
            self.vh = int(round(self.vw * video_h / video_w)) // 2 * 2
            self.vx, self.vy = MARGIN, MARGIN + (box_h - self.vh) // 2
            self.pw = PANEL_W
            self.px = MARGIN + self.vw + GAP
            self.chart = (y + 336, y + 676)
            self.chipbox = (self.px, y + 696, self.px + PANEL_W, H - MARGIN)
        self.W = (self.px + self.pw + MARGIN) // 2 * 2
        # plot area inside the chart card
        self.ax0, self.ax1 = self.px + 118, self.px + self.pw - 30
        self.ay0, self.ay1 = self.chart[0] + 100, self.chart[1] - 46   # pills sit above the plot

        self.base = self.build_base()
        self.mask = self.build_video_mask()
        self.grad = self.build_gradient()

    # ---- static parts, drawn once
    def build_base(self):
        img = Image.new("RGBA", (self.W, H))
        col = np.linspace(0, 1, H)[:, None, None]
        top, bottom = np.array(BG_TOP), np.array(BG_BOTTOM)
        grad = (top + (bottom - top) * col).astype(np.uint8)
        img.paste(Image.fromarray(np.repeat(grad, self.W, axis=1), "RGB"))
        # soft glow blobs for depth
        glow = Image.new("RGBA", (self.W, H), (0, 0, 0, 0))
        g = ImageDraw.Draw(glow)
        g.ellipse((self.px + 300, -260, self.px + 1100, 400), fill=(76, 201, 240, 40))
        g.ellipse((-200, H - 380, 500, H + 280), fill=(124, 230, 120, 26))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(120)))

        # shadows under the cards
        shadow = Image.new("RGBA", (self.W, H), (0, 0, 0, 0))
        s = ImageDraw.Draw(shadow)
        boxes = [(self.vx, self.vy, self.vx + self.vw, self.vy + self.vh)] + \
            self.stat_boxes() + [(self.px, self.chart[0], self.px + self.pw, self.chart[1]),
                                 self.chipbox]
        for b in boxes:
            s.rounded_rectangle((b[0], b[1] + 14, b[2], b[3] + 14), RADIUS, fill=(0, 0, 0, 120))
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))

        L = Layer(0, 0, self.W, H)
        for b in self.stat_boxes():
            L.rrect(*b, RADIUS, fill=CARD, outline=CARD_EDGE)
        L.rrect(self.px, self.chart[0], self.px + self.pw, self.chart[1], RADIUS,
                fill=CARD, outline=CARD_EDGE)
        L.rrect(*self.chipbox, RADIUS, fill=CARD, outline=CARD_EDGE)
        L.rrect(self.vx - 1, self.vy - 1, self.vx + self.vw + 1, self.vy + self.vh + 1,
                RADIUS + 1, outline=CARD_EDGE, width=2)

        # brand
        L.circle(self.px + 12, self.top[0] + 26, 9, fill=GOOD)
        L.text(self.px + 32, self.top[0] + 26, self.ex["title"], 24, "Bold", anchor="lm")

        # static labels
        labels = ["CLEAN REPS", "FORM SCORE", "DEPTH"]
        for b, label in zip(self.stat_boxes(), labels):
            L.text(b[0] + 24, b[1] + 26, label, 15, "SemiBold", MUTED, anchor="lm")
        L.text(self.px + 28, self.chart[0] + 34, self.ex["angle"], 22, "SemiBold", anchor="lm")
        L.text(self.chipbox[0] + 28, self.chipbox[1] + 34, "Rep breakdown", 22, "SemiBold",
               anchor="lm")
        for a, name in self.ex["refs"].items():
            L.text(self.ax0 - 12, self.Y(a), name, 14, "Medium", MUTED, anchor="rm")
        for sec in range(0, int(self.duration) + 1, 5):
            L.text(self.X(sec), self.ay1 + 22, f"{sec}s", 13, "Medium", MUTED, anchor="mm")
        # legend
        lx = self.px + self.pw - 28
        if self.many:
            for p in reversed(self.people):
                w = L.width_of(f"Person {p['person']}", 15, "Medium")
                L.text(lx, self.chart[0] + 34, f"Person {p['person']}", 15, "Medium", MUTED, anchor="rm")
                L.circle(lx - w - 14, self.chart[0] + 34, 6, fill=p["rgb"])
                lx -= w + 40
        else:
            for name, colour in (("no rep", BAD), ("good rep", GOOD)):
                w = L.width_of(name, 15, "Medium")
                L.text(lx, self.chart[0] + 34, name, 15, "Medium", MUTED, anchor="rm")
                L.rrect(lx - w - 26, self.chart[0] + 27, lx - w - 12, self.chart[0] + 41, 4,
                        fill=with_alpha(colour, 0.55))
                lx -= w + 46
        L.onto(img)
        return img

    def stat_boxes(self):
        w = (self.pw - 2 * 20) / 3
        return [(int(self.px + i * (w + 20)), self.stats[0],
                 int(self.px + i * (w + 20) + w), self.stats[1]) for i in range(3)]

    def build_video_mask(self):
        m = Image.new("L", (self.vw * SS, self.vh * SS), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, self.vw * SS - 1, self.vh * SS - 1),
                                            RADIUS * SS, fill=255)
        return m.reduce(SS)

    def build_gradient(self):
        """Vertical fade used under the chart line."""
        h = self.ay1 - self.ay0
        ramp = (np.linspace(1, 0, h) ** 1.4 * 110).astype(np.uint8)
        return Image.fromarray(np.repeat(ramp[:, None], self.ax1 - self.ax0, axis=1), "L")

    # ---- coordinates
    def X(self, t):
        return self.ax0 + (self.ax1 - self.ax0) * t / self.duration

    def Y(self, a):
        a = float(np.clip(a, ANGLE_BOTTOM, ANGLE_TOP))
        return self.ay0 + (self.ay1 - self.ay0) * (ANGLE_TOP - a) / (ANGLE_TOP - ANGLE_BOTTOM)

    def live_angle(self, p, t):
        if not len(p["_times"]):
            return None
        i = int(np.searchsorted(p["_times"], t, side="right")) - 1
        if i < 0 or t - p["series"][i][0] > 0.25:
            return None
        return p["series"][i][1]

    # ---- one frame
    def render(self, frame, t, end_progress=None):
        canvas = self.base.copy()
        video = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (self.vw, self.vh),
                                                        interpolation=cv2.INTER_AREA),
                                             cv2.COLOR_BGR2RGB))
        canvas.paste(video, (self.vx, self.vy), self.mask)
        done = [[r for r in p["reps"] if r["time_s"] <= t] for p in self.people]

        L = Layer(self.px, 0, self.pw, H)
        self.draw_status(L, t, end_progress is not None)
        self.draw_stats(L, t, done)
        L.onto(canvas)
        x0, y0, x1, y1 = self.chipbox
        C = Layer(x0, y0, x1 - x0, y1 - y0)
        self.draw_chips(C, t, done)
        C.onto(canvas)
        self.draw_chart(canvas, t, done)

        V = Layer(self.vx, self.vy, self.vw, self.vh)
        self.draw_view_badge(V)
        self.draw_toast(V, t, done)
        if end_progress is not None:
            self.draw_summary(V, end_progress)
        V.onto(canvas)
        return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)

    def draw_status(self, L, t, done):
        right, cy = self.px + self.pw, self.top[0] + 26
        if done:
            label, colour = "SET COMPLETE", GOOD
        else:
            label, colour = f"LIVE  {int(t) // 60:02d}:{int(t) % 60:02d}", BAD
        w = L.width_of(label, 16, "SemiBold") + 50
        L.rrect(right - w, cy - 19, right, cy + 19, 19, fill=with_alpha(colour, 0.14),
                outline=with_alpha(colour, 0.5))
        pulse = 0.55 + 0.45 * math.sin(t * 6) if not done else 1
        L.circle(right - w + 20, cy, 6, fill=with_alpha(colour, pulse))
        L.text(right - 16, cy, label, 16, "SemiBold", colour, anchor="rm")

    # ---- the three stat cards
    def draw_stats(self, L, t, done):
        a, b, c = self.stat_boxes()
        allreps = sorted((r for reps in done for r in reps), key=lambda r: r["time_s"])
        good = sum(r["valid"] for r in allreps)
        recent = allreps and t - allreps[-1]["time_s"] < 0.35

        # 1) clean reps
        x0, y0, x1, y1 = a
        if self.many:
            for k, (p, reps) in enumerate(zip(self.people, done)):
                yy = y0 + 92 + k * 72
                L.circle(x0 + 34, yy, 8, fill=p["rgb"])
                g = sum(r["valid"] for r in reps)
                L.text(x0 + 52, yy, str(g), 52, "Bold", anchor="lm")
                L.text(x0 + 56 + L.width_of(str(g), 52, "Bold"), yy + 8,
                       f"/ {len(reps)}", 22, "Medium", MUTED, anchor="lm")
        else:
            colour = GOOD if recent and allreps[-1]["valid"] else BAD if recent else TEXT
            L.text(x0 + 22, y0 + 120, str(good), 92, "Bold", colour, anchor="lm")
            L.text(x0 + 30 + L.width_of(str(good), 92, "Bold"), y0 + 138,
                   f"/ {len(allreps)}", 30, "Medium", MUTED, anchor="lm")
        # a dot per rep
        dots = allreps[-12:]
        for i, r in enumerate(dots):
            cx, cy = x0 + 30 + (i % 6) * 34, y1 - 52 + (i // 6) * 26
            L.circle(cx, cy, 9, fill=GOOD if r["valid"] else BAD)
        if not dots:
            L.text(x0 + 24, y1 - 52, "waiting for rep 1", 15, "Medium", MUTED, anchor="lm")

        # 2) form score ring, counting up as reps finish
        x0, y0, x1, y1 = b
        cx, cy, R = (x0 + x1) / 2, y0 + 140, 72
        L.arc(cx, cy, R, 0, 360, FAINT, 14)
        if allreps:
            prev = np.mean([r["score"] for r in allreps[:-1]]) if len(allreps) > 1 else 0
            target = np.mean([r["score"] for r in allreps])
            shown = prev + (target - prev) * ease((t - allreps[-1]["time_s"]) / 0.6)
            colour = GOOD if shown >= 85 else WARN if shown >= 65 else BAD
            L.arc(cx, cy, R, -90, -90 + 360 * shown / 100, colour, 14)
            end = math.radians(-90 + 360 * shown / 100)
            L.circle(cx + R * math.cos(end), cy + R * math.sin(end), 7, fill=colour)
            L.text(cx, cy - 6, f"{shown:.0f}", 46, "Bold", anchor="mm")
            L.text(cx, cy + 32, "avg / 100", 14, "Medium", MUTED, anchor="mm")
        else:
            L.text(cx, cy, "--", 46, "Bold", MUTED, anchor="mm")

        # 3) depth gauge: needle follows the live knee angle
        x0, y0, x1, y1 = c
        cx, cy, R = (x0 + x1) / 2, y0 + 168, 88
        def pos(angle):                      # 180 deg (standing) left -> 40 deg right
            return 180 + 180 * (180 - np.clip(angle, 40, 180)) / 140
        L.arc(cx, cy, R, 180, 360, FAINT, 16)
        L.arc(cx, cy, R, pos(PARALLEL), 360, with_alpha(GOOD, 0.85), 16)
        L.text(x1 - 24, y0 + 26, "target", 13, "Medium", GOOD, anchor="rm")
        for k, p in enumerate(self.people):
            angle = self.live_angle(p, t)
            if angle is None:
                continue
            th = math.radians(pos(angle))
            tip = (cx + (R - 26) * math.cos(th), cy + (R - 26) * math.sin(th))
            colour = p["rgb"] if self.many else (GOOD if angle <= PARALLEL else TEXT)
            L.line([(cx, cy), tip], colour, 5)
            L.circle(*tip, 6, fill=colour)
            if k == 0:
                L.text(cx, cy + 38, f"{angle:.0f}°", 34, "Bold", anchor="mm")
        L.circle(cx, cy, 10, fill=TEXT)

    # ---- chart (line + glow + gradient fill, drawn straight on the canvas)
    def draw_chart(self, canvas, t, done):
        x0, y0, x1, y1 = self.ax0, self.ay0, self.ax1, self.ay1
        C = Layer(x0 - 20, y0 - 44, x1 - x0 + 40, y1 - y0 + 60)

        # rep bands + numbers
        for k, (p, reps) in enumerate(zip(self.people, done)):
            for r in reps:
                colour = GOOD if r["valid"] else BAD
                appear = ease((t - r["time_s"]) / 0.4)
                bx0, bx1 = self.X(r["start_s"]), self.X(r["time_s"])
                if self.many:
                    ly = y0 + 2 + k * 12
                    C.rrect(bx0, ly, bx1, ly + 8, 4, fill=with_alpha(colour, 0.8 * appear))
                else:
                    C.rrect(bx0, y0, bx1, y1, 10, fill=with_alpha(colour, 0.13 * appear))
                    mx = (bx0 + bx1) / 2
                    C.rrect(mx - 14, y0 - 34, mx + 14, y0 - 8, 8, fill=with_alpha(colour, appear))
                    C.text(mx, y0 - 21, str(r["rep"]), 14, "Bold", with_alpha(BG_TOP, appear),
                           anchor="mm")
        # grid
        for a in (STANDING, PARALLEL):
            y = self.Y(a)
            for xs in np.arange(x0, x1, 16):
                C.line([(xs, y), (min(xs + 7, x1), y)], FAINT, 2, joint=None)
        C.line([(x0, y1), (x1, y1)], FAINT, 2, joint=None)
        C.onto(canvas)

        for p in self.people:
            colour = LINE if not self.many else p["rgb"]
            segments, seg, last_t = [], [], None
            for s_t, a in p["series"]:
                if s_t > t:
                    break
                if last_t is not None and s_t - last_t > 0.25:
                    segments.append(seg)
                    seg = []
                seg.append((self.X(s_t), self.Y(a)))
                last_t = s_t
            segments.append(seg)
            segments = [s for s in segments if len(s) > 1]
            if not segments:
                continue

            # gradient fill under the line
            if not self.many:
                m = Image.new("L", self.grad.size, 0)
                md = ImageDraw.Draw(m)
                for s in segments:
                    poly = [(px - x0, py - y0) for px, py in s]
                    poly += [(s[-1][0] - x0, y1 - y0), (s[0][0] - x0, y1 - y0)]
                    md.polygon(poly, fill=255)
                fill = Image.new("RGBA", self.grad.size, (*colour, 0))
                fill.putalpha(ImageChops.multiply(m, self.grad))
                canvas.alpha_composite(fill, (int(x0), int(y0)))

            # glow, then the crisp line on top
            G = Layer(x0 - 20, y0 - 16, x1 - x0 + 40, y1 - y0 + 32)
            for s in segments:
                G.line(s, with_alpha(colour, 0.55), 12)
            glow = G.img.reduce(SS).filter(ImageFilter.GaussianBlur(9))
            canvas.alpha_composite(glow, (G.x0, G.y0))
            Ln = Layer(x0 - 20, y0 - 16, x1 - x0 + 40, y1 - y0 + 32)
            for s in segments:
                Ln.line(s, colour, 4)
            head = segments[-1][-1]
            if p["series"] and t - last_t < 0.25:
                Ln.circle(*head, 12, fill=with_alpha(colour, 0.3))
                Ln.circle(*head, 6, fill=TEXT)
            # bottom-of-rep dots
            for r in [r for r in p["reps"] if r["time_s"] <= t and "bottom_s" in r]:
                i = int(np.searchsorted(p["_times"], r["bottom_s"]))
                if i < len(p["series"]):
                    bx, by = self.X(p["series"][i][0]), self.Y(p["series"][i][1])
                    Ln.circle(bx, by, 8, fill=GOOD if r["valid"] else BAD, outline=CARD, width=3)
            Ln.onto(canvas)

    # ---- rep chips
    def draw_chips(self, L, t, done):
        bx0, by0, bx1, by1 = self.chipbox
        cols, gap = (6 if bx1 - bx0 > 1200 else 5), 14
        rows = sorted(((p, r) for p, reps in zip(self.people, done) for r in reps),
                      key=lambda pr: pr[1]["time_s"])[-cols * 2:]
        x0, y0 = bx0 + 24, by0 + 66
        cw = (bx1 - bx0 - 48 - gap * (cols - 1)) / cols
        ch = (by1 - 20 - y0 - gap) / 2

        allreps = [r for reps in done for r in reps]
        if allreps:
            tempo = np.mean([r["measured"]["tempo"] for r in allreps])
            L.text(bx1 - 28, by0 + 34, f"avg lowering {tempo:.1f}s",
                   15, "Medium", MUTED, anchor="rm")

        for slot in range(cols * 2):
            cx0 = x0 + (slot % cols) * (cw + gap)
            cy0 = y0 + (slot // cols) * (ch + gap)
            if slot >= len(rows):
                L.rrect(cx0, cy0, cx0 + cw, cy0 + ch, 18, outline=with_alpha(FAINT, 0.7), width=2)
                continue
            p, r = rows[slot]
            a = ease((t - r["time_s"]) / 0.35)
            lift = (1 - a) * 14
            colour = GOOD if r["valid"] else BAD
            score_col = GOOD if r["score"] >= 85 else WARN if r["score"] >= 65 else BAD
            L.rrect(cx0, cy0 + lift, cx0 + cw, cy0 + ch + lift, 18,
                    fill=with_alpha(mix(CARD, colour, 0.16), a),
                    outline=with_alpha(colour, 0.85 * a), width=2)
            tag = f"P{p['person']} · #{r['rep']}" if self.many else f"REP {r['rep']}"
            L.text(cx0 + 16, cy0 + 24 + lift, tag, 13, "SemiBold", with_alpha(MUTED, a), anchor="lm")
            L.text(cx0 + cw / 2, cy0 + ch / 2 + 4 + lift, str(r["score"]), 38, "Bold",
                   with_alpha(score_col if r["valid"] else TEXT, a), anchor="mm")
            word = "GOOD" if r["valid"] else self.ex["tags"].get(r["main_fault"], "NO REP")
            L.text(cx0 + cw / 2, cy0 + ch - 22 + lift, word, 14, "Bold",
                   with_alpha(colour, a), anchor="mm")

    # ---- small overlays on the video
    def draw_view_badge(self, V):
        views = {r["view"] for p in self.people for r in p["reps"]}
        label = (" + ".join(sorted(views)) if views else "side").upper() + " VIEW"
        w = V.width_of(label, 14, "SemiBold") + 32
        V.rrect(self.vx + 18, self.vy + 18, self.vx + 18 + w, self.vy + 52, 17,
                fill=(10, 14, 26, 150))
        V.text(self.vx + 18 + w / 2, self.vy + 35, label, 14, "SemiBold", TEXT, anchor="mm")

    def draw_toast(self, V, t, done):
        recent = [(p, r) for p, reps in zip(self.people, done) for r in reps
                  if 0 <= t - r["time_s"] < 1.5][-2:]
        for k, (p, r) in enumerate(recent):
            age = t - r["time_s"]
            a = ease(age / 0.25) * (1 - ease((age - 1.2) / 0.3))
            colour = GOOD if r["valid"] else BAD
            title = f"Rep {r['rep']}" if r["valid"] else "No rep"
            sub = f"{r['score']} / 100" if r["valid"] else self.ex["focus"].get(r["main_fault"], "")
            if self.many:
                title = f"P{p['person']} · {title}"
            w = max(V.width_of(title, 24, "Bold"), V.width_of(sub, 15, "Medium")) + 84
            x = self.vx + (self.vw - w) / 2
            y = self.vy + self.vh - 120 - k * 92 + (1 - a) * 20
            V.rrect(x, y, x + w, y + 76, 22, fill=(10, 14, 26, int(200 * a)),
                    outline=with_alpha(colour, a), width=2)
            V.circle(x + 34, y + 38, 14, fill=with_alpha(colour, a))
            V.text(x + 60, y + 26, title, 24, "Bold", with_alpha(TEXT, a), anchor="lm")
            V.text(x + 60, y + 54, sub, 15, "Medium", with_alpha(MUTED, a), anchor="lm")

    def draw_summary(self, V, progress):
        """Animated 'Set complete' card over the dimmed last frame."""
        a = ease(progress)
        V.rrect(self.vx, self.vy, self.vx + self.vw, self.vy + self.vh, RADIUS,
                fill=(6, 9, 18, int(170 * a)))
        reps = [r for p in self.people for r in p["reps"]]
        if not reps:
            return
        good = sum(r["valid"] for r in reps)
        best = max(reps, key=lambda r: r["score"])
        faults = Counter(r["main_fault"] for r in reps if r["main_fault"])
        focus = self.ex["focus"].get(faults.most_common(1)[0][0], "") if faults else "Great set. Keep it up!"
        avg = np.mean([r["score"] for r in reps])

        w = min(self.vw - 48, 520)
        h = 400
        x = self.vx + (self.vw - w) / 2
        y = self.vy + (self.vh - h) / 2 + (1 - a) * 60
        A = lambda c, k=1.0: with_alpha(c, a * k)
        V.rrect(x, y, x + w, y + h, 28, fill=A(CARD, 0.97), outline=A(CARD_EDGE), width=2)
        cx = x + w / 2
        V.text(cx, y + 46, "SET COMPLETE", 16, "Bold", A(GOOD), anchor="mm")
        V.text(cx, y + 128, f"{good}/{len(reps)}", 92, "Bold", A(TEXT), anchor="mm")
        V.text(cx, y + 196, "clean reps", 18, "Medium", A(MUTED), anchor="mm")
        # two mini stats
        for i, (label, value, colour) in enumerate((
                ("AVG SCORE", f"{avg:.0f}", GOOD if avg >= 85 else WARN if avg >= 65 else BAD),
                ("BEST REP", f"#{best['rep']} · {best['score']}", TEXT))):
            mx = x + w * (0.27 + 0.46 * i)
            V.text(mx, y + 244, label, 13, "SemiBold", A(MUTED), anchor="mm")
            V.text(mx, y + 276, value, 28, "Bold", A(colour), anchor="mm")
        V.rrect(x + 28, y + h - 88, x + w - 28, y + h - 32, 18,
                fill=A(mix(CARD, WARN if faults else GOOD, 0.18)))
        V.text(cx, y + h - 60, f"Next set: {focus}" if faults else focus, 17, "SemiBold",
               A(WARN if faults else GOOD), anchor="mm")


# ---------------------------------------------------------------- main
def media_duration(path):
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def main(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    report_path = f"output/{name}_reps.json"
    skeleton = f"output/{name}_skeleton.mp4"
    voice = f"output/{name}_voice.m4a"
    out_path = f"output/{name}_dashboard.mp4"
    for p in (report_path, skeleton):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Run Step 3 first: python step3_form_score.py {video_path}")
    with open(report_path) as f:
        report = json.load(f)
    if not report["people"]:
        sys.exit("No person was found in the video.")
    has_voice = os.path.exists(voice)
    if not has_voice:
        print("No voice track found, so the video will be silent. Run Step 4 first for voice.")

    cap = cv2.VideoCapture(skeleton)
    fps = cap.get(cv2.CAP_PROP_FPS) or report["fps"]
    dash = Dashboard(report, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # The summary card stays up while the coach finishes talking: 3-6 seconds
    hold_s = 3.0
    if has_voice:
        hold_s = float(np.clip(media_duration(voice) - report["duration_s"] + 0.6, 3.0, 6.0))

    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{dash.W}x{H}",
           "-r", f"{fps}", "-i", "-"]
    if has_voice:
        # pad the voice with silence so the video (not the audio) decides the length
        cmd += ["-i", voice, "-map", "0:v", "-map", "1:a", "-af", "apad",
                "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium",
            "-shortest", "-movflags", "+faststart", out_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i, last = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        enc.stdin.write(dash.render(frame, i / fps).tobytes())
        i += 1
        if i % 30 == 0:
            print(f"  rendered {i}/{total} frames", end="\r")
    end_t = i / fps
    if last is not None:
        for k in range(int(hold_s * fps)):
            enc.stdin.write(dash.render(last, end_t, end_progress=k / (0.6 * fps)).tobytes())
    cap.release()
    enc.stdin.close()
    enc.wait()
    print(f"\nSaved: {os.path.abspath(out_path)}  ({dash.W}x{H})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python step5_dashboard.py path/to/video.mp4")
    main(sys.argv[1])
