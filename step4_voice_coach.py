"""
Step 4 - Voice coach. Talks like a trainer instead of reading the screen:
  - shouts corrections DURING the rep, the moment a fault shows up ("Chest up!")
  - counts good reps in rhythm, and notices patterns ("Deepest one yet!",
    "Three clean in a row!", "You're speeding up, control it")
  - says "no rep" briefly when it already told you why
  - closes with a set summary and the one thing to work on

Reads what Step 3 made:
    output/<name>_form.mp4     the annotated video
    output/<name>_reps.json    reps, scores, faults and their timings
Writes:
    output/<name>_voice.m4a    the coach's voice track (Step 5 uses it)
    output/<name>_coached.mp4  the annotated video with the voice (H.264 + AAC)

Timing: every clip is trimmed of its built-in silence, counts start as the
arms/legs lock out, and nothing is ever queued late: when reps come fast the
coach just says the number, and a line that would land late is dropped.

Usage:
    python step4_voice_coach.py videos/squat.mp4
    python step4_voice_coach.py videos/squat.mp4 --voice en-US-BrianNeural
    python step4_voice_coach.py videos/squat.mp4 --no-music
    python step4_voice_coach.py --sample        (hear 6 voices, then pick one)

Needs internet for the voice (Microsoft's free neural voices via edge-tts).
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter

import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_VOICE = "en-US-AndrewNeural"   # warm, energetic male coach
# Others: en-US-AriaNeural (female), en-GB-RyanNeural (British), en-US-GuyNeural,
#         en-AU-WilliamNeural. See all with: edge-tts --list-voices

GAP_S = 0.12          # silence kept between two spoken lines
COUNT_LEAD_S = 0.10   # counts start this much before the rep finishes, so they land on it
# How late (seconds) a line may start before it's dropped instead: by priority
LATE_OK = {1: 0.25, 2: 0.35, 3: 0.70, 4: 99.0}
MUSIC_VOLUME = 0.35   # the clip's own sound; it is also ducked whenever the coach talks

# How each kind of line is spoken: shouts fast and bright, counts crisp, summary calm
STYLES = {
    "shout": {"rate": "+14%", "pitch": "+6Hz", "volume": "+10%"},
    "count": {"rate": "+6%", "pitch": "+3Hz", "volume": "+5%"},
    "say":   {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
}
SAMPLE_VOICES = ["en-US-AndrewNeural", "en-US-BrianNeural", "en-US-GuyNeural",
                 "en-US-ChristopherNeural", "en-GB-RyanNeural", "en-US-AriaNeural"]

NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
           "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

# Priority: 1 = nice to have, 2 = rep count, 3 = correction, 4 = always said
SHOUT = {      # said mid-rep, the moment the fault appears
    "lean": ["Chest up!", "Chest up, chest up!", "Stay tall!"],
    "heels": ["Heels down!", "Stay on your heels!", "Heels planted!"],
    "knees": ["Knees out!", "Push the knees out!", "Knees out, knees out!"],
}
FIX = {        # the reason, when it wasn't already shouted
    "depth": ["Go lower.", "Hips below knees.", "Sit deeper."],
    "lean": ["Chest up.", "Don't fold forward."],
    "heels": ["Heels down.", "Stay on your heels."],
    "knees": ["Knees out.", "Knees over toes."],
    "tempo": ["Too rushed.", "Slow it down."],
    None: ["Not quite there."],
}
NO_REP = ["That one doesn't count.", "No rep.", "Not that one."]
TIP = {        # a valid rep that still lost some points
    "depth": ["try to sit a little deeper.", "a bit lower next one."],
    "lean": ["keep that chest proud.", "stay a little taller."],
    "heels": ["weight in your heels.", "keep those heels planted."],
    "knees": ["keep your knees over your toes.", "drive the knees out."],
    "tempo": ["slow it down on the way down.", "take your time going down.",
              "control the descent."],
}
PRAISE = ["Nice.", "Good.", "That's it.", "Strong.", "Beautiful.", "Yes!"]
FOCUS = {
    "depth": "getting a little deeper", "lean": "keeping your chest up",
    "heels": "keeping your heels down", "knees": "pushing your knees out",
    "tempo": "slowing down on the way down",
}

# ---- push-ups: same structure, different words
PUSHUP_SHOUT = {
    "sag": ["Hips up!", "Squeeze your glutes!", "Hips up, hips up!"],
    "pike": ["Hips down!", "Straight line!", "Flatten out!"],
    "head": ["Head up!", "Chest first!", "Head in line!"],
}
PUSHUP_FIX = {
    "depth": ["Chest lower.", "All the way down.", "Go lower."],
    "sag": ["Hips up.", "Squeeze your glutes."],
    "pike": ["Hips down.", "Straight line."],
    "head": ["Chest first, not your head."],
    "tempo": ["Too rushed.", "Slow it down."],
    "lockout": ["Lock out at the top."],
    None: ["Not quite there."],
}
PUSHUP_TIP = {
    "depth": ["a little lower next one.", "get that chest closer to the floor."],
    "sag": ["keep those hips up.", "squeeze your glutes."],
    "pike": ["hips down a touch.", "keep a straight line."],
    "head": ["keep your head in line.", "lead with the chest."],
    "tempo": ["slow it down on the way down.", "control the descent."],
    "lockout": ["finish with straight arms.", "lock out at the top."],
}
PUSHUP_FOCUS = {
    "depth": "getting your chest lower", "sag": "keeping your hips up",
    "pike": "keeping a straight body line", "head": "keeping your head in line",
    "tempo": "slowing down on the way down", "lockout": "locking out at the top",
}


def phrases(report):
    """The words the coach uses, for the exercise in the report."""
    if report.get("exercise") == "pushup":
        return {"shout": PUSHUP_SHOUT, "fix": PUSHUP_FIX, "tip": PUSHUP_TIP,
                "focus": PUSHUP_FOCUS, "unit": "push-ups"}
    return {"shout": SHOUT, "fix": FIX, "tip": TIP, "focus": FOCUS, "unit": "reps"}


def pick(options, n):
    """Rotate through phrasings so the coach never repeats itself back to back."""
    return options[n % len(options)]


def say_number(n):
    return NUMBERS[n] if n < len(NUMBERS) else str(n)


# ---------------------------------------------------------------- the script
# Which scoring rule each named fault belongs to (push-up sag/pike are both "body")
RULE_OF = {"sag": "body", "pike": "body"}


def shout_for(rep, ph):
    """The one mid-rep shout worth making, or None. Only the fault that actually
    decided the rep is shouted, so the coach never says "Flatten out!" about a
    rep it then calls clean, or shouts one fault and fails the rep for another."""
    fault = rep["main_fault"]
    if fault not in ph["shout"]:
        return None
    lost = rep["points_lost"].get(RULE_OF.get(fault, fault), 0)
    if rep["valid"] and lost < 5:
        return None
    return next((lf for lf in rep.get("live_faults", []) if lf["fault"] == fault), None)


def lines_for_person(person, many, counter, ph):
    """What the coach says to one person, and when. Every line has a target
    time; count lines also have a short version used when reps come fast."""
    who = f"Person {person['person']}, " if many else ""
    lines, good, streak, deepest = [], 0, 0, None
    first_descent = None
    last_tip = {}                 # fault -> the good-rep number it was last mentioned on

    for rep in person["reps"]:
        m = rep["measured"]
        if first_descent is None:
            first_descent = m["tempo"]
        fault = rep["main_fault"]

        # 1) Mid-rep shout, the moment the deciding fault appears
        lf = shout_for(rep, ph)
        if lf:
            lines.append({"time_s": lf["time_s"], "priority": 3, "style": "shout",
                          "text": who + pick(ph["shout"][fault], counter["shout"])})
            counter["shout"] += 1

        # 2) No rep: short if the reason was already shouted
        if not rep["valid"]:
            streak = 0
            if lf:
                text = "No rep."
            else:
                text = f"No rep. {pick(ph['fix'][fault], counter['fix'])}"
                counter["fix"] += 1
            lines.append({"time_s": rep["time_s"], "priority": 3, "style": "say",
                          "text": who + text, "short": who + "No rep."})
            continue

        # 3) Good rep: the count, plus a comment only if there's time for it
        good += 1
        streak = streak + 1 if rep["score"] >= 85 else 0
        count = say_number(good)
        count = count if who else count.capitalize()
        depth = m["depth"]
        short = f"{who}{count}."

        if lf:                                    # just shouted at: count, no praise
            text, priority = short, 2
        elif fault == "tempo" and good > 2 and m["tempo"] < first_descent * 0.7:
            text, priority = f"{who}{count}. Slow it down.", 3
        elif fault in ph["tip"] and good - last_tip.get(fault, -9) >= 3:
            last_tip[fault] = good                # don't nag: once every 3 reps at most
            text, priority = f"{who}{count}, {pick(ph['tip'][fault], counter['tip'])}", 2
            counter["tip"] += 1
        elif deepest is not None and depth < deepest - 5:
            text, priority = f"{who}{count}. Deepest one yet!", 2
        elif streak == 3:
            text, priority = f"{who}{count}. Three clean in a row!", 2
        elif good % 2 == 0:
            text, priority = short, 2
        else:
            text, priority = f"{who}{count}. {pick(PRAISE, counter['praise'])}", 2
            counter["praise"] += 1
        if many and priority == 2:
            priority = 1                          # two people: corrections matter more
        deepest = depth if deepest is None else min(deepest, depth)
        lines.append({"time_s": max(0.0, rep["time_s"] - COUNT_LEAD_S), "priority": priority,
                      "style": "count", "text": text, "short": short})
    return lines


def summary_line(report):
    ph = phrases(report)
    people = report["people"]
    faults = Counter(r["main_fault"] for p in people for r in p["reps"] if r["main_fault"])
    if len(people) == 1:
        reps = people[0]["reps"]
        good = sum(r["valid"] for r in reps)
        text = f"Set done. {say_number(good).capitalize()} clean out of {say_number(len(reps))}."
    else:
        parts = [f"Person {p['person']}, {say_number(sum(r['valid'] for r in p['reps']))} "
                 f"out of {say_number(len(p['reps']))}." for p in people]
        text = "Set done. " + " ".join(parts)
    avg = [p["average_score"] for p in people if p["average_score"] is not None]
    if faults:
        text += f" Next set, focus on {ph['focus'][faults.most_common(1)[0][0]]}."
    elif avg and min(avg) >= 90:
        text += " That was a great set."
    return text


def write_lines(report):
    people = [p for p in report["people"] if p["reps"]]
    if not people:
        return []
    many = len(people) > 1
    counter = Counter()
    ph = phrases(report)
    lines = []
    for p in people:
        lines += lines_for_person(p, many, counter, ph)

    first = min(r["start_s"] for p in people for r in p["reps"])
    if first >= 1.2:
        lines.append({"time_s": max(0.0, first - 1.2), "priority": 1, "style": "shout",
                      "text": "Let's go!"})
    last = max(r["time_s"] for p in people for r in p["reps"])
    lines.append({"time_s": last + 0.6, "priority": 4, "style": "say",
                  "text": summary_line(report)})
    return sorted(lines, key=lambda l: l["time_s"])


# ---------------------------------------------------------------- speech
async def tts_clip(text, voice, style, folder, cache):
    """Speak one line, trim the silence edge-tts leaves around it, and return
    (wav path, duration). Identical lines are only generated once."""
    key = (text, style)
    if key in cache:
        return cache[key]
    i = len(cache)
    mp3, wav = os.path.join(folder, f"raw_{i:03d}.mp3"), os.path.join(folder, f"line_{i:03d}.wav")
    await edge_tts.Communicate(text, voice, **STYLES[style]).save(mp3)
    trim = ("silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.02,"
            "areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.06,"
            "areverse")
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", mp3,
                    "-af", trim, "-ar", "48000", "-ac", "1", wav], check=True)
    cache[key] = (wav, media_info(wav)["duration"])
    return cache[key]


async def synthesise(lines, voice, folder):
    os.makedirs(folder, exist_ok=True)
    cache = {}
    for line in lines:
        line["audio"], line["duration_s"] = await tts_clip(
            line["text"], voice, line["style"], folder, cache)
        if line.get("short") and line["short"] != line["text"]:
            line["short_audio"], line["short_duration_s"] = await tts_clip(
                line["short"], voice, line["style"], folder, cache)
        print(f"  [{line['time_s']:6.2f}s] {line['text']}")


def media_info(path):
    """Duration and whether there's a sound track, read from ffmpeg's output."""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    return {"duration": duration, "has_audio": "Audio:" in out}


def use_short(line):
    line["text"], line["audio"], line["duration_s"] = (
        line["short"], line["short_audio"], line["short_duration_s"])


def schedule(lines, video_duration):
    """Place every line on the timeline without overlaps and without drift.
    - A long line is swapped for its short version if it would run into the
      next line's moment.
    - A count or comment that would run into a correction is dropped.
    - A line that would start late is dropped rather than queued (corrections
      may be a bit later; the summary always plays)."""
    lines = sorted(lines, key=lambda l: l["time_s"])
    kept, free_at = [], 0.0
    for i, line in enumerate(lines):
        start = max(line["time_s"], free_at)
        if start - line["time_s"] > LATE_OK[line["priority"]]:
            print(f"  dropped (would be {start - line['time_s']:.1f}s late): {line['text']}")
            continue
        if start >= video_duration and line["priority"] < 4:
            continue
        later = [o for o in lines[i + 1:] if o["priority"] >= 2]
        nxt = later[0] if later else None
        if "short_audio" in line and nxt:
            # corrections, and anything just before the summary, may push the next
            # line back a little rather than lose their words
            room = 0.5 if nxt["priority"] == 4 else 0.25 if line["priority"] >= 3 else 0.0
            if start + line["duration_s"] + GAP_S > nxt["time_s"] + room:
                use_short(line)
        fixes = [o for o in lines[i + 1:] if o["priority"] == 3]   # summary can wait; fixes can't
        if line["priority"] < 3 and fixes:
            clash = lambda: start + line["duration_s"] + GAP_S > fixes[0]["time_s"] + 0.1
            if clash() and "short_audio" in line:
                use_short(line)
            if clash():
                print(f"  dropped (a correction is due): {line['text']}")
                continue
        line["start_s"] = start
        free_at = start + line["duration_s"] + GAP_S
        kept.append(line)
    return kept


# ---------------------------------------------------------------- mixing
def mix_voice_track(lines, source_video, out_path, keep_music):
    """Voice lines at their moments, cleaned up (low-cut, gentle compression),
    the clip's own sound ducked under the voice, and the whole thing levelled
    to a standard loudness. 48 kHz stereo AAC."""
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for line in lines:
        cmd += ["-i", line["audio"]]
    music = keep_music and media_info(source_video)["has_audio"]
    if music:
        cmd += ["-i", source_video]
    parts = []
    for i, line in enumerate(lines):
        ms = int(round(line["start_s"] * 1000))
        parts.append(f"[{i}:a]adelay={ms}:all=1[v{i}]")
    voices = "".join(f"[v{i}]" for i in range(len(lines)))
    parts.append(f"{voices}amix=inputs={len(lines)}:normalize=0:duration=longest,"
                 "highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=5:release=150:makeup=2"
                 "[vox]")
    if music:
        parts.append("[vox]asplit=2[voxa][voxsc]")
        parts.append(f"[{len(lines)}:a]aresample=48000,aformat=channel_layouts=mono,"
                     f"volume={MUSIC_VOLUME}[mus]")
        parts.append("[mus][voxsc]sidechaincompress=threshold=0.02:ratio=10:attack=10:"
                     "release=400[duck]")
        parts.append("[voxa][duck]amix=inputs=2:normalize=0:duration=longest[mix]")
        last = "[mix]"
    else:
        last = "[vox]"
    parts.append(f"{last}loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
                 "aformat=channel_layouts=stereo[aout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[aout]",
            "-c:a", "aac", "-b:a", "192k", out_path]
    subprocess.run(cmd, check=True)


def mux(video, audio, out_path):
    """Video + voice track. If the summary runs past the end of the clip,
    the last frame is held so it isn't cut off."""
    extra = max(0.0, media_info(audio)["duration"] - media_info(video)["duration"]) + 0.5
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", audio,
           "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.2f}[v]",
           "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-preset", "medium", "-c:a", "copy", "-shortest",
           "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- voice samples
SAMPLE_SCRIPT = [("Let's go!", "shout"), ("One. Nice.", "count"), ("Two.", "count"),
                 ("Hips down!", "shout"), ("No rep.", "say"), ("Three. Strong.", "count"),
                 ("Set done. Seven clean out of nine. Next set, focus on keeping a "
                  "straight body line.", "say")]


async def make_samples(folder="output/voice_samples"):
    """The same short coaching script in several voices, to choose from."""
    os.makedirs(folder, exist_ok=True)
    for voice in SAMPLE_VOICES:
        cache, clips = {}, []
        for text, style in SAMPLE_SCRIPT:
            clips.append((await tts_clip(text, voice, style, os.path.join(folder, voice),
                                         cache))[0])
        out = os.path.join(folder, f"{voice}.m4a")
        inputs = sum((["-i", c] for c in clips), [])
        gaps = "".join(f"[{i}:a]apad=pad_dur=0.45[a{i}];" for i in range(len(clips)))
        cat = "".join(f"[a{i}]" for i in range(len(clips)))
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *inputs,
                        "-filter_complex", f"{gaps}{cat}concat=n={len(clips)}:v=0:a=1,"
                        "loudnorm=I=-16:TP=-1.5[o]", "-map", "[o]", "-c:a", "aac",
                        "-b:a", "192k", out], check=True)
        print(f"  {out}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Add the voice coach to a Step 3 video.")
    ap.add_argument("video", nargs="?", help="the ORIGINAL video you gave Step 3")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--no-music", action="store_true", help="drop the clip's own sound")
    ap.add_argument("--sample", action="store_true",
                    help="save a short sample in several voices to output/voice_samples")
    args = ap.parse_args()

    if args.sample:
        print("Making voice samples...")
        asyncio.run(make_samples())
        print("Listen, then use your favourite with --voice <name>.")
        return
    if not args.video:
        ap.error("give a video, or use --sample")

    name = os.path.splitext(os.path.basename(args.video))[0]
    form_video, report_path = f"output/{name}_form.mp4", f"output/{name}_reps.json"
    for p in (form_video, report_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Run Step 3 first: python step3_form_score.py {args.video}")
    with open(report_path) as f:
        report = json.load(f)

    lines = write_lines(report)
    if not lines:
        sys.exit("No reps were found in the report, so there's nothing to say.")

    print(f"Generating {len(lines)} voice lines with {args.voice}...")
    try:
        asyncio.run(synthesise(lines, args.voice, f"output/{name}_voice"))
    except Exception as e:
        sys.exit(f"Voice generation failed ({e}). Check your internet connection.")

    lines = schedule(lines, media_info(form_video)["duration"])
    print("\nFinal timing:")
    for l in lines:
        print(f"  {l['start_s']:6.2f}s  {l['text']}")
    voice_path, out_path = f"output/{name}_voice.m4a", f"output/{name}_coached.mp4"
    print("Mixing...")
    mix_voice_track(lines, args.video, voice_path, keep_music=not args.no_music)
    mux(form_video, voice_path, out_path)
    print(f"\nSaved voice track: {os.path.abspath(voice_path)}")
    print(f"Saved video:       {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()