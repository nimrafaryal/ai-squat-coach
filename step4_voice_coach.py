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

Usage:
    python step4_voice_coach.py videos/squat.mp4
    python step4_voice_coach.py videos/squat.mp4 --voice en-GB-RyanNeural
    python step4_voice_coach.py videos/squat.mp4 --no-music

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

GAP_S = 0.2           # silence kept between two spoken lines
MAX_DELAY_S = 1.2     # a line pushed later than this is dropped (the moment passed)
MUSIC_VOLUME = 0.2    # the original clip's sound, kept quietly under the voice

NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
           "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

# Priority: 1 = nice to have, 2 = rep count, 3 = correction, 4 = always said
SHOUT = {      # said mid-rep, the moment the fault appears
    "lean": ["Chest up!", "Chest up, chest up!", "Stay tall!"],
    "heels": ["Heels down!", "Stay on your heels!", "Heels planted!"],
    "knees": ["Knees out!", "Push those knees out!", "Knees out, knees out!"],
}
FIX = {        # the reason, when it wasn't already shouted
    "depth": ["You need to go lower.", "Hips below your knees.", "Sit deeper next time."],
    "lean": ["Keep your chest up.", "Don't fold forward."],
    "heels": ["Keep your heels down.", "Weight back in your heels."],
    "knees": ["Push your knees out.", "Knees over your toes."],
    "tempo": ["Too rushed, slow it down.", "Control it on the way down."],
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


def pick(options, n):
    """Rotate through phrasings so the coach never repeats itself back to back."""
    return options[n % len(options)]


def say_number(n):
    return NUMBERS[n] if n < len(NUMBERS) else str(n)


# ---------------------------------------------------------------- the script
def lines_for_person(person, many, counter):
    """What the coach says to one person, and when."""
    who = f"Person {person['person']}, " if many else ""
    lines, good, streak, deepest = [], 0, 0, None
    first_descent = None
    last_tip = {}                 # fault -> the good-rep number it was last mentioned on

    for rep in person["reps"]:
        m = rep["measured"]
        if first_descent is None:
            first_descent = m["tempo"]

        # 1) Mid-rep shout, at the moment the first fault appeared
        shouted = None
        for lf in rep.get("live_faults", []):
            if lf["fault"] in SHOUT:
                shouted = lf["fault"]
                lines.append({"time_s": lf["time_s"], "priority": 3,
                              "text": who + pick(SHOUT[shouted], counter["shout"])})
                counter["shout"] += 1
                break

        # 2) End of rep
        fault = rep["main_fault"]
        if not rep["valid"]:
            streak = 0
            if shouted and shouted == fault:
                text = pick(NO_REP, counter["norep"])          # reason already given
            else:
                text = f"{pick(NO_REP, counter['norep'])} {pick(FIX[fault], counter['fix'])}"
                counter["fix"] += 1
            counter["norep"] += 1
            lines.append({"time_s": rep["time_s"], "priority": 3, "text": who + text})
            continue

        good += 1
        streak = streak + 1 if rep["score"] >= 85 else 0
        count = say_number(good)
        count = count if who else count.capitalize()
        depth = m["depth"]

        if fault == "tempo" and good > 2 and m["tempo"] < first_descent * 0.7:
            text = f"{count}. You're speeding up, control it."
            priority = 3
        elif fault and fault != shouted and good - last_tip.get(fault, -9) >= 3:
            last_tip[fault] = good                # don't nag: once every 3 reps at most
            text = f"{count}, {pick(TIP[fault], counter['tip'])}"
            counter["tip"] += 1
            priority = 2
        elif deepest is not None and depth < deepest - 5:
            text = f"{count}. Deepest one yet!"
            priority = 2
        elif streak == 3:
            text = f"{count}. Three clean in a row!"
            priority = 2
        elif good % 2 == 0:                       # every other rep: just the count
            text = f"{count}."
            priority = 2
        else:
            text = f"{count}. {pick(PRAISE, counter['praise'])}"
            counter["praise"] += 1
            priority = 2
        if many and priority == 2:
            priority = 1                          # two people: corrections matter more
        deepest = depth if deepest is None else min(deepest, depth)
        lines.append({"time_s": rep["time_s"], "priority": priority, "text": who + text})
    return lines


def summary_line(report):
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
        text += f" Next set, focus on {FOCUS[faults.most_common(1)[0][0]]}."
    elif avg and min(avg) >= 90:
        text += " That was a great set."
    return text


def write_lines(report):
    people = [p for p in report["people"] if p["reps"]]
    if not people:
        return []
    many = len(people) > 1
    counter = Counter()
    lines = []
    for p in people:
        lines += lines_for_person(p, many, counter)

    first = min(r["start_s"] for p in people for r in p["reps"])
    if first >= 1.2:
        lines.append({"time_s": max(0.0, first - 1.2), "priority": 1,
                      "text": "Alright, let's go."})
    last = max(r["time_s"] for p in people for r in p["reps"])
    lines.append({"time_s": last + 0.5, "priority": 4, "text": summary_line(report)})
    return sorted(lines, key=lambda l: l["time_s"])


# ---------------------------------------------------------------- speech
async def synthesise(lines, voice, folder):
    os.makedirs(folder, exist_ok=True)
    for i, line in enumerate(lines):
        path = os.path.join(folder, f"line_{i:03d}.mp3")
        await edge_tts.Communicate(line["text"], voice, rate="+8%").save(path)
        line["audio"] = path
        line["duration_s"] = media_info(path)["duration"]
        print(f"  [{line['time_s']:6.2f}s] {line['text']}")


def media_info(path):
    """Duration and whether there's a sound track, read from ffmpeg's output."""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    return {"duration": duration, "has_audio": "Audio:" in out}


def schedule(lines, video_duration):
    """Lines never talk over each other. Low-priority lines give way to a
    correction; a line that would come too late is dropped, except
    corrections and the summary."""
    lines = sorted(lines, key=lambda l: l["time_s"])

    def blocks_more_important(l):
        # only a correction (priority 3) cancels a line; the summary can just wait
        return any(o is not l and o["priority"] == 3 > l["priority"]
                   and l["time_s"] <= o["time_s"] <= l["time_s"] + l["duration_s"] + GAP_S
                   for o in lines)
    for l in [l for l in lines if l["priority"] < 3 and blocks_more_important(l)]:
        print(f"  skipped (something more important is due): {l['text']}")
    lines = [l for l in lines if not (l["priority"] < 3 and blocks_more_important(l))]

    kept, free_at = [], 0.0
    for line in lines:
        start = max(line["time_s"], free_at)
        if start - line["time_s"] > MAX_DELAY_S and line["priority"] < 3:
            print(f"  skipped (too late): {line['text']}")
            continue
        if start >= video_duration and line["priority"] < 4:
            continue
        line["start_s"] = start
        free_at = start + line["duration_s"] + GAP_S
        kept.append(line)
    return kept


# ---------------------------------------------------------------- mixing
def mix_voice_track(lines, source_video, out_path, keep_music):
    """All voice lines at their moments, plus the clip's own sound, quietly."""
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for line in lines:
        cmd += ["-i", line["audio"]]
    music = keep_music and media_info(source_video)["has_audio"]
    if music:
        cmd += ["-i", source_video]
    parts, labels = [], []
    for i, line in enumerate(lines):
        ms = int(line["start_s"] * 1000)
        parts.append(f"[{i}:a]adelay={ms}|{ms},volume=1.6[v{i}]")
        labels.append(f"[v{i}]")
    if music:
        parts.append(f"[{len(lines)}:a]volume={MUSIC_VOLUME}[m]")
        labels.append("[m]")
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:"
                 f"duration=longest[aout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[aout]",
            "-c:a", "aac", "-b:a", "160k", out_path]
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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Add the voice coach to a Step 3 video.")
    ap.add_argument("video", help="the ORIGINAL video you gave Step 3, e.g. videos/squat.mp4")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--no-music", action="store_true", help="drop the clip's own sound")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    form_video, report_path = f"output/{name}_form.mp4", f"output/{name}_reps.json"
    for p in (form_video, report_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Run Step 3 first: python step3_form_score.py {args.video}")
    with open(report_path) as f:
        report = json.load(f)
    if "live_faults" not in str(report):
        print("Note: this report is from an older Step 3. Re-run Step 3 for mid-rep cues.")

    lines = write_lines(report)
    if not lines:
        sys.exit("No reps were found in the report, so there's nothing to say.")

    print(f"Generating {len(lines)} voice lines with {args.voice}...")
    try:
        asyncio.run(synthesise(lines, args.voice, f"output/{name}_voice"))
    except Exception as e:
        sys.exit(f"Voice generation failed ({e}). Check your internet connection.")

    lines = schedule(lines, media_info(form_video)["duration"])
    voice_path, out_path = f"output/{name}_voice.m4a", f"output/{name}_coached.mp4"
    print("Mixing...")
    mix_voice_track(lines, args.video, voice_path, keep_music=not args.no_music)
    mux(form_video, voice_path, out_path)
    print(f"\nSaved voice track: {os.path.abspath(voice_path)}")
    print(f"Saved video:       {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
