"""
Run the whole AI coach on one video: analysis -> voice -> dashboard.

Usage:
    python run_coach.py videos/squat.mp4                          (squats, up to 2 people)
    python run_coach.py videos/pushups.mp4 --exercise pushup
    python run_coach.py videos/squat.mp4 --people 1
    python run_coach.py videos/squat.mp4 --no-voice
Output:
    output/<name>_dashboard.mp4
"""
import argparse
import json
import os
import subprocess
import sys


def run(*args):
    print("\n>>", " ".join(args))
    if subprocess.run([sys.executable, *args]).returncode != 0:
        sys.exit("Stopped: the step above failed.")


def main():
    ap = argparse.ArgumentParser(description="AI coach: one video in, coached dashboard out.")
    ap.add_argument("video")
    ap.add_argument("--exercise", choices=["squat", "pushup"], default="squat")
    ap.add_argument("--people", type=int, default=2, help="max people to track (default 2)")
    ap.add_argument("--voice", default=None, help="edge-tts voice, e.g. en-GB-RyanNeural")
    ap.add_argument("--no-voice", action="store_true", help="skip the spoken coach")
    ap.add_argument("--no-music", action="store_true", help="drop the clip's own sound")
    a = ap.parse_args()

    analysis = "pushup_form_score.py" if a.exercise == "pushup" else "step3_form_score.py"
    run(analysis, a.video, str(a.people))

    name = os.path.splitext(os.path.basename(a.video))[0]
    with open(f"output/{name}_reps.json") as f:
        report = json.load(f)
    if not any(p["reps"] for p in report["people"]):
        sys.exit(f"\nNo {a.exercise} reps were found, so there's no voice or dashboard to make.\n"
                 "Check the exercise (see the hint above, if any), and that the whole body "
                 "is in frame.")
    if not a.no_voice:
        extra = (["--voice", a.voice] if a.voice else []) + (["--no-music"] if a.no_music else [])
        run("step4_voice_coach.py", a.video, *extra)
    run("step5_dashboard.py", a.video)


if __name__ == "__main__":
    main()
