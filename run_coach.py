"""
Run the whole AI Squat Coach on one video: analysis -> voice -> dashboard.

Usage:
    python run_coach.py videos/squat.mp4            (up to 2 people)
    python run_coach.py videos/squat.mp4 --people 1
    python run_coach.py videos/squat.mp4 --no-voice
Output:
    output/<name>_dashboard.mp4
"""
import argparse
import subprocess
import sys


def run(*args):
    print("\n>>", " ".join(args))
    if subprocess.run([sys.executable, *args]).returncode != 0:
        sys.exit("Stopped: the step above failed.")


def main():
    ap = argparse.ArgumentParser(description="AI Squat Coach: one video in, coached dashboard out.")
    ap.add_argument("video")
    ap.add_argument("--people", type=int, default=2, help="max people to track (default 2)")
    ap.add_argument("--voice", default=None, help="edge-tts voice, e.g. en-GB-RyanNeural")
    ap.add_argument("--no-voice", action="store_true", help="skip the spoken coach")
    ap.add_argument("--no-music", action="store_true", help="drop the clip's own sound")
    a = ap.parse_args()

    run("step3_form_score.py", a.video, str(a.people))
    if not a.no_voice:
        extra = (["--voice", a.voice] if a.voice else []) + (["--no-music"] if a.no_music else [])
        run("step4_voice_coach.py", a.video, *extra)
    run("step5_dashboard.py", a.video)


if __name__ == "__main__":
    main()
