"""Benchmark for reeleehq/mixing#94: contact-sheet frames by N seeks vs one select pass.

Usage: python bench.py VIDEO [--minutes M]
Prints one row per shot density: shots/min, N, seek seconds, one-pass seconds.
"""

import argparse
import subprocess
import sys
import time

from mixing.util import ffmpeg_exe

THUMB_W = 240


def seeks(video, times):
    for t in times:
        subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
             "-vf", f"scale={THUMB_W}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, check=True,
        )


def one_pass(video, times):
    expr = "+".join(f"gte(t,{t:.3f})*lt(prev_pts*TB,{t:.3f})" for t in times)
    script = f"select='{expr}',scale={THUMB_W}:-2"
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", video, "-filter_script:v", "/dev/stdin",
         "-fps_mode", "passthrough", "-f", "image2pipe", "-vcodec", "png", "-"],
        input=script.encode(), capture_output=True, check=True,
    )
    return proc.stdout.count(b"\x89PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--densities", default="1,4,15,60")
    a = ap.parse_args()
    duration = a.minutes * 60
    print("shots_per_min\tN\tseek_s\tone_pass_s\tframes_out")
    for d in map(float, a.densities.split(",")):
        n = max(1, int(round(d * a.minutes)))
        times = [(i + 0.5) * duration / n for i in range(n)]
        t0 = time.perf_counter(); seeks(a.video, times); ts = time.perf_counter() - t0
        t0 = time.perf_counter(); got = one_pass(a.video, times); tp = time.perf_counter() - t0
        print(f"{d:g}\t{n}\t{ts:.1f}\t{tp:.1f}\t{got}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
