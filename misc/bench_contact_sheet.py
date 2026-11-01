"""Benchmark behind ``ONE_PASS_SHOTS_PER_MINUTE`` (reeleehq/mixing#94).

Times :func:`mixing.video.thumbnail.grab_frames` with ``method="seek"`` and
``method="pass"`` on one long video at several shot densities, so the
crossover can be re-measured on other material, resolutions or ffmpeg builds.

Usage::

    python misc/bench_contact_sheet.py VIDEO --minutes 30 --densities 1,4,15,60

Prints one tab-separated row per density: shots per minute, frame count,
seconds by seeking, seconds by one pass.
"""

import argparse
import time

from mixing.video.thumbnail import grab_frames

THUMB_WIDTH = 240


def _timed(video, times, method) -> float:
    began = time.perf_counter()
    grab_frames(video, times, width=THUMB_WIDTH, method=method)
    return time.perf_counter() - began


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--densities", default="1,4,15,60")
    args = parser.parse_args()
    duration = args.minutes * 60
    print("shots_per_min\tN\tseek_s\tone_pass_s")
    for density in map(float, args.densities.split(",")):
        n = max(1, round(density * args.minutes))
        times = [(i + 0.5) * duration / n for i in range(n)]
        seek, one_pass = (_timed(args.video, times, m) for m in ("seek", "pass"))
        print(f"{density:g}\t{n}\t{seek:.1f}\t{one_pass:.1f}", flush=True)


if __name__ == "__main__":
    main()
