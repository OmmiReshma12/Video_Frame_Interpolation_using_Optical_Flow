"""
Runtime benchmark: how long does it take to interpolate ONE frame at different image resolutions?

This mirrors the running-time experiment of the base paper (Meyer et al., "Phase-Based Frame
Interpolation for Video", CVPR 2015, Table 1 / Figure 9), so you can discuss speed and scaling.

IMPORTANT when comparing with the paper: its numbers were measured on a 2015 desktop (Core i7 3.4 GHz) with
different implementations, so only the SHAPE of the curve (how time grows with resolution) is comparable,
not the absolute seconds. Always state your own CPU.

Usage:
  python benchmark_runtime.py
  python benchmark_runtime.py --input UCF-101\\Diving\\v_Diving_g01_c01.avi --repeats 5
  python benchmark_runtime.py --sizes 320x240 640x360 1280x720 1920x1080

Output (in --out_dir): runtime_table.md, runtime.csv, runtime_loglog.png
"""
import argparse
import csv
import os
import platform
import statistics
import time

import cv2
import numpy as np

from flowinterp import FlowPair, interpolate_at


def get_frames(path):
    """Two consecutive frames from a video, or a synthetic moving texture if no video is given."""
    if path:
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < 12:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 12:
            raise SystemExit("Video too short")
        return frames[10], frames[11]
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (1200, 2200, 3), dtype=np.uint8), (0, 0), 3)
    return bg[40:40 + 1080, 40:40 + 1920].copy(), bg[44:44 + 1080, 46:46 + 1920].copy()


def timed(fn, repeats):
    fn()                                    # warm-up (not counted)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", default="", help="optional video; frames are resized to each test size")
    ap.add_argument("--sizes", nargs="+", default=["320x240", "640x360", "1280x720"],
                    help="WIDTHxHEIGHT list (add 1920x1080 if you have time; Farneback gets slow)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out_dir", default="results_runtime")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    a0, b0 = get_frames(args.input)
    rows = []
    for s in args.sizes:
        w, h = (int(v) for v in s.lower().split("x"))
        a = cv2.resize(a0, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b0, (w, h), interpolation=cv2.INTER_AREA)
        res = {"size": f"{w}x{h}", "pixels": w * h}
        for method in ("dis", "farneback"):
            def run(m=method):
                interpolate_at(FlowPair(a, b, m), 0.5)   # both flows + warp + blend = one interpolated frame
            res[method] = timed(run, args.repeats)
        rows.append(res)
        print(f"{res['size']:>10}:  DIS {res['dis']:.3f} s   Farneback {res['farneback']:.3f} s")

    with open(os.path.join(args.out_dir, "runtime.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["size", "pixels", "dis_seconds", "farneback_seconds"])
        for r in rows:
            wr.writerow([r["size"], r["pixels"], f"{r['dis']:.4f}", f"{r['farneback']:.4f}"])

    cpu = platform.processor() or platform.machine()
    lines = [f"Time to interpolate one frame (median of {args.repeats} runs), full pipeline: "
             "flow both ways + warping + occlusion-aware blending.",
             f"Machine: {platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} logical cores, "
             f"OpenCV {cv2.__version__}.", "",
             "| Resolution | DIS pipeline (s) | Farneback pipeline (s) |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['size']} | {r['dis']:.3f} | {r['farneback']:.3f} |")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "runtime_table.md"), "w") as f:
        f.write(text + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 4))
    px = [r["pixels"] for r in rows]
    ax.loglog(px, [r["dis"] for r in rows], "o-", label="DIS pipeline")
    ax.loglog(px, [r["farneback"] for r in rows], "s-", label="Farneback pipeline")
    ax.set_xlabel("pixels per frame")
    ax.set_ylabel("seconds per interpolated frame")
    ax.set_xticks(px, [r["size"] for r in rows], rotation=25)
    ax.minorticks_off()
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "runtime_loglog.png"), dpi=150)
    print(f"\nSaved to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()