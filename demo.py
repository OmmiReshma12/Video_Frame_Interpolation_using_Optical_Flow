"""
Make a slow-motion / higher-frame-rate version of any video using optical-flow interpolation.

Examples
  python demo.py --input UCF-101/Archery/v_Archery_g01_c01.avi --factor 2
  python demo.py --input some_clip.avi --factor 4 --method farneback --compare

Outputs (in --out_dir):
  <name>_x<factor>.mp4        interpolated video, fps multiplied by <factor>
  <name>_compare.mp4          (--compare) side-by-side: original (duplicated frames) | cross-fade | optical flow
  <name>_flow_preview.png     flow visualisation for the first frame pair
"""
import argparse
import os

import cv2
import numpy as np

from flowinterp import (FlowPair, flow_to_color, interpolate_at,
                        interpolate_average, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="demo_output")
    ap.add_argument("--factor", type=int, default=2, help="frame-rate multiplier (2 = one new frame between each pair)")
    ap.add_argument("--method", default="dis", choices=["dis", "farneback"])
    ap.add_argument("--max_frames", type=int, default=0, help="only process the first N input frames (0 = all)")
    ap.add_argument("--compare", action="store_true", help="also write a side-by-side comparison video")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or (args.max_frames and len(frames) >= args.max_frames):
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 2:
        raise SystemExit("Need at least 2 frames")
    h, w = frames[0].shape[:2]
    name = os.path.splitext(os.path.basename(args.input))[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(args.out_dir, f"{name}_x{args.factor}.mp4")
    vw = cv2.VideoWriter(out_path, fourcc, fps * args.factor, (w, h))
    cmp_vw = None
    if args.compare:
        cmp_path = os.path.join(args.out_dir, f"{name}_compare.mp4")
        cmp_vw = cv2.VideoWriter(cmp_path, fourcc, fps * args.factor, (3 * w, h))

    n_mid = args.factor - 1
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        pair = FlowPair(a, b, args.method)
        if i == 0:
            cv2.imwrite(os.path.join(args.out_dir, f"{name}_flow_preview.png"),
                        np.hstack([label(a, "frame 0"), label(b, "frame 1"),
                                   label(flow_to_color(pair.f01), "flow 0->1")]))
        vw.write(a)
        if cmp_vw:
            cmp_vw.write(np.hstack([label(a, "original"), label(a, "cross-fade"), label(a, "optical flow")]))
        for k in range(1, n_mid + 1):
            t = k / (n_mid + 1)
            mid = interpolate_at(pair, t)
            vw.write(mid)
            if cmp_vw:
                cmp_vw.write(np.hstack([label(a, "original (repeat)"),
                                        label(interpolate_average(a, b, t), "cross-fade"),
                                        label(mid, f"optical flow ({args.method})")]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames) - 1} pairs")
    vw.write(frames[-1])
    vw.release()
    if cmp_vw:
        cmp_vw.release()
    print(f"Wrote {out_path}  ({fps:.1f} fps -> {fps * args.factor:.1f} fps)")
    if args.compare:
        print(f"Wrote {cmp_path}")


if __name__ == "__main__":
    main()
