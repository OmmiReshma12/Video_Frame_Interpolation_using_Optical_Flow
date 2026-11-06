"""
Save still-image comparisons from a video, automatically picking the moments with the MOST motion
(so you never have to pause a video at the right moment).

For each chosen moment it hides a real frame, re-creates it from its neighbours and shows:

    top row   : frame A | ground truth (real middle frame) | frame B
    bottom row: cross-fade | optical flow (DIS) | |flow - ground truth| x4

Usage:
  python montage.py --input UCF-101\\Diving\\v_Diving_g01_c01.avi
  python montage.py --input UCF-101\\Diving\\v_Diving_g01_c01.avi --top 5 --gap 2 --out_dir stills
"""
import argparse
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from flowinterp import (FlowPair, compute_flow, interpolate_at,
                        interpolate_average, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="path to a video file")
    ap.add_argument("--out_dir", default="stills")
    ap.add_argument("--top", type=int, default=3, help="how many images to save (highest-motion moments)")
    ap.add_argument("--gap", type=int, default=1,
                    help="frames between A and the hidden middle frame (1 = consecutive, 2-3 = larger motion, "
                         "which makes the cross-fade ghosting easier to see)")
    ap.add_argument("--stride", type=int, default=2, help="check every N-th start position (speed)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    need = 2 * args.gap + 1
    if len(frames) < need:
        raise SystemExit(f"Video too short ({len(frames)} frames).")

    # 1) motion score for every candidate start position
    starts = list(range(0, len(frames) - need + 1, args.stride))
    motion = []
    for s in starts:
        f = compute_flow(frames[s], frames[s + 2 * args.gap], "dis")
        motion.append(float(np.linalg.norm(f, axis=2).mean()))

    # 2) pick the top moments, keeping them apart so they are not near-duplicates
    chosen = []
    for idx in np.argsort(motion)[::-1]:
        s = starts[idx]
        if all(abs(s - c) >= 5 * args.gap for c, _ in chosen):
            chosen.append((s, motion[idx]))
        if len(chosen) == args.top:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    for rank, (s, mo) in enumerate(chosen, 1):
        a, gt, b = frames[s], frames[s + args.gap], frames[s + 2 * args.gap]
        pair = FlowPair(a, b, "dis")
        flow_img = interpolate_at(pair, 0.5)
        avg_img = interpolate_average(a, b, 0.5)
        p_avg = psnr(gt, avg_img, data_range=255)
        p_flow = psnr(gt, flow_img, data_range=255)
        err = np.clip(np.abs(flow_img.astype(int) - gt.astype(int)) * 4, 0, 255).astype(np.uint8)
        tiles = [label(a, f"frame A (#{s})"), label(gt, "ground truth"), label(b, "frame B"),
                 label(avg_img, f"cross-fade  {p_avg:.1f} dB"),
                 label(flow_img, f"optical flow  {p_flow:.1f} dB"),
                 label(err, "|flow - GT| x4")]
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        out = os.path.join(args.out_dir, f"{name}_moment{rank}.png")
        cv2.imwrite(out, grid)
        print(f"Saved {out}   (motion {mo:.1f}px, PSNR cross-fade {p_avg:.2f} dB vs flow {p_flow:.2f} dB)")


if __name__ == "__main__":
    main()