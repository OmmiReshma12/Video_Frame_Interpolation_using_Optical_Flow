"""
Quantitative evaluation of flow-based frame interpolation on UCF101.

Protocol (the standard "drop a frame and re-create it" test):
  * take three frames  A = frame i,  GT = frame i+gap,  B = frame i+2*gap
  * hide GT, predict it from A and B at t = 0.5
  * compare the prediction with GT using PSNR and SSIM

Methods compared:
  average          : cross-fade of A and B (no motion model)  [baseline]
  flow_oneway_dis  : warp A only with DIS flow                 [ablation]
  dis              : full method, DIS flow, bidirectional + occlusion handling
  farneback        : full method, Farneback flow

Usage:
  python evaluate.py --data_dir /path/to/UCF-101 --num_videos 100 --gap 1
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from flowinterp import (FlowPair, interpolate_at, interpolate_average,
                        interpolate_oneway, label)

METHODS = ["average", "flow_oneway_dis", "dis", "farneback"]


def list_videos(data_dir: str):
    vids = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
                vids.append(os.path.join(root, f))
    return sorted(vids)


def read_frames(path: str, max_frames: int = 300):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def predict_all(a, b, t=0.5):
    """Return ({method_name: predicted_frame}, motion_in_pixels) for frames a, b."""
    out = {"average": interpolate_average(a, b, t)}
    p_dis = FlowPair(a, b, "dis")
    out["flow_oneway_dis"] = interpolate_oneway(p_dis, t)
    out["dis"] = interpolate_at(p_dis, t)
    p_fb = FlowPair(a, b, "farneback")
    out["farneback"] = interpolate_at(p_fb, t)
    # average motion between A and B in pixels (used to bucket triplets into low/medium/high motion)
    motion = float(np.linalg.norm(p_dis.f01, axis=2).mean())
    return out, motion


def score(pred, gt):
    g_p = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    g_g = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    return (psnr(g_g, g_p, data_range=255),
            ssim(g_g, g_p, data_range=255, channel_axis=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="UCF101 root folder (contains one sub-folder per class)")
    ap.add_argument("--out_dir", default="results", help="where to write csv / plots / sample images")
    ap.add_argument("--num_videos", type=int, default=100, help="random videos to evaluate (use 0 for ALL)")
    ap.add_argument("--triplets_per_video", type=int, default=3)
    ap.add_argument("--gap", type=int, default=1,
                    help="frame distance between A->GT and GT->B. 1 = consecutive frames (25 fps -> 50 fps),"
                         " 2 or 3 = harder (larger motion)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_samples", type=int, default=6, help="number of qualitative comparison images")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    vids = list_videos(args.data_dir)
    if not vids:
        raise SystemExit(f"No video files found under {args.data_dir!r}")
    print(f"Found {len(vids)} videos")
    if args.num_videos and args.num_videos < len(vids):
        vids = random.sample(vids, args.num_videos)
    print(f"Evaluating {len(vids)} videos, {args.triplets_per_video} triplets each, gap={args.gap}")

    rows = []                       # per-triplet results
    t0 = time.time()
    saved = 0
    for vi, path in enumerate(vids, 1):
        frames = read_frames(path)
        need = 2 * args.gap + 1
        if len(frames) < need:
            continue
        cls = os.path.basename(os.path.dirname(path))
        starts = np.linspace(0, len(frames) - need, args.triplets_per_video).astype(int)
        for s in starts:
            a, gt, b = frames[s], frames[s + args.gap], frames[s + 2 * args.gap]
            preds, motion = predict_all(a, b)
            row = {"video": os.path.basename(path), "class": cls, "start": int(s), "motion_px": motion}
            for m in METHODS:
                p, s_ = score(preds[m], gt)
                row[f"{m}_psnr"], row[f"{m}_ssim"] = p, s_
            rows.append(row)

            if saved < args.save_samples and vi % max(1, len(vids) // max(1, args.save_samples)) == 0:
                tiles = [label(a, "frame A"), label(gt, "ground truth"), label(b, "frame B"),
                         label(preds["average"], "average"),
                         label(preds["dis"], "flow (DIS)"),
                         label(np.abs(preds["dis"].astype(int) - gt.astype(int)).astype(np.uint8) * 4,
                               "|flow-GT| x4")]
                grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
                cv2.imwrite(os.path.join(args.out_dir, "samples", f"sample_{saved:02d}_{cls}.png"), grid)
                saved += 1
        if vi % 10 == 0 or vi == len(vids):
            print(f"  {vi}/{len(vids)} videos  ({time.time() - t0:.0f}s)")

    if not rows:
        raise SystemExit("No triplets evaluated (videos too short?).")

    # ---- per-triplet csv
    csv_path = os.path.join(args.out_dir, "per_triplet_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary
    summary = {}
    for m in METHODS:
        p = np.array([r[f"{m}_psnr"] for r in rows])
        s = np.array([r[f"{m}_ssim"] for r in rows])
        p = p[np.isfinite(p)]
        summary[m] = (p.mean(), p.std(), s.mean(), s.std())
    wins = sum(r["dis_psnr"] > r["average_psnr"] for r in rows)

    lines = [f"Evaluated {len(rows)} triplets from {len(set(r['video'] for r in rows))} videos (gap={args.gap})", "",
             "| Method | PSNR (dB) mean ± std | SSIM mean ± std |", "|---|---|---|"]
    for m in METHODS:
        pm, ps, sm, ss = summary[m]
        lines.append(f"| {m} | {pm:.2f} ± {ps:.2f} | {sm:.4f} ± {ss:.4f} |")
    lines += ["", f"DIS flow beats the cross-fade baseline (PSNR) on {wins}/{len(rows)} triplets "
                  f"({100 * wins / len(rows):.1f}%)."]

    # ---- motion-bucket analysis + paired significance tests
    from scipy import stats
    motion = np.array([r["motion_px"] for r in rows])
    ok = np.ones(len(rows), bool)
    for m in METHODS:
        ok &= np.isfinite([r[f"{m}_psnr"] for r in rows])
    q1, q2 = np.percentile(motion, [33.3, 66.7])
    bucket = np.where(motion <= q1, 0, np.where(motion <= q2, 1, 2))
    names = ["low motion", "medium motion", "high motion"]
    lines += ["", f"### Results by motion amount (mean flow magnitude between A and B; "
                  f"thresholds {q1:.2f} px and {q2:.2f} px, equal-sized groups)", "",
              "| Motion group | n | " + " | ".join(f"{m} PSNR" for m in METHODS) +
              " | DIS gain over average (dB) | DIS win rate |",
              "|---|---|" + "---|" * (len(METHODS) + 2)]
    for bi in range(3):
        sel = (bucket == bi) & ok
        if not sel.any():
            continue
        means = [np.mean([r[f"{m}_psnr"] for r, s_ in zip(rows, sel) if s_]) for m in METHODS]
        d = np.array([r["dis_psnr"] - r["average_psnr"] for r, s_ in zip(rows, sel) if s_])
        lines.append(f"| {names[bi]} | {sel.sum()} | " + " | ".join(f"{x:.2f}" for x in means) +
                     f" | {d.mean():+.2f} | {100 * (d > 0).mean():.1f}% |")

    lines += ["", "### Paired significance tests on per-triplet PSNR (two-sided)", "",
              "| Comparison | mean diff (dB) | paired t-test p | Wilcoxon p |", "|---|---|---|---|"]
    for a_, b_ in [("dis", "average"), ("farneback", "average"), ("dis", "flow_oneway_dis"), ("dis", "farneback")]:
        pa = np.array([r[f"{a_}_psnr"] for r in rows])[ok]
        pb = np.array([r[f"{b_}_psnr"] for r in rows])[ok]
        d = pa - pb
        t_p = stats.ttest_rel(pa, pb).pvalue
        try:
            w_p = stats.wilcoxon(d).pvalue
        except ValueError:
            w_p = float("nan")
        lines.append(f"| {a_} vs {b_} | {d.mean():+.3f} | {t_p:.2e} | {w_p:.2e} |")
    lines += ["", "Note: triplets from the same video are not fully independent, so p-values are optimistic."]

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(text + "\n")

    # ---- per-class table
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    with open(os.path.join(args.out_dir, "per_class_psnr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n"] + [f"{m}_psnr" for m in METHODS])
        for c, rs in sorted(by_cls.items()):
            w.writerow([c, len(rs)] + [f"{np.mean([r[f'{m}_psnr'] for r in rs]):.3f}" for m in METHODS])

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(METHODS))
    ax[0].bar(x, [summary[m][0] for m in METHODS], yerr=[summary[m][1] for m in METHODS], capsize=3,
              color=["#9aa0a6", "#c9a227", "#1f77b4", "#2ca02c"])
    ax[0].set_xticks(x, METHODS, rotation=20)
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_title("Mean PSNR (higher is better)")
    ax[1].bar(x, [summary[m][2] for m in METHODS], yerr=[summary[m][3] for m in METHODS], capsize=3,
              color=["#9aa0a6", "#c9a227", "#1f77b4", "#2ca02c"])
    ax[1].set_xticks(x, METHODS, rotation=20)
    ax[1].set_ylabel("SSIM")
    ax[1].set_title("Mean SSIM (higher is better)")
    lo = min(summary[m][2] for m in METHODS)
    ax[1].set_ylim(max(0, lo - 0.1), 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "metrics_bar.png"), dpi=150)

    gain = [r["dis_psnr"] - r["average_psnr"] for r in rows if np.isfinite(r["dis_psnr"]) and np.isfinite(r["average_psnr"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(gain, bins=30, color="#1f77b4")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("PSNR gain of DIS-flow over cross-fade (dB)")
    ax.set_ylabel("# triplets")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "psnr_gain_hist.png"), dpi=150)

    # gain vs motion scatter
    mo = np.array([r["motion_px"] for r in rows])[ok]
    g = np.array([r["dis_psnr"] - r["average_psnr"] for r in rows])[ok]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.scatter(mo, g, s=8, alpha=0.4, color="#1f77b4")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("average motion between A and B (pixels)")
    ax.set_ylabel("PSNR gain of DIS over cross-fade (dB)")
    ax.set_title("DIS gain vs. amount of motion")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "gain_vs_motion.png"), dpi=150)
    print(f"\nSaved results to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()