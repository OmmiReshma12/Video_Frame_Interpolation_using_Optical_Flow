"""
Does combining the two flow estimators beat either one alone?

Methods compared on the SAME test triplets that evaluate.py uses:
  dis             DIS flow alone
  farneback       Farneback flow alone
  average_output  average of the DIS frame and the Farneback frame
  fused_flow      per-pixel choice between the two flows (smaller forward-backward error wins)   [fusion.py]
  motion_switch   DIS when the motion is below a threshold, Farneback above it
                  (threshold chosen on a SEPARATE validation set)

Usage:
  python fusion_experiment.py --data_dir UCF-101 --gap 2
  python fusion_experiment.py --data_dir UCF-101 --gap 2 --num_test 20 --num_val 20     (quick check)
"""
import argparse

import numpy as np

from evaluate import score
from extras import mean_ci, paired, save, split_videos, triplets
from flowinterp import FlowPair, interpolate_at, interpolate_average
from fusion import fuse_pairs


def collect(vids, gap, per_video, alpha):
    rows = {k: [] for k in ("dis", "farneback", "average_output", "fused_flow")}
    ssims = {k: [] for k in rows}
    motion = []
    for i, (a, gt, b) in enumerate(triplets(vids, gap, per_video), 1):
        pd, pf = FlowPair(a, b, "dis"), FlowPair(a, b, "farneback")
        out_d = interpolate_at(pd, 0.5, alpha)
        out_f = interpolate_at(pf, 0.5, alpha)
        preds = {"dis": out_d, "farneback": out_f,
                 "average_output": interpolate_average(out_d, out_f, 0.5),
                 "fused_flow": interpolate_at(fuse_pairs(pd, pf), 0.5, alpha)}
        for k, p in preds.items():
            ps, ss = score(p, gt)
            rows[k].append(ps)
            ssims[k].append(ss)
        motion.append(float(np.linalg.norm(pd.f01, axis=2).mean()))
        if i % 100 == 0:
            print(f"  {i} triplets")
    return {k: np.array(v) for k, v in rows.items()}, {k: np.array(v) for k, v in ssims.items()}, np.array(motion)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--num_test", type=int, default=200)
    ap.add_argument("--num_val", type=int, default=100)
    ap.add_argument("--triplets_per_video", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or f"extras_gap{args.gap}"

    test, val = split_videos(args.data_dir, args.num_test, args.num_val, args.seed)
    if not val:
        raise SystemExit("No videos left for validation (use a smaller --num_test).")
    print(f"Validation: {len(val)} videos")
    vP, _, vM = collect(val, args.gap, args.triplets_per_video, args.alpha)
    print(f"Test: {len(test)} videos")
    tP, tS, tM = collect(test, args.gap, args.triplets_per_video, args.alpha)

    # motion switch: DIS below the threshold, Farneback above it; threshold picked on validation only
    cands = [float(np.percentile(vM, q)) for q in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
    best_t = max(cands, key=lambda t: np.nanmean(np.where(vM < t, vP["dis"], vP["farneback"])))
    tP["motion_switch"] = np.where(tM < best_t, tP["dis"], tP["farneback"])

    names = ["dis", "farneback", "average_output", "fused_flow", "motion_switch"]
    lines = [f"Combining DIS and Farneback (gap={args.gap}, {len(tM)} test triplets, {len(val)} validation videos)", "",
             f"The motion-switch threshold ({best_t:.2f} px) was chosen on the validation set only.", "",
             "| Method | PSNR (dB) mean ± 95% CI | SSIM mean | PSNR gain over DIS | paired t-test p | beats DIS in |",
             "|---|---|---|---|---|---|"]
    for n in names:
        m, ci = mean_ci(tP[n])
        ssim_txt = f"{np.nanmean(tS[n]):.4f}" if n in tS else "-"
        if n == "dis":
            lines.append(f"| {n} | {m:.2f} ± {ci:.2f} | {ssim_txt} | - | - | - |")
        else:
            d, p = paired(tP[n], tP["dis"])
            win = 100 * float(np.mean(tP[n] > tP["dis"]))
            lines.append(f"| {n} | {m:.2f} ± {ci:.2f} | {ssim_txt} | {d:+.3f} | {p:.2e} | {win:.0f}% of triplets |")
    lines += ["", "A gain close to zero (or a p-value that is not small) means combining the estimators did not "
                  "help beyond the noise. Report whichever result you get."]
    save(out_dir, f"fusion_gap{args.gap}.md", lines)


if __name__ == "__main__":
    main()