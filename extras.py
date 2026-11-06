"""
Extra experiments that make the project more convincing.

  ablation    which part of the method adds what?  (cross-fade -> one-way warp -> two-way warp -> + occlusion weights)
  alpha       choose the occlusion strength alpha on a SEPARATE validation set, then report it on the test set
  brightness  does a brightness change between the frames hurt flow?  (the base paper says phase-based methods are robust)
  hybrid      "use cross-fade when there is almost no motion, otherwise flow": threshold chosen on the validation set

The TEST videos are exactly the ones evaluate.py used (same seed, same sampling), so the numbers line up with your
results_gap* folders. The VALIDATION videos never overlap with them, so nothing is tuned on the test data.

Usage (one at a time):
  python extras.py ablation   --data_dir UCF-101 --gap 2
  python extras.py alpha      --data_dir UCF-101 --gap 2
  python extras.py hybrid     --data_dir UCF-101 --gap 2
  python extras.py brightness --data_dir UCF-101 --gap 2 --num_test 100
Results are written to extras_gap<N>/  (markdown tables + a png).
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
from scipy import stats

from evaluate import list_videos, read_frames, score
from flowinterp import (FlowPair, interpolate_at, interpolate_average,
                        interpolate_oneway)


# ----------------------------------------------------------------------------- data
def split_videos(data_dir, num_test, num_val, seed=0, val_seed=123):
    """Reproduce evaluate.py's test sample, and draw a validation sample that does not overlap with it."""
    vids = list_videos(data_dir)
    if not vids:
        raise SystemExit(f"No videos under {data_dir!r}")
    if num_test and num_test < len(vids):
        test = random.Random(seed).sample(vids, num_test)       # same call evaluate.py makes
    else:
        test = list(vids)
    rest = sorted(set(vids) - set(test))
    if num_val > len(rest):
        num_val = len(rest)
    val = random.Random(val_seed).sample(rest, num_val) if num_val else []
    return test, val


def triplets(vids, gap, per_video=3):
    """Yield (a, gt, b) exactly as evaluate.py picks them."""
    need = 2 * gap + 1
    for path in vids:
        frames = read_frames(path)
        if len(frames) < need:
            continue
        for s in np.linspace(0, len(frames) - need, per_video).astype(int):
            yield frames[s], frames[s + gap], frames[s + 2 * gap]


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    p = stats.ttest_rel(a[ok], b[ok]).pvalue if len(d) > 1 else float("nan")
    return d.mean(), p


def save(out_dir, name, lines):
    os.makedirs(out_dir, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved {os.path.join(out_dir, name)}")


# ----------------------------------------------------------------------------- experiments
def run_ablation(args):
    test, _ = split_videos(args.data_dir, args.num_test, 0, args.seed)
    names = ["cross-fade", "one-way warp (A only)", "two-way warp, no occlusion weights (alpha=0)",
             f"full method (alpha={args.alpha})"]
    P = {n: [] for n in names}
    S = {n: [] for n in names}
    for i, (a, gt, b) in enumerate(triplets(test, args.gap, args.triplets_per_video), 1):
        pair = FlowPair(a, b, "dis")
        preds = [interpolate_average(a, b), interpolate_oneway(pair, 0.5),
                 interpolate_at(pair, 0.5, occlusion_alpha=0.0), interpolate_at(pair, 0.5, occlusion_alpha=args.alpha)]
        for n, p in zip(names, preds):
            ps, ss = score(p, gt)
            P[n].append(ps)
            S[n].append(ss)
        if i % 100 == 0:
            print(f"  {i} triplets")
    lines = [f"Ablation (DIS flow, gap={args.gap}, {len(P[names[0]])} test triplets)", "",
             "| Variant | PSNR (dB) mean ± 95% CI | SSIM mean | PSNR gain vs previous row | paired t-test p |",
             "|---|---|---|---|---|"]
    prev = None
    for n in names:
        m, ci = mean_ci(P[n])
        row = f"| {n} | {m:.2f} ± {ci:.2f} | {np.nanmean(S[n]):.4f} | "
        if prev is None:
            row += "- | - |"
        else:
            d, p = paired(P[n], P[prev])
            row += f"{d:+.3f} | {p:.2e} |"
        lines.append(row)
        prev = n
    lines += ["", "Read it top to bottom: each row adds one idea to the row above it. "
                  "A negative gain means that idea did not help on average."]
    save(args.out_dir, f"ablation_gap{args.gap}.md", lines)


def run_alpha(args):
    test, val = split_videos(args.data_dir, args.num_test, args.num_val, args.seed)
    if not val:
        raise SystemExit("No videos left for validation (use a smaller --num_test).")
    alphas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    def collect(vids):
        res = {al: [] for al in alphas}
        base = []
        for a, gt, b in triplets(vids, args.gap, args.triplets_per_video):
            pair = FlowPair(a, b, "dis")
            base.append(score(interpolate_average(a, b), gt)[0])
            for al in alphas:
                res[al].append(score(interpolate_at(pair, 0.5, occlusion_alpha=al), gt)[0])
        return res, base

    print(f"Validation: {len(val)} videos")
    vres, vbase = collect(val)
    vm = {al: float(np.nanmean(vres[al])) for al in alphas}
    best = max(alphas, key=lambda al: vm[al])
    print(f"Test: {len(test)} videos")
    tres, tbase = collect(test)
    lines = [f"Choosing the occlusion strength alpha (DIS, gap={args.gap})", "",
             f"Validation set: {len(val)} videos not used in testing. Test set: the same {len(test)} videos as evaluate.py.", "",
             "| alpha | validation PSNR (dB) | test PSNR (dB) |", "|---|---|---|"]
    for al in alphas:
        mark = "  **<- chosen on validation**" if al == best else ""
        lines.append(f"| {al} | {vm[al]:.3f} | {np.nanmean(tres[al]):.3f}{mark} |")
    lines += ["", f"Cross-fade for reference: validation {np.nanmean(vbase):.3f} dB, test {np.nanmean(tbase):.3f} dB."]
    d, p = paired(tres[best], tres[0.5])
    lines.append(f"Chosen alpha={best} vs the original alpha=0.5 on the test set: {d:+.3f} dB (paired t-test p={p:.2e}).")
    lines.append("Alpha was selected using the validation PSNR only. The test column is for reporting, not for choosing.")
    save(args.out_dir, f"alpha_gap{args.gap}.md", lines)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(range(len(alphas)), [vm[al] for al in alphas], "o-", label="validation")
    ax.plot(range(len(alphas)), [np.nanmean(tres[al]) for al in alphas], "s-", label="test")
    ax.set_xticks(range(len(alphas)), [str(al) for al in alphas])
    ax.set_xlabel("occlusion strength alpha")
    ax.set_ylabel("PSNR (dB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"alpha_gap{args.gap}.png"), dpi=150)


def run_hybrid(args):
    test, val = split_videos(args.data_dir, args.num_test, args.num_val, args.seed)
    if not val:
        raise SystemExit("No videos left for validation (use a smaller --num_test).")

    def collect(vids):
        m, p_avg, p_dis = [], [], []
        for a, gt, b in triplets(vids, args.gap, args.triplets_per_video):
            pair = FlowPair(a, b, "dis")
            m.append(float(np.linalg.norm(pair.f01, axis=2).mean()))
            p_avg.append(score(interpolate_average(a, b), gt)[0])
            p_dis.append(score(interpolate_at(pair, 0.5, occlusion_alpha=args.alpha), gt)[0])
        return np.array(m), np.array(p_avg), np.array(p_dis)

    print(f"Validation: {len(val)} videos")
    vm, va, vd = collect(val)
    print(f"Test: {len(test)} videos")
    tm, ta, td = collect(test)

    def hybrid(m, pa, pd, thr):
        return np.where(m < thr, pa, pd)

    cands = [0.0] + [float(np.percentile(vm, q)) for q in (5, 10, 15, 20, 25, 30, 40, 50)]
    vscore = {t: float(np.nanmean(hybrid(vm, va, vd, t))) for t in cands}
    best = max(cands, key=lambda t: vscore[t])
    lines = [f"Motion-aware hybrid (gap={args.gap}): use cross-fade when the average motion is below a threshold, "
             "otherwise optical flow (DIS).", "",
             f"Threshold chosen on {len(vm)} validation triplets: {best:.2f} px "
             f"({'it equals pure flow, so the rule does not help' if best == 0.0 else 'cross-fade below this motion'}).", "",
             "| Method | validation PSNR | test PSNR |", "|---|---|---|",
             f"| cross-fade | {np.nanmean(va):.3f} | {np.nanmean(ta):.3f} |",
             f"| DIS flow | {np.nanmean(vd):.3f} | {np.nanmean(td):.3f} |",
             f"| hybrid (threshold {best:.2f} px) | {vscore[best]:.3f} | {np.nanmean(hybrid(tm, ta, td, best)):.3f} |"]
    d, p = paired(hybrid(tm, ta, td, best), td)
    share = float(np.mean(tm < best)) * 100
    lines += ["", f"Hybrid minus plain DIS on the test set: {d:+.3f} dB (paired t-test p={p:.2e}); "
                  f"the rule switches to cross-fade on {share:.0f}% of test triplets.",
              "A tiny or zero gain is an honest result: it means flow already does about as well as cross-fade on still scenes."]
    save(args.out_dir, f"hybrid_gap{args.gap}.md", lines)


def shift(img, delta):
    return np.clip(img.astype(np.int16) + int(round(delta)), 0, 255).astype(np.uint8)


def run_brightness(args):
    test, _ = split_videos(args.data_dir, args.num_test, 0, args.seed)
    deltas = [0, 10, 20, 40, 60]
    methods = ["cross-fade", "dis", "farneback"]
    P = {(m, d): [] for m in methods for d in deltas}
    n = 0
    for a, gt, b in triplets(test, args.gap, args.triplets_per_video):
        n += 1
        for d in deltas:
            # brightness drifts linearly from A (+0) through the hidden frame (+d/2) to B (+d)
            a2, g2, b2 = shift(a, 0), shift(gt, d / 2), shift(b, d)
            P[("cross-fade", d)].append(score(interpolate_average(a2, b2), g2)[0])
            for m in ("dis", "farneback"):
                P[(m, d)].append(score(interpolate_at(FlowPair(a2, b2, m), 0.5, args.alpha), g2)[0])
        if n % 20 == 0:
            print(f"  {n} triplets")
    lines = [f"Robustness to a brightness change between the frames (gap={args.gap}, {n} test triplets)", "",
             "Brightness drifts from frame A (+0 grey levels) through the hidden frame (+d/2) to frame B (+d). "
             "Optical flow assumes brightness stays constant, so a drift can confuse it.", "",
             "| d (grey levels) | cross-fade PSNR | DIS PSNR | Farneback PSNR | DIS gain over cross-fade |",
             "|---|---|---|---|---|"]
    for d in deltas:
        c, di, fa = (np.nanmean(P[(m, d)]) for m in methods)
        lines.append(f"| {d} | {c:.2f} | {di:.2f} | {fa:.2f} | {di - c:+.2f} |")
    lines += ["", "If the gain column shrinks or turns negative as d grows, flow is sensitive to brightness change. "
                  "The base paper reports that phase-based interpolation is more robust to this."]
    save(args.out_dir, f"brightness_gap{args.gap}.md", lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("experiment", choices=["ablation", "alpha", "brightness", "hybrid"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--num_test", type=int, default=200, help="same as --num_videos in evaluate.py")
    ap.add_argument("--num_val", type=int, default=100, help="validation videos (never overlap with the test ones)")
    ap.add_argument("--triplets_per_video", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="occlusion strength used when it is not being tuned")
    ap.add_argument("--seed", type=int, default=0, help="same as --seed in evaluate.py")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = f"extras_gap{args.gap}"
    {"ablation": run_ablation, "alpha": run_alpha, "brightness": run_brightness, "hybrid": run_hybrid}[args.experiment](args)


if __name__ == "__main__":
    main()