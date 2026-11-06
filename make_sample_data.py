"""
Generate a tiny UCF101-style dataset of SYNTHETIC videos (320x240, 25 fps, .avi)
so you can smoke-test the code without downloading the real 6.5 GB dataset.

  python make_sample_data.py --out sample_data --videos 6

This is NOT UCF101 -- for your report use the real dataset with evaluate.py.
"""
import argparse
import os

import cv2
import numpy as np


def make_video(path, seed, n_frames=40, w=320, h=240):
    rng = np.random.default_rng(seed)
    # textured, wide background so it can pan
    bg = rng.integers(0, 255, (h + 80, w + 160, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.normalize(bg, None, 40, 220, cv2.NORM_MINMAX)
    pan = rng.uniform(-2.5, 2.5, 2)                      # background motion px/frame
    obj_pos = np.array([w / 2, h / 2], float)
    obj_vel = rng.uniform(-4, 4, 2)
    obj_col = tuple(int(c) for c in rng.integers(0, 255, 3))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (w, h))
    for i in range(n_frames):
        ox = int(80 + pan[0] * i) % 160
        oy = int(40 + pan[1] * i) % 80
        frame = bg[oy:oy + h, ox:ox + w].copy()
        obj_pos += obj_vel
        for d in (0, 1):
            lim = (w, h)[d]
            if not (20 < obj_pos[d] < lim - 20):
                obj_vel[d] *= -1
        cv2.circle(frame, tuple(obj_pos.astype(int)), 22, obj_col, -1)
        cv2.rectangle(frame, tuple((obj_pos - 10).astype(int)), tuple((obj_pos + 10).astype(int)), (255, 255, 255), 2)
        vw.write(frame)
    vw.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_data")
    ap.add_argument("--videos", type=int, default=6)
    a = ap.parse_args()
    classes = ["SyntheticPan", "SyntheticBounce"]
    for i in range(a.videos):
        c = classes[i % 2]
        d = os.path.join(a.out, c)
        os.makedirs(d, exist_ok=True)
        make_video(os.path.join(d, f"v_{c}_g{i + 1:02d}_c01.avi"), seed=i)
    print(f"Wrote {a.videos} synthetic videos to {a.out}/")
