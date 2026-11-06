"""
Helper functions for the web UI (app.py). Kept free of any Streamlit code so they can be tested on their own.
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from flowinterp import (FlowPair, compute_flow, flow_to_color, interpolate_at,
                        interpolate_average, interpolate_oneway)
from fusion import make_pair


def load_frames(path: str, max_frames: int = 150, max_width: int = 640):
    """Read up to max_frames frames (BGR). Frames wider than max_width are shrunk to keep things fast.
    Returns (frames, fps)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 25.0
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > max_width:
            fr = cv2.resize(fr, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    return frames, float(fps)


def motion_scores(frames, gap: int, stride: int = 2):
    """Mean DIS flow magnitude between A and B for every candidate start position.
    Returns (starts, motions)."""
    need = 2 * gap + 1
    starts = list(range(0, len(frames) - need + 1, stride))
    motions = []
    for s in starts:
        f = compute_flow(frames[s], frames[s + 2 * gap], "dis")
        motions.append(float(np.linalg.norm(f, axis=2).mean()))
    return starts, motions


def _scores(pred, gt):
    p = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    return float(psnr(g, p, data_range=255)), float(ssim(g, p, data_range=255, channel_axis=2))


def compare_at(frames, start: int, gap: int, method: str = "dis", alpha: float = 0.5):
    """Hide frame start+gap, rebuild it from start and start+2*gap, and score every method."""
    a, gt, b = frames[start], frames[start + gap], frames[start + 2 * gap]
    pair = make_pair(a, b, method)
    flow_img = interpolate_at(pair, 0.5, occlusion_alpha=alpha)
    avg_img = interpolate_average(a, b, 0.5)
    one_img = interpolate_oneway(pair, 0.5)
    err = np.clip(np.abs(flow_img.astype(int) - gt.astype(int)) * 4, 0, 255).astype(np.uint8)
    out = {
        "A": a, "GT": gt, "B": b,
        "crossfade": avg_img, "oneway": one_img, "flow": flow_img, "error": err,
        "flow_color": flow_to_color(pair.f01),
        "motion_px": float(np.linalg.norm(pair.f01, axis=2).mean()),
    }
    for key in ("crossfade", "oneway", "flow"):
        out[key + "_psnr"], out[key + "_ssim"] = _scores(out[key], gt)
    return out


def make_slowmo(frames, factor: int, method: str = "dis", alpha: float = 0.5, progress=None):
    """Insert factor-1 new frames between each consecutive pair."""
    out = []
    n = len(frames) - 1
    for i in range(n):
        pair = make_pair(frames[i], frames[i + 1], method)
        out.append(frames[i])
        for k in range(1, factor):
            out.append(interpolate_at(pair, k / factor, occlusion_alpha=alpha))
        if progress:
            progress((i + 1) / n)
    out.append(frames[-1])
    return out


def write_h264(frames, fps: float, path: str) -> bool:
    """Write an H.264 mp4 that browsers can play (needs the imageio-ffmpeg package).
    Returns True on success."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2                      # H.264 needs even sizes
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
           "-r", f"{fps:.3f}", "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", path]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr[:h2, :w2]).tobytes())
        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0
    except Exception:
        return False


def _shift(img, delta):
    return np.clip(img.astype(np.int16) + int(round(delta)), 0, 255).astype(np.uint8)


def brightness_demo(frames, start: int, gap: int, method: str = "dis", alpha: float = 0.5, delta: float = 0):
    """Brightness drifts from A (+0) through the hidden frame (+delta/2) to B (+delta).
    Returns the three real frames and both rebuilt frames with their PSNR / SSIM."""
    a = frames[start]
    gt = _shift(frames[start + gap], delta / 2)
    b = _shift(frames[start + 2 * gap], delta)
    pair = make_pair(a, b, method)
    out = {"A": a, "GT": gt, "B": b,
           "crossfade": interpolate_average(a, b, 0.5),
           "flow": interpolate_at(pair, 0.5, occlusion_alpha=alpha)}
    for key in ("crossfade", "flow"):
        out[key + "_psnr"], out[key + "_ssim"] = _scores(out[key], gt)
    return out