"""
Core of the project: video frame interpolation using dense optical flow.

Pipeline (for two consecutive frames I0 and I1, and a time t in (0, 1)):

  1. Estimate dense optical flow in BOTH directions:
         F01 : I0 -> I1        F10 : I1 -> I0
     (Farneback or DIS, both classical, non-learned algorithms in OpenCV).

  2. Approximate the flow from the unknown middle frame It back to I0 and I1
     (Jiang et al., "Super SloMo", CVPR 2018, eq. 1-2):
         F_t0 = -(1-t)*t*F01 + t^2 * F10
         F_t1 =  (1-t)^2 * F01 - t*(1-t) * F10

  3. Backward-warp I0 and I1 to time t with those flows (cv2.remap).

  4. Handle occlusions with a forward-backward consistency check:
     where F01(x) and F10(x + F01(x)) do not cancel out, the pixel is
     probably occluded in the other frame, so its contribution is down-weighted.

  5. Blend:  It = (w0 * W0 + w1 * W1) / (w0 + w1),
     with  w0 = (1-t) * V0  and  w1 = t * V1  (V = visibility from step 4).

Baselines used for comparison in the evaluation:
  * "average"      : (I0 + I1) / 2                       (no motion model)
  * "flow_oneway"  : warp only I0 with -t * F01 style flow (ablation, no
                     bidirectional blending / no occlusion handling)
"""
from __future__ import annotations

import cv2
import numpy as np

FLOW_METHODS = ("dis", "farneback")


# --------------------------------------------------------------------------
# Optical flow
# --------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def compute_flow(a: np.ndarray, b: np.ndarray, method: str = "dis") -> np.ndarray:
    """Dense optical flow from frame a to frame b. Returns float32 HxWx2 (dx, dy)."""
    ga, gb = _gray(a), _gray(b)
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(
            ga, gb, None,
            pyr_scale=0.5, levels=5, winsize=15, iterations=5,
            poly_n=7, poly_sigma=1.5, flags=0,
        ).astype(np.float32)
    if method == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis.calc(ga, gb, None).astype(np.float32)
    raise ValueError(f"unknown flow method {method!r}, choose from {FLOW_METHODS}")


# --------------------------------------------------------------------------
# Warping helpers
# --------------------------------------------------------------------------
def backward_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    out(x) = img(x + flow(x))   (flow is defined on the OUTPUT grid).
    Works for 2-D or 3-D images, uint8 or float32.
    """
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def fb_consistency_error(f_fwd: np.ndarray, f_bwd: np.ndarray) -> np.ndarray:
    """
    Forward-backward flow error defined on the grid of the frame that f_fwd starts from:
        e(x) = || f_fwd(x) + f_bwd(x + f_fwd(x)) ||
    Large values => occlusion / unreliable flow.
    """
    bwd_at_target = backward_warp(f_bwd, f_fwd)
    return np.linalg.norm(f_fwd + bwd_at_target, axis=2).astype(np.float32)


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------
class FlowPair:
    """Pre-computed flows between two frames so that many t values are cheap."""

    def __init__(self, i0: np.ndarray, i1: np.ndarray, method: str = "dis"):
        self.i0, self.i1 = i0, i1
        self.f01 = compute_flow(i0, i1, method)
        self.f10 = compute_flow(i1, i0, method)
        self.err0 = fb_consistency_error(self.f01, self.f10)   # on I0's grid
        self.err1 = fb_consistency_error(self.f10, self.f01)   # on I1's grid


def interpolate_at(pair: FlowPair, t: float, occlusion_alpha: float = 0.5) -> np.ndarray:
    """Bidirectional flow interpolation with occlusion-aware blending."""
    f01, f10 = pair.f01, pair.f10
    f_t0 = -(1 - t) * t * f01 + (t * t) * f10
    f_t1 = ((1 - t) ** 2) * f01 - t * (1 - t) * f10

    w0 = backward_warp(pair.i0.astype(np.float32), f_t0)
    w1 = backward_warp(pair.i1.astype(np.float32), f_t1)

    # visibility maps brought to time t
    v0 = np.exp(-occlusion_alpha * backward_warp(pair.err0, f_t0))[..., None]
    v1 = np.exp(-occlusion_alpha * backward_warp(pair.err1, f_t1))[..., None]

    a0 = (1 - t) * v0
    a1 = t * v1
    out = (a0 * w0 + a1 * w1) / np.maximum(a0 + a1, 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def interpolate_oneway(pair: FlowPair, t: float) -> np.ndarray:
    """Ablation: warp I0 only (no bidirectional blend, no occlusion handling)."""
    f_t0 = -t * pair.f01
    return backward_warp(pair.i0, f_t0)


def interpolate_average(i0: np.ndarray, i1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Baseline without motion: simple cross-fade."""
    out = (1 - t) * i0.astype(np.float32) + t * i1.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def interpolate_pair(i0, i1, n_mid: int = 1, method: str = "dis"):
    """Return the n_mid in-between frames for a pair (t = k/(n_mid+1))."""
    pair = FlowPair(i0, i1, method)
    return [interpolate_at(pair, k / (n_mid + 1)) for k in range(1, n_mid + 1)]


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Standard HSV optical-flow visualisation (hue = direction, value = magnitude)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    if max_mag is None:
        max_mag = max(float(np.percentile(mag, 99)), 1e-3)
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small caption into the top-left of an image (returns a copy)."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (8 + 8 * len(text), 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out
