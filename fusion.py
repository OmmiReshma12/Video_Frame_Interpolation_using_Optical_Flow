"""
Combining the two optical-flow estimators (DIS and Farneback).

Per-pixel fusion: both estimators give an arrow at every pixel. At each pixel we keep the arrow whose
forward-backward consistency error is smaller, i.e. the one that agrees better with the flow in the opposite direction.
The occlusion weights are then recomputed from the fused flows.
"""
import numpy as np

from flowinterp import FlowPair, fb_consistency_error


def fuse_pairs(p1: FlowPair, p2: FlowPair) -> FlowPair:
    p = FlowPair.__new__(FlowPair)                     # build a FlowPair without recomputing the flows
    p.i0, p.i1 = p1.i0, p1.i1
    p.f01 = np.where((p1.err0 <= p2.err0)[..., None], p1.f01, p2.f01)
    p.f10 = np.where((p1.err1 <= p2.err1)[..., None], p1.f10, p2.f10)
    p.err0 = fb_consistency_error(p.f01, p.f10)
    p.err1 = fb_consistency_error(p.f10, p.f01)
    return p


def make_pair(a, b, method: str = "dis") -> FlowPair:
    """FlowPair for 'dis', 'farneback' or 'fused' (DIS + Farneback, chosen per pixel)."""
    if method == "fused":
        return fuse_pairs(FlowPair(a, b, "dis"), FlowPair(a, b, "farneback"))
    return FlowPair(a, b, method)