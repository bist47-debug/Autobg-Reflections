"""Plan B input for the student: the lower half of the frame (car bottom + floor) with a faint MIRROR HINT -
the car flipped upside down under its own ground line - so the student only has to learn what v2 adds
(perspective, fade, blending into the floor), not the whole mirror geometry.
Used identically by data prep, evaluation and production."""
import numpy as np, cv2
from PIL import Image

SW, SH = 1024, 384          # student works on this strip size
TOP = 0.5                   # strip = rows from 50 % of the height to the bottom (car is placed with tyres at 88 %)


def car_mask(in_rgb, bg_rgb):
    """car pixels = where the input differs from the plain backdrop (both at the same size, lossless)"""
    d = np.abs(in_rgb.astype(np.int16) - bg_rgb.astype(np.int16)).max(-1)
    m = (d > 3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(bool)            # holes (windows etc.) filled


def mirror_hint(in_rgb, m, strength=0.45, length=0.35):
    """paint the car mirrored about its ground line (bottom of its convex hull), fading out"""
    H, W = m.shape
    ys, xs = np.where(m)
    if len(xs) < 50:
        return in_rgb.copy()
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    hm = np.zeros_like(m, np.uint8); cv2.fillConvexPoly(hm, hull, 1)
    cols = np.where(hm.any(0))[0]
    g = np.full(W, -1, np.int32)
    g[cols] = H - 1 - np.argmax(hm[::-1, cols] > 0, 0)          # ground line per column
    D = max(4, int(length * (ys.max() - ys.min())))
    out = in_rgb.astype(np.float32)
    mir = np.zeros_like(out); al = np.zeros((H, W), np.float32)
    d = np.arange(D)
    for x in cols:
        src = g[x] - d; dst = g[x] + 1 + d
        ok = (src >= 0) & (dst < H)
        src, dst, dd = src[ok], dst[ok], d[ok]
        keep = m[src, x] & ~m[dst, x]
        mir[dst[keep], x] = in_rgb[src[keep], x]
        al[dst[keep], x] = strength * (1 - dd[keep] / D)
    al = cv2.GaussianBlur(al, (0, 0), 1.2)
    mir = cv2.GaussianBlur(mir, (0, 0), 1.2)
    out = out * (1 - al[..., None]) + mir * al[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def strip(img_rgb):
    """bottom part of the frame -> student size"""
    H = img_rgb.shape[0]; y0 = int(round(TOP * H))
    return cv2.resize(img_rgb[y0:], (SW, SH), interpolation=cv2.INTER_AREA), y0


def unstrip(full_rgb, strip_rgb, y0):
    """put the student's strip back into the full frame"""
    out = full_rgb.copy(); H, W = out.shape[:2]
    out[y0:] = cv2.resize(strip_rgb, (W, H - y0), interpolation=cv2.INTER_CUBIC)
    return out


def make_input(in_rgb, bg_rgb):
    m = car_mask(in_rgb, bg_rgb)
    hinted = mirror_hint(in_rgb, m)
    s, y0 = strip(hinted)
    return s, y0
