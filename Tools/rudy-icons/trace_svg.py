"""Trace the Rudy line art into vector paths and write rudy.svg / rudy-lines.svg.

Run from the repository root (needs Pillow and numpy):

    python3 Tools/rudy-icons/trace_svg.py Tools/rudy-icons/rudy.webp Tools/rudy-icons

Contours come from marching squares over the image, then get fitted with cubic
Beziers (Schneider's algorithm), keeping sharp corners such as the lapels.
Outer contours and holes wind in opposite directions, so the paths render
correctly with either the nonzero or the evenodd fill rule.

rudy.svg        white sticker shape (face fill plus a thin outline) under the black lines
rudy-lines.svg  the black lines alone
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HALO = 20          # outline around the art for colored backgrounds, in source pixels
FIT_ERROR = 0.6    # max distance between traced contour and fitted curve, in source pixels
CORNER_ANGLE = 50  # turns sharper than this many degrees stay corners
MARGIN = 8


def load_darkness(path):
    gray = np.asarray(Image.open(path).convert('L')).astype(np.float64) / 255.0
    # the source is a screenshot crop with a stray dark line along its edge
    gray[:4, :] = gray[-4:, :] = 1.0
    gray[:, :4] = gray[:, -4:] = 1.0
    return 1.0 - gray


def dilate(mask, r):
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] |= \
                mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def outside_of(walls):
    padded = np.pad(walls, 1, constant_values=False)
    im = Image.fromarray(np.where(padded, 255, 0).astype(np.uint8)).copy()
    ImageDraw.floodfill(im, (0, 0), 128)
    return (np.asarray(im) == 128)[1:-1, 1:-1]


# ---- marching squares ----

def contours(field, level=0.5):
    """Closed iso-contours of field at level, walked with the inside (> level) on the right."""
    f = np.pad(field, 1, constant_values=0.0)
    h, w = f.shape
    tl, tr = f[:-1, :-1], f[:-1, 1:]
    bl, br = f[1:, :-1], f[1:, 1:]
    case = ((tl > level) * 8 + (tr > level) * 4 + (br > level) * 2 + (bl > level) * 1).astype(np.int8)
    center = (tl + tr + bl + br) / 4.0 > level

    def lerp(a, b):
        return (level - a) / (b - a)

    # edges: 0 top, 1 right, 2 bottom, 3 left; each case maps entry edge -> exit edge
    # chosen so the region above level is on the right of the walk (y grows downward).
    table = {
        1: [(3, 2)], 2: [(2, 1)], 3: [(3, 1)], 4: [(1, 0)], 6: [(2, 0)], 7: [(3, 0)],
        8: [(0, 3)], 9: [(0, 2)], 11: [(0, 1)], 12: [(1, 3)], 13: [(1, 2)], 14: [(2, 3)],
    }

    def point(y, x, edge):
        if edge == 0:
            return (x + lerp(f[y, x], f[y, x + 1]), y)
        if edge == 1:
            return (x + 1, y + lerp(f[y, x + 1], f[y + 1, x + 1]))
        if edge == 2:
            return (x + lerp(f[y + 1, x], f[y + 1, x + 1]), y + 1)
        return (x, y + lerp(f[y, x], f[y + 1, x]))

    neighbour = {0: (-1, 0, 2), 1: (0, 1, 3), 2: (1, 0, 0), 3: (0, -1, 1)}
    links = {}
    ys, xs = np.nonzero((case != 0) & (case != 15))
    for y, x in zip(ys.tolist(), xs.tolist()):
        c = int(case[y, x])
        if c == 5:
            pairs = [(3, 0), (1, 2)] if center[y, x] else [(3, 2), (1, 0)]
        elif c == 10:
            pairs = [(0, 1), (2, 3)] if center[y, x] else [(0, 3), (2, 1)]
        else:
            pairs = table[c]
        for entry, exit_ in pairs:
            links[(y, x, entry)] = exit_

    loops = []
    while links:
        start = next(iter(links))
        y, x, entry = start
        pts = []
        while (y, x, entry) in links:
            exit_ = links.pop((y, x, entry))
            px, py = point(y, x, exit_)
            pts.append((px - 1.0, py - 1.0))  # undo the padding
            dy, dx, entry = neighbour[exit_]
            y, x = y + dy, x + dx
        if len(pts) >= 8:
            loops.append(np.array(pts))
    return loops


# ---- Bezier fitting (Schneider, Graphics Gems 1990) ----

def bezier(ctrl, t):
    t = t[:, None]
    mt = 1 - t
    return mt ** 3 * ctrl[0] + 3 * mt ** 2 * t * ctrl[1] + 3 * mt * t ** 2 * ctrl[2] + t ** 3 * ctrl[3]


def bezier_d1(ctrl, t):
    t = t[:, None]
    mt = 1 - t
    return 3 * mt ** 2 * (ctrl[1] - ctrl[0]) + 6 * mt * t * (ctrl[2] - ctrl[1]) + 3 * t ** 2 * (ctrl[3] - ctrl[2])


def bezier_d2(ctrl, t):
    t = t[:, None]
    return 6 * (1 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def chord_params(pts):
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else d


def generate(pts, u, t1, t2):
    p0, p3 = pts[0], pts[-1]
    b = np.stack([(1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2, u ** 3], axis=1)
    a1 = b[:, 1:2] * t1
    a2 = b[:, 2:3] * t2
    c00 = (a1 * a1).sum()
    c01 = (a1 * a2).sum()
    c11 = (a2 * a2).sum()
    tmp = pts - (b[:, 0:1] + b[:, 1:2]) * p0 - (b[:, 2:3] + b[:, 3:4]) * p3
    x0 = (a1 * tmp).sum()
    x1 = (a2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = np.linalg.norm(p3 - p0)
    if abs(det) > 1e-12:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det
    else:
        alpha1 = alpha2 = 0.0
    if alpha1 < 1e-6 * seg or alpha2 < 1e-6 * seg:
        alpha1 = alpha2 = seg / 3.0
    return np.array([p0, p0 + t1 * alpha1, p3 + t2 * alpha2, p3])


def reparameterize(ctrl, pts, u):
    d = bezier(ctrl, u) - pts
    d1 = bezier_d1(ctrl, u)
    d2 = bezier_d2(ctrl, u)
    num = (d * d1).sum(axis=1)
    den = (d1 * d1).sum(axis=1) + (d * d2).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = np.where(np.abs(den) > 1e-12, u - num / den, u)
    return np.clip(nu, 0, 1)


def max_error(ctrl, pts, u):
    dist = np.linalg.norm(bezier(ctrl, u) - pts, axis=1)
    i = int(np.argmax(dist))
    return dist[i], i


def fit_cubic(pts, t1, t2, error, out):
    if len(pts) == 2:
        dist = np.linalg.norm(pts[1] - pts[0]) / 3.0
        out.append(np.array([pts[0], pts[0] + t1 * dist, pts[1] + t2 * dist, pts[1]]))
        return
    u = chord_params(pts)
    ctrl = generate(pts, u, t1, t2)
    err, split = max_error(ctrl, pts, u)
    if err < error:
        out.append(ctrl)
        return
    if err < error * 4:
        for _ in range(20):
            u = reparameterize(ctrl, pts, u)
            ctrl = generate(pts, u, t1, t2)
            err, split = max_error(ctrl, pts, u)
            if err < error:
                out.append(ctrl)
                return
    split = min(max(split, 1), len(pts) - 2)
    center = unit(pts[split - 1] - pts[split + 1])
    fit_cubic(pts[:split + 1], t1, center, error, out)
    fit_cubic(pts[split:], -center, t2, error, out)


def smooth_loop(pts, passes=2):
    for _ in range(passes):
        pts = (np.roll(pts, 1, axis=0) + 2 * pts + np.roll(pts, -1, axis=0)) / 4.0
    return pts


def corners(pts, k=4):
    a = pts - np.roll(pts, k, axis=0)
    b = np.roll(pts, -k, axis=0) - pts
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    idx = [i for i in range(len(pts)) if turn[i] > CORNER_ANGLE
           and turn[i] >= turn[(i - 1) % len(pts)] and turn[i] >= turn[(i + 1) % len(pts)]]
    keep = []
    for i in idx:  # one corner per sharp turn
        if not keep or i - keep[-1] > k:
            keep.append(i)
    if len(keep) > 1 and keep[0] + len(pts) - keep[-1] <= k:
        keep.pop()
    return keep


def fit_loop(raw):
    pts = smooth_loop(raw)
    cut = corners(pts)
    n = len(pts)
    if not cut:
        cut = [0, n // 2]
        smooth_joints = True
    else:
        smooth_joints = False
    curves = []
    for j, start in enumerate(cut):
        end = cut[(j + 1) % len(cut)]
        idx = np.arange(start, end + (n if end <= start else 0) + 1) % n
        seg = pts[idx]
        if smooth_joints:
            t1 = unit(pts[(start + 2) % n] - pts[(start - 2) % n])
            t2 = unit(pts[(end - 2) % n] - pts[(end + 2) % n])
        else:
            t1 = unit(seg[min(3, len(seg) - 1)] - seg[0])
            t2 = unit(seg[max(-4, -len(seg))] - seg[-1])
        fit_cubic(seg, t1, t2, FIT_ERROR, curves)
    return curves


# ---- output ----

def fmt(v):
    s = f'{v:.2f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def path_data(loops, ox, oy):
    parts = []
    for curves in loops:
        p0 = curves[0][0]
        # contour points sit on pixel centers; SVG pixels span [i, i + 1]
        parts.append(f'M{fmt(p0[0] - ox + 0.5)},{fmt(p0[1] - oy + 0.5)}')
        for c in curves:
            parts.append('C' + ' '.join(f'{fmt(p[0] - ox + 0.5)},{fmt(p[1] - oy + 0.5)}' for p in c[1:]))
        parts.append('Z')
    return ''.join(parts)


def flatten(curves, steps=12):
    t = np.linspace(0, 1, steps, endpoint=False)
    return np.concatenate([bezier(c, t) for c in curves])


def rasterize(loops, shape, scale=1):
    """Even-odd fill of the fitted loops, for checking the trace against the source."""
    acc = np.zeros((shape[0] * scale, shape[1] * scale), bool)
    for curves in loops:
        poly = [tuple(p) for p in (flatten(curves) * scale + scale / 2.0 - 0.5).tolist()]
        im = Image.new('1', (acc.shape[1], acc.shape[0]), 0)
        ImageDraw.Draw(im).polygon(poly, fill=1)
        acc ^= np.asarray(im, dtype=bool)
    return acc


def main(src, out_dir):
    darkness = load_darkness(src)
    ink = darkness > 0.5
    sticker = ~outside_of(dilate(ink, HALO))

    line_loops = [fit_loop(l) for l in contours(darkness)]
    sticker_field = np.asarray(Image.fromarray((sticker * 255).astype(np.uint8))
                               .filter(ImageFilter.GaussianBlur(2))).astype(np.float64) / 255.0
    sticker_loops = [fit_loop(l) for l in contours(sticker_field)]

    ys, xs = np.nonzero(sticker)
    x0, y0 = xs.min() - MARGIN, ys.min() - MARGIN
    w, h = xs.max() - xs.min() + 1 + 2 * MARGIN, ys.max() - ys.min() + 1 + 2 * MARGIN
    lines_d = path_data(line_loops, x0, y0)
    sticker_d = path_data(sticker_loops, x0, y0)

    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    with open(os.path.join(out_dir, 'rudy.svg'), 'w') as fh:
        fh.write(head + f'<path fill="#FFFFFF" d="{sticker_d}"/><path fill="#000000" d="{lines_d}"/></svg>\n')
    with open(os.path.join(out_dir, 'rudy-lines.svg'), 'w') as fh:
        fh.write(head + f'<path fill="#000000" d="{lines_d}"/></svg>\n')

    traced = rasterize(line_loops, ink.shape)
    union = (traced | ink).sum()
    iou = (traced & ink).sum() / union
    curves = sum(len(c) for c in line_loops)
    print(f'lines: {len(line_loops)} contours, {curves} curves, IoU vs source {iou:.4f}')
    print(f'sticker: {len(sticker_loops)} contours, {sum(len(c) for c in sticker_loops)} curves')
    print(f'viewBox 0 0 {w} {h}, offset {x0},{y0}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
