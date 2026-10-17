"""Generate Rudy launcher, themed and notification icons for the Android client.

Run from the repository root (needs Pillow and numpy):

    python3 Tools/rudy-icons/make_icons.py Tools/rudy-icons/rudy.webp \
        TMessagesProj/src/main/res /tmp/rudy-preview

The source is black line art on white. Besides the resources it writes a
preview sheet (rudy_icons_preview.png) into the last directory.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC, RES, PREVIEW = sys.argv[1], sys.argv[2], sys.argv[3]

DENSITIES = {'mdpi': 1.0, 'hdpi': 1.5, 'xhdpi': 2.0, 'xxhdpi': 3.0, 'xxxhdpi': 4.0}
HALO = 20            # white outline around the art, in source pixels
SAFE_RADIUS_DP = 31  # adaptive icon safe zone is a 33dp radius circle on a 108dp canvas


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
    """Pixels reachable from the image border without crossing walls."""
    padded = np.pad(walls, 1, constant_values=False)
    im = Image.fromarray(np.where(padded, 255, 0).astype(np.uint8)).copy()
    ImageDraw.floodfill(im, (0, 0), 128)
    return (np.asarray(im) == 128)[1:-1, 1:-1]


gray = np.asarray(Image.open(SRC).convert('L')).astype(np.float32) / 255.0
# the source is a screenshot crop with a stray dark line along its edge
gray[:4, :] = gray[-4:, :] = 1.0
gray[:, :4] = gray[:, -4:] = 1.0
ink = gray < 0.5
darkness = np.clip((1.0 - gray) * 1.15, 0, 1)

sticker = ~outside_of(dilate(ink, HALO))
silhouette = ~outside_of(dilate(ink, 3))

ys, xs = np.nonzero(sticker)
cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
art_radius = math.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max())


def rgba(rgb_value, alpha):
    h, w = alpha.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = (np.clip(rgb_value, 0, 1) * 255).astype(np.uint8)[..., None]
    out[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out, 'RGBA')


soft_sticker = np.asarray(Image.fromarray((sticker * 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255.0
STICKER = rgba(gray, soft_sticker)                       # black lines, white face, white outline
LINES = rgba(np.ones_like(gray), darkness)               # line art only, for themed icons


def place(layer, size, radius_px):
    """Center the art on a size x size canvas so its farthest pixel sits radius_px from the middle."""
    half = art_radius * (size / 2.0) / radius_px
    box = (int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half)))
    pad = max(0, -box[0], -box[1], box[2] - layer.width, box[3] - layer.height)
    canvas = Image.new('RGBA', (layer.width + 2 * pad, layer.height + 2 * pad), (255, 255, 255, 0))
    canvas.paste(layer, (pad, pad))
    crop = canvas.crop((box[0] + pad, box[1] + pad, box[2] + pad, box[3] + pad))
    return crop.resize((size, size), Image.LANCZOS)


# ---- backgrounds, mirroring drawable/icon_*_background_sa ----

def hex_rgba(c):
    c = c.lstrip('#')
    if len(c) == 6:
        c = 'ff' + c
    return np.array([int(c[i:i + 2], 16) for i in (2, 4, 6, 0)], np.float32) / 255.0


def ramp(t, colors):
    t = np.clip(t, 0, 1)[..., None]
    if len(colors) == 2:
        return colors[0] * (1 - t) + colors[1] * t
    lo = colors[0] * (1 - t * 2) + colors[1] * (t * 2)
    hi = colors[1] * (2 - t * 2) + colors[2] * (t * 2 - 1)
    return np.where(t < 0.5, lo, hi)


def linear(size, angle, *colors):
    a = angle % 360
    points = {0: ((0, 0), (1, 0)), 45: ((0, 1), (1, 0)), 90: ((0, 1), (0, 0)), 315: ((0, 0), (1, 1))}
    (x0, y0), (x1, y1) = points[a]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    dx, dy = x1 - x0, y1 - y0
    t = ((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy)
    return ramp(t, [hex_rgba(c) for c in colors])


def radial(size, center_x, center_y, radius, *colors):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    t = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / radius
    return ramp(t, [hex_rgba(c) for c in colors])


def over(bottom, top):
    a = top[..., 3:4]
    out = bottom.copy()
    out[..., :3] = top[..., :3] * a + bottom[..., :3] * (1 - a)
    out[..., 3:4] = a + bottom[..., 3:4] * (1 - a)
    return out


def to_image(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), 'RGBA')


def nox_background(size):
    return Image.open(os.path.join(RES, 'mipmap-xxxhdpi', 'icon_2_background_sa.png')).convert('RGBA') \
        .resize((size, size), Image.LANCZOS)


BACKGROUNDS = {
    'ic_launcher': lambda s: Image.new('RGBA', (s, s), (255, 255, 255, 255)),
    'icon_2_launcher': nox_background,
    'icon_3_launcher': lambda s: to_image(over(over(
        linear(s, -45, '#FC5DDC', '#9484FF'),
        radial(s, 1, 0, 0.65, '#ffF4817F', '#00F4817F')),
        radial(s, 0, 1, 0.80, '#ff5381FE', '#925381FE', '#005381FE'))),
    'icon_4_launcher': lambda s: to_image(linear(s, 45, '#137BE5', '#02A3CC', '#00DC6B')),
    'icon_5_launcher': lambda s: to_image(linear(s, 45, '#9651FF', '#3A8FFF', '#49F9FA')),
    'icon_6_launcher': lambda s: to_image(linear(s, 90, '#1D93D2', '#37AFE3')),
}


def circle_mask(size, radius):
    scale = 4
    m = Image.new('L', (size * scale, size * scale), 0)
    c = size * scale / 2.0
    r = radius * scale
    ImageDraw.Draw(m).ellipse((c - r, c - r, c + r, c + r), fill=255)
    return m.resize((size, size), Image.LANCZOS)


def legacy_icon(name, size):
    """Pre-Android 8 launcher icon: art on a round badge."""
    radius = size * 0.46
    badge = BACKGROUNDS[name](size)
    badge.alpha_composite(place(STICKER, size, radius * 0.9))
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    if name == 'ic_launcher':
        out.paste((224, 224, 224, 255), mask=circle_mask(size, radius))
        radius -= max(1.0, size / 48.0)
    out.paste(badge, mask=circle_mask(size, radius))
    return out


def notification_icon(size):
    """Status bar icon: white head silhouette with the face features cut out."""
    near_edge = dilate(~silhouette, 24)
    # only the eyebrows, eyes and nose; hair, bow and suit lines turn to noise at 24dp
    face_band = np.zeros_like(ink)
    face_band[590:800, :] = True
    features = dilate(ink & ~near_edge & face_band, 7)
    alpha = (silhouette & ~features).astype(np.float32)
    layer = rgba(np.ones_like(gray), alpha)
    return place(layer, size, size * 0.46)


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.webp'):
        img.save(path, lossless=True, quality=100, method=6)
    else:
        img.save(path, optimize=True)


for density, scale in DENSITIES.items():
    folder = os.path.join(RES, 'mipmap-' + density)
    fg_size = int(round(108 * scale))
    fg = place(STICKER, fg_size, SAFE_RADIUS_DP * scale)
    mono = place(LINES, fg_size, SAFE_RADIUS_DP * scale)
    for name in ('icon_foreground', 'icon_foreground_round', 'icon_foreground_sa'):
        save(fg, os.path.join(folder, name + '.png'))
    save(mono, os.path.join(folder, 'icon_monochrome.png'))
    for name in BACKGROUNDS:
        icon = legacy_icon(name, int(round(48 * scale)))
        save(icon, os.path.join(folder, name + '.png'))
        save(icon, os.path.join(folder, name + '_round.png'))
        if name == 'ic_launcher':
            # account, call and conversation-person icon
            save(icon, os.path.join(RES, 'drawable-' + density, 'ic_launcher_dr.webp'))
    if density != 'xxxhdpi':
        save(notification_icon(int(round(24 * scale))),
             os.path.join(RES, 'drawable-' + density, 'notification.webp'))


# ---- preview sheet ----

def squircle_mask(size):
    scale = 4
    n = size * scale
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    u = np.abs((xx + 0.5) / n * 2 - 1)
    v = np.abs((yy + 0.5) / n * 2 - 1)
    m = Image.fromarray(((u ** 4 + v ** 4) <= 1).astype(np.uint8) * 255)
    return m.resize((size, size), Image.LANCZOS)


def adaptive_preview(name, size, mask):
    canvas = int(size * 108 / 72)
    bg = BACKGROUNDS[name](canvas)
    bg.alpha_composite(place(STICKER, canvas, SAFE_RADIUS_DP * canvas / 108))
    off = (canvas - size) // 2
    visible = bg.crop((off, off, off + size, off + size))
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(visible, mask=mask)
    return out


os.makedirs(PREVIEW, exist_ok=True)
cell, pad = 192, 24
rows = []
for mask_fn in (lambda s: circle_mask(s, s / 2), squircle_mask):
    row = Image.new('RGBA', (len(BACKGROUNDS) * (cell + pad) + pad, cell + 2 * pad), (238, 240, 243, 255))
    for i, name in enumerate(BACKGROUNDS):
        row.alpha_composite(adaptive_preview(name, cell, mask_fn(cell)), (pad + i * (cell + pad), pad))
    rows.append(row)

themed = Image.new('RGBA', rows[0].size, (36, 40, 52, 255))
for i, (fill, tint) in enumerate([((211, 227, 253), (4, 30, 73)), ((4, 30, 73), (211, 227, 253)),
                                  ((255, 216, 228), (49, 17, 29)), ((49, 17, 29), (255, 216, 228))]):
    canvas = int(cell * 108 / 72)
    m = place(LINES, canvas, SAFE_RADIUS_DP * canvas / 108)
    tile = Image.new('RGBA', (canvas, canvas), fill + (255,))
    tile.paste(tint + (255,), mask=m.split()[3])
    off = (canvas - cell) // 2
    tile = tile.crop((off, off, off + cell, off + cell))
    out = Image.new('RGBA', (cell, cell), (0, 0, 0, 0))
    out.paste(tile, mask=circle_mask(cell, cell / 2))
    themed.alpha_composite(out, (pad + i * (cell + pad), pad))
rows.append(themed)

small = Image.new('RGBA', rows[0].size, (255, 255, 255, 255))
x = pad
for name in BACKGROUNDS:
    for s in (48, 96):
        small.alpha_composite(legacy_icon(name, s), (x, pad + (96 - s)))
        x += s + 12
bar = Image.new('RGBA', (300, 72), (24, 24, 27, 255))
for i, s in enumerate((24, 36, 48, 72)):
    bar.alpha_composite(notification_icon(s), (8 + sum((24, 36, 48, 72)[:i]) + 12 * i, (72 - s) // 2))
small.alpha_composite(bar, (rows[0].width - 300 - pad, pad + 24))
rows.append(small)

sheet = Image.new('RGBA', (rows[0].width, sum(r.height for r in rows)), (255, 255, 255, 255))
y = 0
for r in rows:
    sheet.alpha_composite(r, (0, y))
    y += r.height
sheet.convert('RGB').save(os.path.join(PREVIEW, 'rudy_icons_preview.png'))
print('art radius', art_radius, 'center', cx, cy)
