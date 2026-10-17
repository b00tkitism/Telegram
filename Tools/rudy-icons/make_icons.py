"""Generate every Rudy brand asset of the Android client from rudy.svg.

Run from the repository root (needs Pillow and numpy):

    python3 Tools/rudy-icons/make_icons.py . /tmp/rudy-preview

rudy.svg holds two paths: the white sticker shape (face fill plus a thin
outline) and the black line art. Edit it, or re-trace it from a bitmap with
trace_svg.py, then run this script again. It writes:

  launcher icons     mipmap-*/icon_foreground*.png, ic_launcher*.png, icon_*_launcher*.png
  themed icon        drawable/icon_monochrome.xml
  splash screen      drawable/tg_splash_320.xml
  status bar, menu   drawable/notification.xml, drawable/menu_invit_telegram.xml
  account and calls  drawable-*/ic_launcher_dr.webp
  intro animation    drawable-*/intro_tg_plane.webp
  terms dialog       drawable-*/logo_middle.*
  QR code logos      raw/qr_logo.svg, raw/qr_code_logo.json, raw/plane_logo_plain.json
  wordmarks          drawable/telegram_logo.xml, drawable/telegram_logo_2.xml

and a preview sheet, rudy_icons_preview.png, into the preview directory.
"""
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trace_svg  # noqa: E402  contour tracing and Bezier fitting

ROOT, PREVIEW = sys.argv[1], sys.argv[2]
RES = os.path.join(ROOT, 'TMessagesProj', 'src', 'main', 'res')
FONT = os.path.join(ROOT, 'TMessagesProj', 'src', 'main', 'assets', 'fonts', 'rmedium.ttf')

DENSITIES = {'mdpi': 1.0, 'hdpi': 1.5, 'xhdpi': 2.0, 'xxhdpi': 3.0, 'xxxhdpi': 4.0}
SAFE_RADIUS_DP = 31  # adaptive icon safe zone is a 33dp radius circle on a 108dp canvas
INTRO_BOX_DP = (124, 107)  # matches telegram_plane in jni/intro/IntroRenderer.c


# ---- geometry ----

def parse_d(d):
    loops, cur = [], None
    for cmd, args in re.findall(r'([MCZ])([^MCZ]*)', d):
        nums = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', args)]
        if cmd == 'M':
            cur = [nums[0:2]]
        elif cmd == 'C':
            for i in range(0, len(nums), 6):
                cur += [nums[i:i + 2], nums[i + 2:i + 4], nums[i + 4:i + 6]]
        else:
            pts = np.array(cur)
            loops.append(np.array([pts[i:i + 4] for i in range(0, len(pts) - 1, 3)]))
    return loops


class Shape:
    """Closed cubic loops plus the circle and box that enclose them."""

    def __init__(self, loops):
        self.loops = loops
        pts = np.concatenate([trace_svg.flatten(c, 16) for c in loops])
        self.x0, self.y0 = pts.min(axis=0)
        self.x1, self.y1 = pts.max(axis=0)
        self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        self.radius = float(np.sqrt(((pts - [self.cx, self.cy]) ** 2).sum(axis=1)).max())
        self.width, self.height = self.x1 - self.x0, self.y1 - self.y0


class Place:
    """Maps shape coordinates to a target: (x - cx) * scale + tx."""

    def __init__(self, ref, tx, ty, scale):
        self.cx, self.cy, self.tx, self.ty, self.scale = ref.cx, ref.cy, tx, ty, scale

    @staticmethod
    def radius(ref, tx, ty, r):
        return Place(ref, tx, ty, r / ref.radius)

    @staticmethod
    def box(ref, tx, ty, w, h):
        return Place(ref, tx, ty, min(w / ref.width, h / ref.height))

    def apply(self, pts):
        return (pts - [self.cx, self.cy]) * self.scale + [self.tx, self.ty]

    def group_attrs(self):
        return (f'android:translateX="{fmt(self.tx - self.cx * self.scale)}" '
                f'android:translateY="{fmt(self.ty - self.cy * self.scale)}" '
                f'android:scaleX="{fmt(self.scale, 6)}" android:scaleY="{fmt(self.scale, 6)}"')


def fmt(v, digits=2):
    s = f'{v:.{digits}f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def reverse(loop):
    return loop[::-1, ::-1]


def path_data(loops, place=None):
    parts = []
    for curves in loops:
        c = curves if place is None else np.array([place.apply(seg) for seg in curves])
        parts.append(f'M{fmt(c[0][0][0])},{fmt(c[0][0][1])}')
        for seg in c:
            parts.append('C' + ' '.join(f'{fmt(x)},{fmt(y)}' for x, y in seg[1:]))
        parts.append('Z')
    return ''.join(parts)


def load_svg(path):
    text = open(path).read()
    paths = dict(re.findall(r'<path fill="(#[0-9A-Fa-f]{6})" d="([^"]+)"/>', text))
    return Shape(parse_d(paths['#FFFFFF'])), Shape(parse_d(paths['#000000']))


STICKER, LINES = load_svg(os.path.join(HERE, 'rudy.svg'))


# ---- rasterizing ----

def coverage(loops, size, place, ss=4):
    """Antialiased nonzero-equivalent fill (loops never overlap, so even-odd works)."""
    w, h = size
    acc = np.zeros((h * ss, w * ss), bool)
    for curves in loops:
        pts = place.apply(trace_svg.flatten(curves, 16)) * ss - 0.5
        im = Image.new('1', (w * ss, h * ss), 0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in pts.tolist()], fill=1)
        acc ^= np.asarray(im, dtype=bool)
    return Image.fromarray(acc.astype(np.uint8) * 255).resize((w, h), Image.BOX)


def art(size, place, sticker=True, lines_color=(0, 0, 0)):
    out = Image.new('RGBA', size, (255, 255, 255, 0))
    if sticker:
        out.paste((255, 255, 255, 255), mask=coverage(STICKER.loops, size, place))
    out.paste(lines_color + (255,), mask=coverage(LINES.loops, size, place))
    return out


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.webp'):
        img.save(path, lossless=True, quality=100, method=6)
    else:
        img.save(path, optimize=True)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)


# ---- launcher backgrounds, mirroring drawable/icon_*_background_sa ----

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
    points = {0: ((0, 0), (1, 0)), 45: ((0, 1), (1, 0)), 90: ((0, 1), (0, 0)), 315: ((0, 0), (1, 1))}
    (x0, y0), (x1, y1) = points[angle % 360]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    dx, dy = x1 - x0, y1 - y0
    return ramp(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), [hex_rgba(c) for c in colors])


def radial(size, center_x, center_y, radius, *colors):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    return ramp(np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / radius, [hex_rgba(c) for c in colors])


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
    ss = 4
    m = Image.new('L', (size * ss, size * ss), 0)
    c, r = size * ss / 2.0, radius * ss
    ImageDraw.Draw(m).ellipse((c - r, c - r, c + r, c + r), fill=255)
    return m.resize((size, size), Image.LANCZOS)


def badge(name, size, radius):
    """Art on a round badge: pre-Android 8 launcher icons and the terms dialog logo."""
    inner = BACKGROUNDS[name](size)
    inner.alpha_composite(art((size, size), Place.radius(STICKER, size / 2, size / 2, radius * 0.9)))
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    if name == 'ic_launcher':
        out.paste((224, 224, 224, 255), mask=circle_mask(size, radius))
        radius -= max(1.0, size / 48.0)
    out.paste(inner, mask=circle_mask(size, radius))
    return out


# ---- small single-color glyph: head silhouette with the face cut out ----

def silhouette_shape():
    pad = 40
    w, h = int(math.ceil(LINES.x1)) + 2 * pad, int(math.ceil(LINES.y1)) + 2 * pad
    shift = Place(LINES, LINES.cx + pad, LINES.cy + pad, 1.0)
    ink = np.asarray(coverage(LINES.loops, (w, h), shift, ss=2)) > 127
    head = ~trace_svg.outside_of(trace_svg.dilate(ink, 3))
    near_edge = trace_svg.dilate(~head, 24)
    # keep only the eyebrows, eyes and nose; hair, bow and suit lines turn to noise at 24dp
    band = np.zeros_like(ink)
    band[pad + 333:pad + 543, :] = True
    features = trace_svg.dilate(ink & ~near_edge & band, 7)
    field = np.asarray(Image.fromarray(((head & ~features) * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(1.5))).astype(np.float64) / 255.0
    loops = [np.array(trace_svg.fit_loop(l)) for l in trace_svg.contours(field)]
    return Shape([c - [pad - 0.5, pad - 0.5] for c in loops])


def wordmark_shape(text='Rudy'):
    font = ImageFont.truetype(FONT, 480)
    x0, y0, x1, y1 = font.getbbox(text)
    im = Image.new('L', (x1 - x0 + 80, y1 - y0 + 80), 0)
    ImageDraw.Draw(im).text((40 - x0, 40 - y0), text, font=font, fill=255)
    field = np.asarray(im).astype(np.float64) / 255.0
    return Shape([np.array(trace_svg.fit_loop(l)) for l in trace_svg.contours(field)])


# ---- vector drawables, SVG, Lottie ----

def vector_xml(width_dp, height_dp, viewport, place, paths):
    body = ''.join(f'\n        <path\n            android:fillColor="{fill}"\n'
                   f'            android:pathData="{path_data(loops)}" />' for fill, loops in paths)
    return (f'<?xml version="1.0" encoding="utf-8"?>\n'
            f'<!-- Generated by Tools/rudy-icons/make_icons.py -->\n'
            f'<vector xmlns:android="http://schemas.android.com/apk/res/android"\n'
            f'    android:width="{fmt(width_dp, 4)}dp"\n    android:height="{fmt(height_dp, 4)}dp"\n'
            f'    android:viewportWidth="{fmt(viewport[0])}"\n    android:viewportHeight="{fmt(viewport[1])}">\n'
            f'    <group {place.group_attrs()}>{body}\n    </group>\n</vector>\n')


def splash_xml():
    place = Place.radius(STICKER, 160, 160, 96)
    fade = ('            <objectAnimator android:propertyName="fillAlpha" android:startOffset="200" '
            'android:duration="100" android:valueFrom="0" android:valueTo="1" android:valueType="floatType" '
            'android:interpolator="@android:interpolator/fast_out_slow_in"/>')
    grow = '\n'.join(
        f'                <objectAnimator android:propertyName="{p}" android:startOffset="200" '
        f'android:duration="300" android:valueFrom="0.5" android:valueTo="1" android:valueType="floatType" '
        f'android:interpolator="@android:anim/overshoot_interpolator"/>' for p in ('scaleX', 'scaleY'))
    return f'''<?xml version="1.0" encoding="utf-8"?>
<!-- Generated by Tools/rudy-icons/make_icons.py -->
<animated-vector
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:aapt="http://schemas.android.com/aapt">
    <aapt:attr name="android:drawable">
        <vector
            android:width="320dp"
            android:height="320dp"
            android:viewportWidth="320"
            android:viewportHeight="320">
            <group
                android:name="fg"
                android:pivotX="160"
                android:pivotY="160"
                android:scaleX="0.5"
                android:scaleY="0.5">
                <group {place.group_attrs()}>
                    <path
                        android:name="sticker"
                        android:fillColor="#FFFFFF"
                        android:fillAlpha="0"
                        android:pathData="{path_data(STICKER.loops)}" />
                    <path
                        android:name="lines"
                        android:fillColor="#000000"
                        android:fillAlpha="0"
                        android:pathData="{path_data(LINES.loops)}" />
                </group>
            </group>
        </vector>
    </aapt:attr>
    <target android:name="sticker">
        <aapt:attr name="android:animation">
{fade}
        </aapt:attr>
    </target>
    <target android:name="lines">
        <aapt:attr name="android:animation">
{fade}
        </aapt:attr>
    </target>
    <target android:name="fg">
        <aapt:attr name="android:animation">
            <set>
{grow}
            </set>
        </aapt:attr>
    </target>
</animated-vector>
'''


def qr_logo_svg():
    place = Place.radius(STICKER, 75, 75, 66)
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<svg width="150px" height="150px" viewBox="0 0 150 150" version="1.1" xmlns="http://www.w3.org/2000/svg">'
            '<circle fill="#FFFFFF" cx="75" cy="75" r="75" />'
            f'<path fill="#FFFFFF" d="{path_data(STICKER.loops, place)}" />'
            f'<path fill="#000000" d="{path_data(LINES.loops, place)}" />'
            '</svg>\n')


def lottie_group(name, loops, place, color):
    items = []
    for i, curves in enumerate(loops):
        c = np.array([place.apply(seg) for seg in curves])
        v = c[:, 0]
        o = c[:, 1] - c[:, 0]
        prev = np.roll(c, 1, axis=0)
        i_tan = prev[:, 2] - prev[:, 3]
        r = lambda a: np.round(a, 2).tolist()  # noqa: E731
        items.append({'ty': 'sh', 'nm': f'Path {i + 1}', 'ks': {'a': 0, 'k': {
            'i': r(i_tan), 'o': r(o), 'v': r(v), 'c': True}}})
    items.append({'ty': 'fl', 'nm': 'Fill', 'c': {'a': 0, 'k': list(color) + [1]}, 'o': {'a': 0, 'k': 100}, 'r': 1})
    items.append({'ty': 'tr', 'p': {'a': 0, 'k': [0, 0]}, 'a': {'a': 0, 'k': [0, 0]}, 's': {'a': 0, 'k': [100, 100]},
                  'r': {'a': 0, 'k': 0}, 'o': {'a': 0, 'k': 100}, 'sk': {'a': 0, 'k': 0}, 'sa': {'a': 0, 'k': 0}})
    return {'ty': 'gr', 'nm': name, 'it': items}


def lottie_json(groups):
    """Static 512x512 animation with the frame range the app expects (20..74 at 60fps)."""
    layer = {'ddd': 0, 'ind': 1, 'ty': 4, 'nm': 'Rudy', 'sr': 1,
             'ks': {'o': {'a': 0, 'k': 100}, 'r': {'a': 0, 'k': 0}, 'p': {'a': 0, 'k': [256, 256, 0]},
                    'a': {'a': 0, 'k': [0, 0, 0]}, 's': {'a': 0, 'k': [100, 100, 100]}},
             'ao': 0, 'shapes': groups, 'ip': 0, 'op': 180, 'st': 0, 'bm': 0}
    return json.dumps({'tgs': 1, 'v': '5.5.2', 'fr': 60, 'ip': 20, 'op': 74, 'w': 512, 'h': 512, 'nm': 'Rudy',
                       'ddd': 0, 'assets': [], 'layers': [layer]}, separators=(',', ':')) + '\n'


# ---- outputs ----

def remove(path):
    if os.path.exists(path):
        os.remove(path)


SILHOUETTE = silhouette_shape()
WORDMARK = wordmark_shape()

for density, scale in DENSITIES.items():
    folder = os.path.join(RES, 'mipmap-' + density)
    fg_size = int(round(108 * scale))
    fg = art((fg_size, fg_size), Place.radius(STICKER, fg_size / 2, fg_size / 2, SAFE_RADIUS_DP * scale))
    for name in ('icon_foreground', 'icon_foreground_round', 'icon_foreground_sa'):
        save(fg, os.path.join(folder, name + '.png'))
    remove(os.path.join(folder, 'icon_monochrome.png'))
    for name in BACKGROUNDS:
        size = int(round(48 * scale))
        icon = badge(name, size, size * 0.46)
        save(icon, os.path.join(folder, name + '.png'))
        save(icon, os.path.join(folder, name + '_round.png'))
        if name == 'ic_launcher':
            save(icon, os.path.join(RES, 'drawable-' + density, 'ic_launcher_dr.webp'))
    if density == 'xxxhdpi':
        continue
    drawables = os.path.join(RES, 'drawable-' + density)
    remove(os.path.join(drawables, 'notification.webp'))
    remove(os.path.join(drawables, 'menu_invit_telegram.webp'))
    box = (int(round(INTRO_BOX_DP[0] * scale)), int(round(INTRO_BOX_DP[1] * scale)))
    save(art(box, Place.box(STICKER, box[0] / 2, box[1] / 2, box[0], box[1])),
         os.path.join(drawables, 'intro_tg_plane.webp'))
    for ext in ('png', 'webp'):
        path = os.path.join(drawables, 'logo_middle.' + ext)
        if os.path.exists(path):
            size = int(round(68 * scale))
            save(badge('ic_launcher', size, size / 2 - 0.5), path)

write(os.path.join(RES, 'drawable', 'icon_monochrome.xml'),
      vector_xml(108, 108, (108, 108), Place.radius(STICKER, 54, 54, SAFE_RADIUS_DP), [('#FFFFFF', LINES.loops)]))
write(os.path.join(RES, 'drawable', 'notification.xml'),
      vector_xml(24, 24, (24, 24), Place.radius(SILHOUETTE, 12, 12, 11), [('#FFFFFF', SILHOUETTE.loops)]))
write(os.path.join(RES, 'drawable', 'menu_invit_telegram.xml'),
      vector_xml(24, 24, (24, 24), Place.radius(SILHOUETTE, 12, 12, 10.5), [('#FFFFFF', SILHOUETTE.loops)]))
write(os.path.join(RES, 'drawable', 'tg_splash_320.xml'), splash_xml())
write(os.path.join(RES, 'raw', 'qr_logo.svg'), qr_logo_svg())
write(os.path.join(RES, 'raw', 'qr_code_logo.json'), lottie_json([
    lottie_group('Lines', LINES.loops, Place.radius(STICKER, 0, 0, 236), (0, 0, 0)),
    lottie_group('Sticker', STICKER.loops, Place.radius(STICKER, 0, 0, 236), (1, 1, 1))]))
# one white shape with the lines cut out, for use on colored backgrounds
write(os.path.join(RES, 'raw', 'plane_logo_plain.json'), lottie_json([
    lottie_group('Rudy', [reverse(l) for l in STICKER.loops] + LINES.loops,
                 Place.radius(STICKER, 0, 0, 220), (1, 1, 1))]))

# the dialogs title sizes itself to the drawable; the intro title draws into fixed 115x26.33dp bounds
word_h = 20.408922
write(os.path.join(RES, 'drawable', 'telegram_logo_2.xml'),
      vector_xml(word_h * WORDMARK.width / WORDMARK.height, word_h, (WORDMARK.width, WORDMARK.height),
                 Place(WORDMARK, WORDMARK.width / 2, WORDMARK.height / 2, 1.0), [('#FFFFFFFF', WORDMARK.loops)]))
write(os.path.join(RES, 'drawable', 'telegram_logo.xml'),
      vector_xml(83, 19, (249, 57), Place.box(WORDMARK, 124.5, 28.5, 249, 57), [('#FFFFFFFF', WORDMARK.loops)]))


# ---- preview sheet ----

def squircle_mask(size):
    ss = 4
    n = size * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    u, v = np.abs((xx + 0.5) / n * 2 - 1), np.abs((yy + 0.5) / n * 2 - 1)
    return Image.fromarray(((u ** 4 + v ** 4) <= 1).astype(np.uint8) * 255).resize((size, size), Image.LANCZOS)


def adaptive_preview(name, size, mask):
    canvas = int(size * 108 / 72)
    bg = BACKGROUNDS[name](canvas)
    bg.alpha_composite(art((canvas, canvas), Place.radius(STICKER, canvas / 2, canvas / 2,
                                                          SAFE_RADIUS_DP * canvas / 108)))
    off = (canvas - size) // 2
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(bg.crop((off, off, off + size, off + size)), mask=mask)
    return out


def glyph(shape, size, radius, color, bg):
    tile = Image.new('RGBA', (size, size), bg)
    tile.paste(color, mask=coverage(shape.loops, (size, size), Place.radius(shape, size / 2, size / 2, radius)))
    return tile


os.makedirs(PREVIEW, exist_ok=True)
cell, pad = 192, 24
width = len(BACKGROUNDS) * (cell + pad) + pad
rows = []
for mask_fn in (lambda s: circle_mask(s, s / 2), squircle_mask):
    row = Image.new('RGBA', (width, cell + 2 * pad), (238, 240, 243, 255))
    for i, name in enumerate(BACKGROUNDS):
        row.alpha_composite(adaptive_preview(name, cell, mask_fn(cell)), (pad + i * (cell + pad), pad))
    rows.append(row)

row = Image.new('RGBA', (width, cell + 2 * pad), (36, 40, 52, 255))
for i, (fill, tint) in enumerate([((211, 227, 253, 255), (4, 30, 73, 255)), ((4, 30, 73, 255), (211, 227, 253, 255))]):
    canvas = int(cell * 108 / 72)
    tile = Image.new('RGBA', (canvas, canvas), fill)
    tile.paste(tint, mask=coverage(LINES.loops, (canvas, canvas),
                                   Place.radius(STICKER, canvas / 2, canvas / 2, SAFE_RADIUS_DP * canvas / 108)))
    off = (canvas - cell) // 2
    out = Image.new('RGBA', (cell, cell), (0, 0, 0, 0))
    out.paste(tile.crop((off, off, off + cell, off + cell)), mask=circle_mask(cell, cell / 2))
    row.alpha_composite(out, (pad + i * (cell + pad), pad))
for i, bg in enumerate([(255, 255, 255, 255), (31, 39, 50, 255)]):  # splash, light and dark
    tile = Image.new('RGBA', (cell, cell), bg)
    tile.alpha_composite(art((cell, cell), Place.radius(STICKER, cell / 2, cell / 2, cell * 96 / 320)))
    row.alpha_composite(tile, (pad + (2 + i) * (cell + pad), pad))
sphere = Image.new('RGBA', (cell, cell), (255, 255, 255, 255))  # intro: art over the blue sphere
sphere.paste((51, 144, 236, 255), mask=circle_mask(cell, cell / 2))
k = cell / 150
sphere.alpha_composite(art((cell, cell), Place.box(STICKER, cell / 2, cell / 2, INTRO_BOX_DP[0] * k, INTRO_BOX_DP[1] * k)))
row.alpha_composite(sphere, (pad + 4 * (cell + pad), pad))
qr = Image.new('RGBA', (cell, cell), (255, 255, 255, 255))
qr.alpha_composite(art((cell, cell), Place.radius(STICKER, cell / 2, cell / 2, cell * 66 / 150)))
row.alpha_composite(qr, (pad + 5 * (cell + pad), pad))
rows.append(row)

row = Image.new('RGBA', (width, 150), (255, 255, 255, 255))
x = pad
for name in BACKGROUNDS:
    for s in (48, 96):
        row.alpha_composite(badge(name, s, s * 0.46), (x, pad + (96 - s)))
        x += s + 12
rows.append(row)

row = Image.new('RGBA', (width, 150), (24, 24, 27, 255))
x = pad
for s in (24, 48, 72, 96):
    row.alpha_composite(glyph(SILHOUETTE, s, s * 11 / 24, (255, 255, 255, 255), (24, 24, 27, 255)), (x, pad))
    x += s + 16
knock = Image.new('RGBA', (96, 96), (51, 144, 236, 255))
knock.paste((255, 255, 255, 255), mask=coverage([reverse(l) for l in STICKER.loops] + LINES.loops, (96, 96),
                                                Place.radius(STICKER, 48, 48, 41)))
row.alpha_composite(knock, (x + 16, pad))
title_h = 40
ww = int(title_h * WORDMARK.width / WORDMARK.height)
word = Image.new('RGBA', (ww, title_h), (24, 24, 27, 255))
word.paste((255, 255, 255, 255), mask=coverage(WORDMARK.loops, (ww, title_h),
                                               Place.box(WORDMARK, ww / 2, title_h / 2, ww, title_h)))
row.alpha_composite(word, (x + 140, pad + 28))
rows.append(row)

sheet = Image.new('RGBA', (width, sum(r.height for r in rows)), (255, 255, 255, 255))
y = 0
for r in rows:
    sheet.alpha_composite(r, (0, y))
    y += r.height
sheet.convert('RGB').save(os.path.join(PREVIEW, 'rudy_icons_preview.png'))
print('sticker radius', round(STICKER.radius, 1), 'silhouette curves', sum(len(c) for c in SILHOUETTE.loops),
      'wordmark curves', sum(len(c) for c in WORDMARK.loops))
