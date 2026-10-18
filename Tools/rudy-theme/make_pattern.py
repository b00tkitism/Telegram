"""Generate the Rudy chat wallpaper pattern (res/raw/default_pattern.svg).

Run from the repository root (needs Pillow and numpy):

    python3 Tools/rudy-theme/make_pattern.py

Draws Rudy doodles (Rudy faces, ties, bows, stars, hearts, speech bubbles,
sparkles) on a seamless tile, repeats it over Telegram's 1440x2960 pattern
canvas and traces the result into filled SVG paths. The app tints the paths
and lays them over the wallpaper gradient, as with Telegram's own pattern.
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.getcwd()
ICONS = os.path.join(ROOT, 'Tools', 'rudy-icons')
sys.path.insert(0, ICONS)
import trace_svg  # noqa: E402

# Reuse the Rudy artwork loader from make_icons.py without running its generation step.
src_path = os.path.join(ICONS, 'make_icons.py')
src = open(src_path).read()
argv, sys.argv = sys.argv, [src_path, ROOT, '/tmp']
ns = {'__file__': src_path, '__name__': 'make_icons_defs'}
exec(compile(src[:src.index('# ---- outputs ----')], src_path, 'exec'), ns)
sys.argv = argv
STICKER, LINES, Place, coverage = ns['STICKER'], ns['LINES'], ns['Place'], ns['coverage']

W, H = 1440, 2960
TILE = 720
STROKE = 5


def doodle_tile(seed=11):
    rnd = random.Random(seed)
    tile = Image.new('L', (TILE, TILE), 0)

    def stamp(draw_fn, cx, cy, s, angle):
        g = Image.new('L', (s * 2, s * 2), 0)
        draw_fn(ImageDraw.Draw(g), s)
        g = g.rotate(angle, resample=Image.BICUBIC)
        for dx in (-TILE, 0, TILE):  # wrap so the tile repeats seamlessly
            for dy in (-TILE, 0, TILE):
                tile.paste(255, (int(cx - s + dx), int(cy - s + dy)), mask=g)

    def face(d, s):
        m = coverage(LINES.loops, (2 * s, 2 * s), Place.radius(STICKER, s, s, s * 0.85))
        d.bitmap((0, 0), m, fill=255)

    def tie(d, s):
        c, k = s, s / 44
        d.polygon([(c - 12 * k, c - 34 * k), (c + 12 * k, c - 34 * k), (c + 7 * k, c - 20 * k), (c - 7 * k, c - 20 * k)],
                  outline=255, width=STROKE)
        d.polygon([(c - 7 * k, c - 20 * k), (c + 7 * k, c - 20 * k), (c + 16 * k, c + 28 * k), (c, c + 42 * k),
                   (c - 16 * k, c + 28 * k)], outline=255, width=STROKE)

    def bubble(d, s):
        c, k = s, s / 44
        d.rounded_rectangle((c - 38 * k, c - 26 * k, c + 38 * k, c + 20 * k), radius=18 * k, outline=255, width=STROKE)
        d.line((c - 16 * k, c + 20 * k, c - 28 * k, c + 36 * k, c - 2 * k, c + 20 * k), fill=255, width=STROKE)

    def star(d, s):
        c, pts = s, []
        for i in range(10):
            r = (34 if i % 2 == 0 else 15) * s / 44
            a = math.pi / 2 + i * math.pi / 5
            pts.append((c + r * math.cos(a), c - r * math.sin(a)))
        d.polygon(pts, outline=255, width=STROKE)

    def heart(d, s):
        c, k = s, s / 44
        d.arc((c - 32 * k, c - 28 * k, c, c + 4 * k), 140, 360, fill=255, width=STROKE)
        d.arc((c, c - 28 * k, c + 32 * k, c + 4 * k), 180, 40, fill=255, width=STROKE)
        d.line((c - 30 * k, c - 4 * k, c, c + 30 * k, c + 30 * k, c - 4 * k), fill=255, width=STROKE)

    def sparkle(d, s):
        c, k = s, s / 44
        d.polygon([(c, c - 30 * k), (c + 8 * k, c - 8 * k), (c + 30 * k, c), (c + 8 * k, c + 8 * k), (c, c + 30 * k),
                   (c - 8 * k, c + 8 * k), (c - 30 * k, c), (c - 8 * k, c - 8 * k)], outline=255, width=STROKE)

    def bow(d, s):
        c, k = s, s / 44
        d.polygon([(c - 5 * k, c), (c - 34 * k, c - 20 * k), (c - 34 * k, c + 20 * k)], outline=255, width=STROKE)
        d.polygon([(c + 5 * k, c), (c + 34 * k, c - 20 * k), (c + 34 * k, c + 20 * k)], outline=255, width=STROKE)
        d.ellipse((c - 9 * k, c - 9 * k, c + 9 * k, c + 9 * k), outline=255, width=STROKE)

    shapes = [face, face, tie, bubble, star, heart, sparkle, bow]
    cells = 6
    step = TILE / cells
    for i in range(cells):
        for j in range(cells):
            fn = shapes[(i * cells + j + rnd.randrange(3)) % len(shapes)]
            cx = (i + 0.5) * step + rnd.uniform(-step * 0.18, step * 0.18) + (step / 2 if j % 2 else 0)
            cy = (j + 0.5) * step + rnd.uniform(-step * 0.18, step * 0.18)
            stamp(fn, cx % TILE, cy, 76 if fn is face else 50, rnd.uniform(-35, 35))
    return tile


def main():
    tile = doodle_tile()
    canvas = Image.new('L', (W, H), 0)
    for x in range(0, W, TILE):
        for y in range(0, H, TILE):
            canvas.paste(tile, (x, y))
    field = np.asarray(canvas.filter(ImageFilter.GaussianBlur(0.8))).astype(np.float64) / 255.0
    trace_svg.FIT_ERROR = 0.9
    loops = [np.array(trace_svg.fit_loop(l)) for l in trace_svg.contours(field)]
    parts = []
    for curves in loops:
        p0 = curves[0][0]
        d = [f'M{p0[0] + 0.5:.1f},{p0[1] + 0.5:.1f}']
        for c in curves:
            d.append('C' + ' '.join(f'{x + 0.5:.1f},{y + 0.5:.1f}' for x, y in c[1:]))
        parts.append('<path d="' + ''.join(d) + 'Z"/>')
    svg = (f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" viewBox="0 0 {W} {H}">\n'
           + '\n'.join(parts) + '\n</svg>\n')
    out = os.path.join(ROOT, 'TMessagesProj', 'src', 'main', 'res', 'raw', 'default_pattern.svg')
    with open(out, 'w') as fh:
        fh.write(svg)
    print(f'{len(loops)} contours, {sum(len(c) for c in loops)} curves, {len(svg) // 1024} KB')


if __name__ == '__main__':
    main()
