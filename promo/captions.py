#!/usr/bin/env python3
"""Caption plates v2 for EN promo reel. Left-aligned ink-runs, autofit, no overflow."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = '/tmp/textbehind/promo/caps'
W, H = 1080, 1920
CY = 1420
AMBER = (245, 158, 11)
WHITE = (250, 250, 250)
PF = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
MAXW = 1000

# (name, runs[(text, amber)], newline markers via '\n' inside run text)
PLATES = [
    ('p01', [('YOUR ', 0), ('AI AGENTS', 1)]),
    ('p02', [('CAN EDIT VIDEO NOW', 0)]),
    ('p03', [('WATCH THIS', 0)]),
    ('p04', [('TEXT-BEHIND-SUBJECT', 1)]),
    ('p05', [('BIG TEXT,', 0)]),
    ('p06', [('SANDWICHED', 1), (' BETWEEN', 0)]),
    ('p07', [('BACKGROUND × SUBJECT', 0)]),
    ('p08', [('AN AGENT', 0)]),
    ('p09', [('TRACKS EVERY FRAME', 1)]),
    ('p10', [('RENDERS BEHIND IT', 0)]),
    ('p11', [('ONE SKILL FILE', 1)]),
    ('p12', [('HAND IT TO ANY AGENT', 0)]),
    ('p13', [('CLAUDE, CODEX,', 0), ('\nYOUR OWN', 0)]),
]

def measure(runs, size):
    f = ImageFont.truetype(PF, size, index=8)
    im = Image.new('RGBA', (8, 8))
    d = ImageDraw.Draw(im)
    lines, cur, curw = [], [], 0.0
    for t, amber in runs:
        parts = t.split('\n')
        for j, part in enumerate(parts):
            if j > 0:
                lines.append((cur, curw)); cur, curw = [], 0.0
            if part:
                w = d.textlength(part, font=f)
                cur.append((part, amber)); curw += w
    lines.append((cur, curw))
    return f, lines, max(lw for _, lw in lines)

def render(name, runs):
    for size in range(56, 39, -2):
        f, lines, maxw = measure(runs, size)
        if maxw <= MAXW:
            break
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    lh = int(size * 1.18)
    y0 = CY - lh * (len(lines) - 1) / 2
    for li, (runs_l, lw) in enumerate(lines):
        y = y0 + li * lh
        x = (W - lw) / 2
        for t, amber in runs_l:
            w = d.textlength(t, font=f)
            col = AMBER if amber else WHITE
            d.text((x + 2, y + 2), t, font=f, fill=(0, 0, 0, 170), anchor='lm')
            d.text((x, y), t, font=f, fill=col + (255,), anchor='lm')
            x += w
    im.save(f'{OUT}/{name}.png')
    return size, len(lines)

os.makedirs(OUT, exist_ok=True)
for old in os.listdir(OUT):
    os.remove(f'{OUT}/{old}')
for name, runs in PLATES:
    size, nl = render(name, runs)
    print(f'{name} size={size} lines={nl}')
