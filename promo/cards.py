#!/usr/bin/env python3
"""Promo reel cards: terminal typing card (L5) + endcard (L6). 1080x1920 PNG sequences."""
import os, shutil, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
FPS = 24
AMBER = (245, 158, 11)
WHITE = (240, 240, 240)
GREY = (150, 150, 155)

MENLO = '/System/Library/Fonts/Menlo.ttc'
PF = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
LOGO = '/Users/cloofsmacbookpro/Pictures/cloofsiu.com/logo/Cloof Logo/Transparent/Cloof Logo_white_transparent_工作區域 1.png'

f_term = ImageFont.truetype(MENLO, 34)
f_ttitle = ImageFont.truetype(MENLO, 26)
f_eyebrow = ImageFont.truetype(PF, 34, index=8)
f_big = ImageFont.truetype(PF, 88, index=8)
f_mid = ImageFont.truetype(PF, 46, index=4)
f_follow = ImageFont.truetype(PF, 40, index=4)


def new_frame():
    return Image.new('RGB', (W, H), (8, 8, 10))


def draw_terminal(frame, visible_lines, cursor_on, cursor_pos):
    d = ImageDraw.Draw(frame)
    # window
    x0, y0, x1, y1 = 55, 545, 1025, 1345
    d.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(17, 17, 20), outline=(70, 70, 78), width=2)
    # title bar
    d.rounded_rectangle([x0, y0, x1, y0 + 58], radius=22, fill=(28, 28, 32))
    d.rectangle([x0, y0 + 34, x1, y0 + 58], fill=(28, 28, 32))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([x0 + 26 + i * 40, y0 + 19, x0 + 46 + i * 40, y0 + 39], fill=c)
    d.text(((x0 + x1) / 2, y0 + 29), 'cloof@agent — zsh', font=f_ttitle, fill=(160, 160, 166), anchor='mm')
    # lines
    ty = y0 + 92
    lh = 50
    for li, (line, kind) in enumerate(visible_lines):
        x = x0 + 34
        yy = ty + li * lh
        if kind == 'prompt':
            d.text((x, yy), line, font=f_term, fill=AMBER, anchor='lm')
        elif kind == 'args':
            d.text((x + 40, yy), line, font=f_term, fill=(200, 200, 208), anchor='lm')
        elif kind == 'ok':
            d.text((x, yy), line, font=f_term, fill=(120, 200, 120), anchor='lm')
        elif kind == 'done':
            d.text((x, yy), line, font=f_term, fill=AMBER, anchor='lm')
        if cursor_pos == li and cursor_on:
            wpx = d.textlength(line, font=f_term)
            cx = x + (40 if kind == 'args' else 0) + wpx + 4
            d.rectangle([cx, yy - 17, cx + 18, yy + 17], fill=WHITE)
    return frame


def build_terminal(total=5.918):
    k = total / 5.918
    n = int(total * FPS)
    # full script: (text, kind, type_start, type_cps)  — typed lines; result lines pop at fixed t
    # times scale by k (= total/5.918), cps by 1/k so typing pace matches the faster tempo
    typed = [
        ('$ agent run text-behind-subject \\', 'prompt', 0.25 * k, 24 / k),
        ('    --input clip.mp4 --text "AGENTS"', 'args', 1.75 * k, 30 / k),
    ]
    pops = [
        ('', None, 3.30 * k),
        ('▸ tracking subject ......... ok   [240 frames]', 'ok', 3.55 * k),
        ('▸ depth mask propagated .... ok', 'ok', 4.10 * k),
        ('▸ compositing text ......... ok', 'ok', 4.65 * k),
        ('✔ output/render.mp4  — 8.0s · 24fps', 'done', 5.15 * k),
    ]
    os.makedirs(f'{OUT}/term', exist_ok=True)
    for f in range(n):
        t = f / FPS
        vis = []
        cur_line, cur_txt = 0, ''
        for text, kind, ts, cps in typed:
            if t < ts:
                break
            nch = min(len(text), int((t - ts) * cps))
            vis.append((text[:nch], kind))
            if nch < len(text):
                cur_line, cur_txt = len(vis) - 1, text[:nch]
                break
            cur_line = len(vis)
        for text, kind, tp in pops:
            if t >= tp:
                vis.append((text, kind))
                cur_line = len(vis) - 1
        cursor_on = (f % 14) < 8
        if t >= 5.15 * k:  # typing finished: cursor after done-line only briefly
            cursor_on = cursor_on and t < 5.5 * k
        fr = draw_terminal(new_frame(), vis, cursor_on, cur_line)
        fr.save(f'{OUT}/term/f{f:04d}.png')
    print('terminal frames:', n)


def build_endcard(total=3.43):
    n = int(total * FPS)
    im = new_frame()
    d = ImageDraw.Draw(im)
    # eyebrow
    d.text((540, 500), 'TEXT-BEHIND-SUBJECT · SKILL', font=f_eyebrow, fill=AMBER, anchor='mm')
    d.line([(340, 560), (740, 560)], fill=(60, 60, 66), width=2)
    # big CTA two-tone
    t1 = 'COMMENT '
    w1 = d.textlength(t1, font=f_big)
    w2 = d.textlength('"SKILL"', font=f_big)
    x = 540 - (w1 + w2) / 2
    d.text((x, 680), t1, font=f_big, fill=WHITE, anchor='lm')
    d.text((x + w1, 680), '"SKILL"', font=f_big, fill=AMBER, anchor='lm')
    d.text((540, 790), "and I'll DM you the skill file.", font=f_mid, fill=(200, 200, 208), anchor='mm')
    # logo
    try:
        lg = Image.open(LOGO).convert('RGBA')
        bbox = lg.getbbox()
        lg = lg.crop(bbox)
        lw = 430
        lh = int(lg.height * lw / lg.width)
        lg = lg.resize((lw, lh), Image.LANCZOS)
        im.paste(lg, (540 - lw // 2, 1120), lg)
    except Exception as e:
        print('LOGO FAIL', e)
    d.text((540, 1310), 'FOLLOW @cloofsiu', font=f_follow, fill=WHITE, anchor='mm')
    d.text((540, 1375), 'cv.cloofsiu.com', font=ImageFont.truetype(PF, 30, index=4), fill=(140, 140, 146), anchor='mm')
    os.makedirs(f'{OUT}/end', exist_ok=True)
    for f in range(n):
        im.save(f'{OUT}/end/f{f:04d}.png')
    print('endcard frames:', n)


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        build_terminal(float(sys.argv[1]))
        build_endcard(float(sys.argv[2]))
    else:
        build_terminal()
        build_endcard()
