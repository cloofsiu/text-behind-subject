#!/usr/bin/env python3
"""Assemble EN promo reel: beats+cards+captions+logo+VO -> 1080x1920 SDR."""
import json, subprocess, os
from PIL import Image

P = os.path.expanduser('~/Documents/Projects/text-behind-subject/promo')
HD = os.path.expanduser('~/Documents/Projects/text-behind-subject/promo_v2')
FPS = 24
LEAD, GAP, TAIL = 0.25, 0.3, 0.8
OUT = os.path.expanduser('~/Downloads/text-behind-tests/Reel_TEXTBEHIND_EN_v2.mp4')
PF = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'

def dur(path, video=False):
    e = 'format=duration' if not video else 'format=duration'
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', e, '-of', 'csv=p=0', path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())

# --- audio timeline from measured mp3 durations ---
d = [dur(f'{P}/l15_{i}.mp3') for i in range(1, 7)]
L = []  # (start, end) per line
t = LEAD
for i in range(6):
    L.append((t, t + d[i]))
    t = L[-1][1] + GAP
TOTAL = L[-1][1] + TAIL

# --- segments tile the timeline ---
segs = [
    ('deck', dur(f'{HD}/p_deck.mp4')),
    ('skull', dur(f'{HD}/p_skull.mp4')),
    ('sota', dur(f'{HD}/p_sota.mp4')),
    ('hl3', dur(f'{HD}/p_hl3.mp4')),
    ('term', len(os.listdir(f'{P}/term')) / FPS),
    ('end', len(os.listdir(f'{P}/end')) / FPS),
]
starts = [0.0, L[1][0], L[2][0], L[3][0], L[4][0], L[5][0]]
segs_win = []
for i, (name, sd) in enumerate(segs):
    a = starts[i]
    b = starts[i + 1] if i < 5 else TOTAL
    segs_win.append((name, sd, a, b, b - a))

# --- caption windows (char-weight splits, 13 plates) ---
L1, L2, L3, L4, L5 = L[0], L[1], L[2], L[3], L[4]
def third(a, b, *ws):
    """split [a,b] by char weights -> list of (start,end)"""
    tot = sum(ws); out = []; x = a
    for w in ws:
        out.append((x, x + (b - a) * w / tot)); x = out[-1][1]
    return out
raw = (
    third(L1[0], L1[1], 14, 18, 10) +      # p01 p02 p03
    [(L2[0], L2[1])] +                      # p04
    third(L3[0], L3[1], 9, 18, 20) +        # p05 p06 p07
    third(L4[0], L4[1], 8, 18, 17) +        # p08 p09 p10
    third(L5[0], L5[1], 14, 20, 23)         # p11 p12 p13
)
caps = [(f'p{i+1:02d}', round(a, 3), round(b, 3)) for i, (a, b) in enumerate(raw)]

timeline = {
    'lines': [tuple(round(x, 3) for x in li) for li in L],
    'segs': [(n, round(sd, 3), round(a, 3), round(b, 3), round(dd, 3)) for n, sd, a, b, dd in segs_win],
    'caps': caps, 'total': round(TOTAL, 3),
}
json.dump(timeline, open(f'{P}/timing.json', 'w'), indent=1)
print(json.dumps(timeline, indent=1))

# --- inputs ---
inp = [
    '-i', f'{HD}/p_deck.mp4', '-i', f'{HD}/p_skull.mp4', '-i', f'{HD}/p_sota.mp4', '-i', f'{HD}/p_hl3.mp4',
    '-framerate', '24', '-i', f'{P}/term/f%04d.png',
    '-framerate', '24', '-i', f'{P}/end/f%04d.png',
    '-f', 'lavfi', '-t', str(LEAD), '-i', 'anullsrc=r=44100:cl=stereo',
]
for i in range(6):
    if i:
        inp += ['-f', 'lavfi', '-t', str(GAP), '-i', 'anullsrc=r=44100:cl=stereo']
    inp += ['-i', f'{P}/l15_{i+1}.mp3']  # MUST match timing source (l15 = +15% files)
for n, a, b in caps:
    inp += ['-loop', '1', '-framerate', '24', '-t', f'{b - a:.3f}', '-i', f'{P}/caps/{n}.png']
inp += ['-loop', '1', '-framerate', '24', '-t', f'{TOTAL:.3f}', '-i', f'{P}/logo_240.png']

# --- filtergraph ---
fx = []
PAD = 2.0
for i, (name, sd, a, b, dd) in enumerate(segs_win):
    s = f'stop_duration={dd - sd + PAD:.3f}' if dd > sd else 'stop_duration=0.1'
    fill = (f'tpad=stop_mode=clone:{s},trim=duration={dd:.3f},setpts=PTS-STARTPTS,'
            f'format=yuv420p,setsar=1')
    if name in ('skull', 'sota', 'hl3'):
        fx.append(f'[{i}:v]split[bg{i}][fg{i}]')
        fx.append(f'[bg{i}]scale=1080:1920:force_original_aspect_ratio=increase,'
                  f'crop=1080:1920,gblur=sigma=30,eq=brightness=-0.12[bgp{i}]')
        fx.append(f'[fg{i}]scale=1080:608[fgp{i}]')
        fx.append(f'[bgp{i}][fgp{i}]overlay=0:560:format=auto,{fill}[v{i}]')
    else:
        fx.append(f'[{i}:v]{fill}[v{i}]')
fx.append('[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[vc]')

# audio: lead + l1..l6 with gaps
aidx = 6  # input 6 = lead silence, then l1 at 7, gap at 8, l2 at 9...
albl = []
fx.append(f'[6:a]aformat=sample_fmts=fltp:channel_layouts=stereo[s0]')
albl.append('[s0]')
for i in range(6):
    g = 8 + 2 * i
    a = 7 + 2 * i
    fx.append(f'[{a}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i+1}]')
    albl.append(f'[a{i+1}]')
    if i < 5:
        fx.append(f'[{g}:a]aformat=sample_fmts=fltp:channel_layouts=stereo[s{i+1}]')
        albl.append(f'[s{i+1}]')
fx.append(''.join(albl) + 'concat=n=12:v=0:a=1[araw]')
fx.append(f'[araw]loudnorm=I=-14:TP=-1.5:LRA=11,apad,atrim=duration={TOTAL:.3f},'
          f'asetpts=PTS-STARTPTS[af]')

# captions + logo overlays  (captions = inputs 18..27, logo = 28)
CAP_BASE = 18
prev = 'vc'
for k, (n, a, b) in enumerate(caps):
    ci = CAP_BASE + k
    nxt = f'o{k}'
    fx.append(f'[{prev}][{ci}:v]overlay=0:0:format=auto:'
              f'enable=\'between(t,{a:.3f},{b:.3f})\'[{nxt}]')
    prev = nxt
logo_in = CAP_BASE + len(caps)
fx.append(f'[{prev}][{logo_in}:v]overlay=x=800:y=40:format=auto:'
          f'enable=\'lt(t,{starts[5]:.3f})\'[vlg]')
fx.append('[vlg]setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv,'
          'format=yuv420p[vout]')

open(f'{P}/fx.txt', 'w').write(';\n'.join(fx) + ';\n')

cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-/filter_complex', f'{P}/fx.txt']
cmd += inp + ['-map', '[vout]', '-map', '[af]', '-t', f'{TOTAL:.3f}',
              '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
              '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', OUT]
print('RUN ffmpeg...')
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stderr[-3000:] if r.returncode else 'ENCODE OK')
if r.returncode == 0:
    print(json.dumps(json.loads(json.dumps({'out': OUT, 'duration': dur(OUT)}))))
