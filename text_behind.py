#!/usr/bin/env python3
"""
text_behind.py — put BIG modern text BEHIND the subject in a video
(text sandwiched between background and subject, the "text behind person" reel effect).

Pipeline: SAM2 subject mask (refreshed every N frames, morphologically cleaned
and feathered) -> PIL text layer (no border, soft blurred shadow) ->
composite bg -> text -> subject.

KEY INSIGHT for static-camera shots: you do NOT need a mask every frame.
Refresh the mask every N frames (0.25-0.5s) and reuse it in between — the
subject barely moves relative to the frame, so this is ~10x faster and
visually identical. Handheld shots need a smaller --every and the built-in
EMA smoothing handles residual sway.

EDGE QUALITY: the (low-res) mask is upscaled, then morphologically cleaned
(close pinholes / open specks / drop small islands) and feathered with a
1.8px Gaussian — the soft alpha pastes the subject back over the text with
a clean edge instead of an aliased staircase.

Requirements:
  pip install ultralytics pillow numpy opencv-python-headless
  ffmpeg on PATH

Usage (auto-detect people with YOLO):
  python text_behind.py -i clip.mp4 -t "CLOOF AI" -s "DIY CYBERDECK" \
      -o out.mp4 --start 2 --duration 4 --auto-person

Usage (object / device shots — give SAM2 box prompts in source pixels):
  python text_behind.py -i clip.mp4 -t "跑緊真 Linux" -s "RASPBERRY PI ZERO 2 W" \
      -o out.mp4 --start 14 --duration 4.5 \
      --boxes "2,2,1800,1000" --every 8

Static camera? use --every 8 (mask every 1/3 s). Handheld? --every 4.
"""
import argparse, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import cv2  # opencv-python-headless: mask cleanup + feathering

def refine_mask(mask_bool_small, W, H, guide_gray=None):
    """upscale -> guided-filter edge snap -> morphology -> per-column TOP-EDGE
    SNAP -> 1D median boundary smoothing -> drop specks -> tight 1px feather.

    guide_gray: float32 grayscale (H,W), 0..1 — snaps the mask boundary to the
    frame's real image edges. The top-edge snap additionally pulls the mask's
    top boundary down onto the strongest horizontal image edge within a small
    band, which removes the systematic "fabric strip" bias that generous SAM2
    box prompts leave above the true silhouette top."""
    mfull = cv2.resize(mask_bool_small.astype(np.float32), (W, H),
                       interpolation=cv2.INTER_LINEAR)
    if guide_gray is not None:
        mfull = cv2.ximgproc.guidedFilter(guide_gray, mfull, radius=16, eps=1e-3)
    m8 = (mfull > 0.5).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)

    Gb = None
    if guide_gray is not None:
        gray_u8 = (guide_gray * 255).astype(np.uint8)
        Gb = np.abs(cv2.GaussianBlur(cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3), (0, 0), 2))
    if Gb is not None:
        mb_ = m8 > 127
        Hh, Ww = m8.shape
        ys = np.zeros(Ww, int); has = mb_.any(axis=0)
        ys[has] = np.argmax(mb_, axis=0)[has]
        for x in np.where(has)[0]:
            yb = ys[x]
            lo, hi = max(0, yb-15), min(Hh, yb+36)
            g = Gb[lo:hi, x]
            ystar = lo + int(g.argmax())
            if Gb[ystar, x] > 90 and ystar > yb + 1:
                ys[x] = ystar
        ysm = ys.copy()
        for x in np.where(has)[0]:
            lo2, hi2 = max(0, x-7), min(Ww, x+8)
            seg = ys[lo2:hi2][has[lo2:hi2]]
            if len(seg): ysm[x] = int(np.median(seg))
        for x in np.where(has)[0]:
            top = int(np.argmax(m8[:, x] > 127))
            if ysm[x] > top:
                m8[top:ysm[x], x] = 0

    num, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 3000:
            m8[lab == i] = 0
    return Image.fromarray(cv2.GaussianBlur(m8, (0, 0), 1.0))

def find_ffmpeg():
    try:
        subprocess.run(['ffmpeg','-version'], capture_output=True, check=True)
    except Exception:
        sys.exit('ffmpeg not found on PATH')

def build_text_layer(W, H, main, sub, main_x, main_y, sub_x, sub_y, main_size, sub_size,
                     fill, font_path, main_index=8, sub_index=4):
    A = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(A)
    f_main = ImageFont.truetype(font_path, main_size, index=main_index)
    f_sub  = ImageFont.truetype(font_path, sub_size,  index=sub_index)
    d.text((main_x, main_y), main, font=f_main, fill=255, anchor='mm')
    if sub:
        d.text((sub_x + 2, sub_y), sub, font=f_sub, fill=255, anchor='mm')
    shadow = A.filter(ImageFilter.GaussianBlur(12))
    txt = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    txt.paste(Image.new('RGBA', (W, H), fill + (255,)), mask=A)
    sh = Image.new('RGBA', (W, H), (8, 8, 8, 255))
    sh.putalpha(shadow.point(lambda a: int(a * 0.55)))
    return sh, txt

def main():
    ap = argparse.ArgumentParser(description='Text-behind-subject video effect (SAM2 + PIL + ffmpeg)')
    ap.add_argument('-i','--input', required=True)
    ap.add_argument('-t','--text', required=True, help='main line (CJK ok)')
    ap.add_argument('-s','--sub', default='', help='small subline')
    ap.add_argument('-o','--out', required=True)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--every', type=int, default=8, help='recompute mask every N frames (static camera: 8-12, handheld: 4)')
    ap.add_argument('--auto-person', action='store_true', help='YOLO auto person detection instead of box prompts')
    ap.add_argument('--boxes', default='', help='SAM2 box prompts "x1,y1,x2,y2;x1,y1,x2,y2" in source pixels')
    ap.add_argument('--text-y', type=int, default=None, help='main text centre y (default: 50%% of height)')
    ap.add_argument('--text-x', type=int, default=None, help='main text centre x (default: 50%% of width)')
    ap.add_argument('--sub-y', type=int, default=None, help='subline centre y (default: main + 130)')
    ap.add_argument('--main-size', type=int, default=130)
    ap.add_argument('--sub-size', type=int, default=44)
    ap.add_argument('--font', default='/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc',
                    help='TTC/TTF font (default: macOS PingFang HK)')
    ap.add_argument('--amber', action='store_true', help='amber text instead of white')
    ap.add_argument('--keep-frames', action='store_true')
    a = ap.parse_args()
    find_ffmpeg()

    # probe source size via 1-frame extract (stream metadata can lie about rotation)
    os.makedirs('.tbs_probe', exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss','0','-i',a.input,
                    '-frames:v','1','.tbs_probe/size.png'], check=True)
    with Image.open('.tbs_probe/size.png') as im:
        W, H = im.size
    import shutil; shutil.rmtree('.tbs_probe', ignore_errors=True)
    fps_n = a.fps
    n = int(a.duration * fps_n)

    from ultralytics import SAM, YOLO
    sam = SAM('sam2.1_b.pt')
    yolo = YOLO('yolov8n.pt') if a.auto_person else None

    boxes = []
    for chunk in a.boxes.split(';'):
        chunk = chunk.strip()
        if chunk:
            boxes.append(tuple(float(v) for v in chunk.split(',')))

    fill = (245, 158, 11) if a.amber else (255, 255, 255)
    main_x = a.text_x if a.text_x is not None else W // 2
    main_y = a.text_y if a.text_y is not None else H // 2
    sub_x  = a.text_x if a.text_x is not None else W // 2
    sub_y  = a.sub_y  if a.sub_y  is not None else main_y + 130
    sh, txt = build_text_layer(W, H, a.text, a.sub, main_x, main_y, sub_x, sub_y,
                               a.main_size, a.sub_size, fill, a.font)

    work = a.out + '_work'
    os.makedirs(work, exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(a.start),'-t',str(a.duration),
                    '-i',a.input,'-vf',f'fps={fps_n}','-pix_fmt','yuvj420p','-start_number','0','-q:v','2',
                    f'{work}/s%04d.jpg'], check=True)

    prev = None
    m_small = None
    for i in range(n):
        fi = f'{work}/s{i:04d}.jpg'
        if not os.path.exists(fi): break
        frame = Image.open(fi).convert('RGB')
        if i % a.every == 0 or prev is None:
            small = frame.resize((W // 2, H // 2), Image.BILINEAR)
            arr = np.asarray(small)
            M = None
            if yolo is not None:
                det = yolo.predict(arr, classes=[0], verbose=False, conf=0.4)[0]
                if det.boxes is not None and len(det.boxes):
                    for b in det.boxes.xyxy.cpu().numpy():
                        sx, sy = W / (W // 2), H / (H // 2)
                        r = sam.predict(arr, bboxes=[b.tolist()], verbose=False)[0]
                        m = r.masks.data.cpu().numpy().max(axis=0) > 0.5
                        M = m if M is None else (M | m)
            else:
                for b in boxes:
                    r = sam.predict(arr, bboxes=[list(b)], verbose=False)[0]
                    if r.masks is None or r.masks.data is None or r.masks.data.numel() == 0:
                        continue
                    m = r.masks.data.cpu().numpy().max(axis=0) > 0.5
                    M = m if M is None else (M | m)
            if M is None:
                M = prev if prev is not None else np.zeros((H // 2, W // 2), bool)
            elif prev is not None:
                M = (0.6 * M + 0.4 * prev) > 0.4   # EMA vs flicker
            prev = m_small = M
        guide = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mimg = refine_mask(m_small, W, H, guide)

        comp = frame.convert('RGBA')
        comp.alpha_composite(sh, (6, 9))
        comp.alpha_composite(txt)
        comp.paste(frame, (0, 0), mask=mimg)   # subject occludes the text
        comp.convert('RGB').save(f'{work}/c{i:04d}.jpg', quality=90)
        if i % 24 == 0:
            print(f'frame {i}/{n}', flush=True)

    cmd = ['ffmpeg','-y','-loglevel','error','-framerate',str(fps_n),'-i',f'{work}/c%04d.jpg']
    cmd += ['-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-movflags','+faststart', a.out]
    subprocess.run(cmd, check=True)
    if not a.keep_frames:
        import shutil; shutil.rmtree(work, ignore_errors=True)
    print('done ->', a.out)

if __name__ == '__main__':
    main()
