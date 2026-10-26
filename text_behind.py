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

def hug_of(mask_bool_small, W, H, g_full):
    """edge-hug score: mean distance from the mask boundary to the nearest real
    image edge (px, lower = tighter). Used to reject garbage masks."""
    _, m8 = refine_mask(mask_bool_small, W, H, g_full)
    img_edges = cv2.Canny((g_full * 255).astype(np.uint8), 60, 160)
    dtm = cv2.distanceTransform((img_edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    me = cv2.Canny(m8, 50, 150)
    return float(dtm[me > 0].mean()) if me.any() else 999.0

def _propagate_guarded(mask_small, prev_gray_u8, cur_gray_u8, frame, W, H, hug_scores, report, i, last_good):
    """flow-warp the mask, then ALWAYS verify the warped mask with the hug
    score — flow polluted by baked motion graphics (sweeping cartoons, whip
    pans) smears the mask even when the *mean* flow looks small (localized
    big motion diluted by a static background). On failure: RESTORE the last
    accepted fresh SAM2 anchor — the current mask may already carry
    incremental pollution (the ratchet effect), so freezing it is not enough.
    Returns (mask, flow_mag)."""
    m_new, mag = _propagate(mask_small, prev_gray_u8, cur_gray_u8)
    g_full = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hug = hug_of(m_new, W, H, g_full)
    med = float(np.median(hug_scores)) if hug_scores else 20.0
    guard = max(40.0, 3.0 * med)
    if hug > guard:
        if report:
            print(f'  freeze mask at frame #{i} (flow {mag:.2f}px, warped hug {hug:.0f}px > guard {guard:.0f}px) -> restore last good anchor', flush=True)
        return (last_good if last_good is not None else mask_small), mag
    return m_new, mag

def _propagate(mask_small, prev_gray_u8, cur_gray_u8):
    """optical-flow warp: move the mask WITH the subject between SAM2 refreshes.
    ⚠ Farneback on this machine returns an all-zero field for float32 input —
    it only works with UINT8 grays. Returns (new_mask, mean flow magnitude)."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray_u8, cur_gray_u8, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    h_, w_ = mask_small.shape
    gx, gy = np.meshgrid(np.arange(w_, dtype=np.float32),
                         np.arange(h_, dtype=np.float32))
    mx = gx + flow[..., 0]; my = gy + flow[..., 1]
    m_new = cv2.remap(mask_small.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE) > 0.5
    mag = float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
    return m_new, mag

def refine_mask(mask_bool_small, W, H, guide_gray=None):
    """upscale -> guided-filter edge snap -> per-column top-edge snap ->
    1D median smoothing -> morphology -> drop specks -> tight 1px feather.

    guide_gray: float32 grayscale (H,W), 0..1 — snaps the boundary to the
    frame's real image edges; the top-edge snap pulls the mask's top boundary
    down onto the strongest horizontal image edge in a small band, removing
    the "fabric strip" bias that generous SAM2 box prompts leave above the
    true silhouette top. Returns (feathered alpha image, raw uint8 mask)."""
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
    alpha = Image.fromarray(cv2.GaussianBlur(m8, (0, 0), 1.0))
    return alpha, m8

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


def run_video_engine(a, W, H, fps_n, n, sh, txt, work, out_path):
    """Official SAM2 VIDEO predictor: prompt frame 0 once, the model's memory
    module propagates the mask to EVERY frame with temporal consistency —
    the fix for jumping text on moving subjects. MPS (Apple Metal) ≈ 0.9s/frame."""
    import subprocess as sp, numpy as np, torch, os
    from sam2.build_sam import build_sam2_video_predictor

    dev = a.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt = os.environ.get('SAM2_CKPT') or os.path.expanduser(
        '~/.cache/huggingface/hub/models--facebook--sam2.1-hiera-small/snapshots/ee5bba1d82bb8749febdf90f45e84b687142ba03/sam2.1_hiera_small.pt')
    if not os.path.exists(ckpt):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id='facebook/sam2.1-hiera-small', filename='sam2.1_hiera_small.pt')
    # the config ships INSIDE the sam2 package
    cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'   # hydra resolves pkg://sam2
    pred = build_sam2_video_predictor(cfg, ckpt, device=dev)
    print(f'video engine: {dev}', flush=True)

    # the loader parses file STEMS as ints — needs pure-number names
    vin = out_path + '_vin'
    os.makedirs(vin, exist_ok=True)
    for k, fn in enumerate(sorted(f for f in os.listdir(work) if f.endswith('.jpg'))):
        dst = f'{vin}/{k:05d}.jpg'
        try: os.link(f'{work}/{fn}', dst)
        except OSError: import shutil as _sh; _sh.copy(f'{work}/{fn}', dst)

    with torch.inference_mode():
        state = pred.init_state(video_path=vin, offload_video_to_cpu=True, offload_state_to_cpu=True)
        # prompts on frame 0: manual boxes (obj per box) and/or YOLO persons
        boxes = a.boxes
        objs = 0
        for chunk in boxes.split(';'):
            chunk = chunk.strip()
            if not chunk: continue
            v = [float(x) for x in chunk.split(',')]
            pred.add_new_points_or_box(state, frame_idx=0, obj_id=objs, box=np.array(v))
            objs += 1
        if a.auto_person:
            from ultralytics import YOLO
            yolo = YOLO('yolov8n.pt')
            f0img = Image.open(f'{work}/' + sorted(os.listdir(work))[0]).convert('RGB')
            det = yolo.predict(np.asarray(f0img), classes=[0], verbose=False, conf=a.conf)[0]
            if det.boxes is not None:
                for b in det.boxes.xyxy.cpu().numpy():
                    pred.add_new_points_or_box(state, frame_idx=0, obj_id=objs, box=np.array(b.tolist()))
                    objs += 1
        if objs == 0:
            print('no prompts — nothing to segment'); return False

        masks_dir = out_path + '_masks'
        os.makedirs(masks_dir, exist_ok=True)
        count = 0
        for frame_idx, obj_ids, masks in pred.propagate_in_video(state):
            acc = None
            for m in masks:
                mm = (m > 0.0).squeeze().cpu().numpy()
                acc = mm if acc is None else (acc | mm)
            if acc is not None:
                np.save(f'{masks_dir}/m{frame_idx:04d}.npy', acc)
            count += 1
            if count % 24 == 0: print('propagated', count, flush=True)
    print(f'propagation done: {count} frames', flush=True)

    # composite
    files = sorted(f for f in os.listdir(work) if f.endswith('.jpg'))
    idx = 0
    for fn in files:
        fi = f'{masks_dir}/m{idx:04d}.npy'
        frame = Image.open(f'{work}/{fn}').convert('RGB')
        if os.path.exists(fi):
            mb = np.load(fi)
            mimg = Image.fromarray((mb*255).astype('uint8')).resize((W,H), Image.BILINEAR)
        else:
            mimg = Image.new('L',(W,H),0)
        comp = frame.convert('RGBA')
        comp.alpha_composite(sh, (6,9)); comp.alpha_composite(txt)
        comp.paste(frame,(0,0),mask=mimg)
        comp.convert('RGB').save(f'{work}/c{idx:04d}.jpg', quality=90)
        idx += 1
        if idx % 24 == 0: print(f'comp {idx}', flush=True)
    sp.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps_n),'-i',f'{work}/c%04d.jpg',
            '-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-movflags','+faststart', out_path], check=True)
    if not a.keep_frames:
        import shutil; shutil.rmtree(masks_dir, ignore_errors=True); shutil.rmtree(vin, ignore_errors=True)
    print('done ->', out_path)
    return True

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
    ap.add_argument('--amber', action='store_true', help='amber text instead of white (legacy alias of --color amber)')
    ap.add_argument('--color', default=None, choices=['auto','white','black','amber'],
                    help='text colour: auto = sample the background luminance (light bg -> black text). Default auto unless --amber given.')
    ap.add_argument('--report', action='store_true', help='print edge-hug score (mask boundary distance to real image edges, px — lower = tighter)')
    ap.add_argument('--conf', type=float, default=0.4, help='YOLO person confidence threshold')
    ap.add_argument('--engine', default='video', choices=['video','flow'],
                    help='video = official SAM2 video predictor (prompt once, model propagates every frame — stable on moving subjects, default); flow = legacy refresh+optical-flow hybrid')
    ap.add_argument('--device', default='auto', help='torch device for the video engine: auto / mps / cuda / cpu')
    ap.add_argument('--keep-frames', action='store_true')
    a = ap.parse_args()
    find_ffmpeg()

    if a.amber and a.color is None:
        a.color = 'amber'
    if a.color is None:
        a.color = 'auto'

    boxes = []
    for chunk in a.boxes.split(';'):
        chunk = chunk.strip()
        if chunk:
            boxes.append(tuple(float(v) for v in chunk.split(',')))

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

    fill = None
    if a.amber and a.color is None:
        a.color = 'amber'
    if a.color is None:
        a.color = 'auto'
    main_x = a.text_x if a.text_x is not None else W // 2
    main_y = a.text_y if a.text_y is not None else H // 2
    sub_x  = a.text_x if a.text_x is not None else W // 2
    sub_y  = a.sub_y  if a.sub_y  is not None else main_y + 130

    work = a.out + '_work'
    os.makedirs(work, exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(a.start),'-t',str(a.duration),
                    '-i',a.input,'-vf',f'fps={fps_n}','-pix_fmt','yuvj420p','-start_number','0','-q:v','2',
                    f'{work}/s%04d.jpg'], check=True)

    # ADAPTIVE TEXT COLOUR: sample the luminance under the visible (upper) half
    # of the text bbox on the first frame — light background -> dark text.
    # White text on a white studio wall is the #1 way this effect fails.
    tmp = Image.new('L', (W, H), 0)
    dt_ = ImageDraw.Draw(tmp)
    f_main = ImageFont.truetype(a.font, a.main_size, index=8)
    dt_.text((main_x, main_y), a.text, font=f_main, fill=255, anchor='mm')
    bb = tmp.getbbox()
    fill = (255, 255, 255)
    if bb:
        zone = Image.open(f'{work}/s0000.jpg').convert('L').crop((bb[0], bb[1], bb[2], (bb[1]+bb[3])//2))
        lum = float(np.asarray(zone).mean())
        if a.color == 'auto':
            fill = (18, 18, 18) if lum > 130 else (255, 255, 255)
        elif a.color == 'black':
            fill = (18, 18, 18)
        elif a.color == 'white':
            fill = (255, 255, 255)
        elif a.color == 'amber':
            fill = (245, 158, 11)
        print(f'text zone luminance {lum:.0f} -> fill {fill}')
    sh, txt = build_text_layer(W, H, a.text, a.sub, main_x, main_y, sub_x, sub_y,
                               a.main_size, a.sub_size, fill, a.font)

    if a.engine == 'video':
        try:
            ok = run_video_engine(a, W, H, fps_n, n, sh, txt, work, a.out)
            if ok:
                if not a.keep_frames:
                    import shutil; shutil.rmtree(work, ignore_errors=True)
                return
        except Exception as e:
            print(f'video engine failed ({e}) — falling back to flow engine', flush=True)

    prev_gray_u8 = None
    m_small = None
    hug_scores = []
    fresh = False
    last_good = None
    sw, shh = W // 2, H // 2
    for i in range(n):
        fi = f'{work}/s{i:04d}.jpg'
        if not os.path.exists(fi): break
        frame = Image.open(fi).convert('RGB')
        small = frame.resize((sw, shh), Image.BILINEAR)
        arr = np.asarray(small)
        cur_gray_u8 = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)   # uint8 — Farneback needs it
        fresh = False

        if i % a.every == 0 or m_small is None:
            # --- refresh: SAM2 anchor (persons and/or boxes, UNIONED) ---
            fresh = True
            M_p = M_b = None
            if yolo is not None:
                det = yolo.predict(arr, classes=[0], verbose=False, conf=a.conf)[0]
                if det.boxes is not None and len(det.boxes):
                    all_boxes = [b.tolist() for b in det.boxes.xyxy.cpu().numpy()]
                    r = sam.predict(arr, bboxes=all_boxes, verbose=False)[0]
                    if r.masks is not None and r.masks.data is not None and r.masks.data.numel() > 0:
                        M_p = r.masks.data.cpu().numpy().max(axis=0) > 0.5
            if boxes:
                sx = sw / W; sy = shh / H
                small_boxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes]
                r = sam.predict(arr, bboxes=small_boxes, verbose=False)[0]
                if r.masks is not None and r.masks.data is not None and r.masks.data.numel() > 0:
                    M_b = r.masks.data.cpu().numpy().max(axis=0) > 0.5
            M = M_p if M_p is not None else M_b
            if M_p is not None and M_b is not None:
                M = M_p | M_b   # combined mode: person AND boxes occlude
            if M is not None:
                # HUG-GUARD: score the fresh mask; reject garbage (e.g. baked
                # graphics + subject fused into one blob) and keep propagating.
                g_full = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                hug = hug_of(M, W, H, g_full)
                med = float(np.median(hug_scores)) if hug_scores else hug
                guard = max(40.0, 3.0 * med)
                if hug > guard:
                    print(f'  reject bad mask at frame #{i} (hug {hug:.0f}px > guard {guard:.0f}px), keeping propagated mask', flush=True)
                    M = None   # fall through to propagation below
                elif a.report:
                    print(f'  refresh #{i}: hug {hug:.1f}px <= guard {guard:.1f}px accepted', flush=True)
            if M is not None:
                m_small = (0.65 * M + 0.35 * m_small) > 0.4 if m_small is not None else M
                last_good = m_small.copy()
            elif m_small is not None:
                m_small, fresh_ok = _propagate_guarded(
                    m_small, prev_gray_u8, cur_gray_u8, frame, W, H, hug_scores, a.report, i, last_good)
        elif m_small is not None:
            # --- propagate: optical flow moves the mask WITH the subject so
            #     the cut edge stays continuous between SAM2 refreshes ---
            m_small, fresh_ok = _propagate_guarded(
                m_small, prev_gray_u8, cur_gray_u8, frame, W, H, hug_scores, a.report, i, last_good)
        prev_gray_u8 = cur_gray_u8
        guide = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mimg, m8 = refine_mask(m_small, W, H, guide)
        # FINAL PASTE GUARD: if the refined mask is wildly misaligned with the
        # current frame (subject covered by a baked title card / wrong blob),
        # the paste would repaint garbage over the text — skip it instead.
        # Threshold well above the freeze guard: legitimately soft-edge scenes
        # (dark subject on white, hug ~70px) must NOT trigger this.
        med_h = float(np.median(hug_scores)) if hug_scores else 20.0
        img_edges = cv2.Canny((guide * 255).astype(np.uint8), 60, 160)
        dtm = cv2.distanceTransform((img_edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        me = cv2.Canny(m8, 50, 150)
        if me.any():
            hug_final = float(dtm[me > 0].mean())
            if hug_final > max(120.0, 4.0 * med_h):
                mimg = Image.new('L', (W, H), 0)   # no paste: text stays fully visible
        if a.report and fresh:
            img_edges = cv2.Canny((guide * 255).astype(np.uint8), 60, 160)
            dt = cv2.distanceTransform((img_edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
            me = cv2.Canny(m8, 50, 150)
            if me.any():
                hug_scores.append(float(dt[me > 0].mean()))

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
    if a.report and hug_scores:
        import statistics as st
        mean_h = st.mean(hug_scores); worst_h = max(hug_scores)
        flags = [f'#{k*a.every}' for k,s in enumerate(hug_scores) if s > 100]
        warn = f'  ⚠ anomalous masks at frames {flags} — spot-check' if flags else ''
        print(f'edge-hug: mean {mean_h:.2f}px  worst {worst_h:.2f}px  over {len(hug_scores)} refreshes (lower = tighter){warn}')
    print('done ->', a.out)

if __name__ == '__main__':
    main()
