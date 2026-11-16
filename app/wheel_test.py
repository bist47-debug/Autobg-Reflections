r"""
wheel_test.py - test the new tyre reflections on a folder of cars, no Qwen / v2 needed (runs in ~1 s per car).

    python wheel_test.py "batch 2\cars"

Writes debug\wheels_test.jpg: per car, left = result (backdrop + tyre reflection + AutoBG shadow + car),
right = what was found (green = near wheel, yellow = far wheel, red line = contact point).
Options: --strength 0.6 --length 0.55 --far 1.0 --backdrop path
"""
import argparse, glob, os, time
import numpy as np, cv2
from PIL import Image
import wheels as WH

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--backdrop", default=os.path.join(HERE, "floor_satin_light.png"))
ap.add_argument("--strength", type=float, default=0.6)
ap.add_argument("--length", type=float, default=0.55)
ap.add_argument("--fade", type=float, default=1.0)
ap.add_argument("--far", type=float, default=1.0)
ap.add_argument("--hidden", type=float, default=1.0)
ap.add_argument("--detail", type=float, default=0.1)
ap.add_argument("--out", default=os.path.join(HERE, "debug", "wheels_test.jpg"))
A = ap.parse_args()


def place(bg, car, sh, bottom=0.95, width=0.55, height=0.55):
    BW, BH = bg.size
    box = car.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    bw, bh = box[2] - box[0], box[3] - box[1]
    s = min(width * BW / bw, height * BH / bh)
    size = (max(1, round(car.width * s)), max(1, round(car.height * s)))
    x0, y0 = round(BW / 2 - (box[0] + bw / 2) * s), round(bottom * BH - box[3] * s)

    def layer(img):
        L = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
        im = img.resize(size, Image.LANCZOS); L.paste(im, (x0, y0), im)
        return L
    return layer(car), layer(sh) if sh is not None else None


def over(base, rgba):
    a = rgba[..., 3:].astype(np.float32) / 255
    return base * (1 - a) + rgba[..., :3].astype(np.float32) * a


bg = Image.open(A.backdrop).convert("RGBA")
bgc = cv2.cvtColor(np.asarray(bg.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float32)
cars = [f for f in sorted(glob.glob(os.path.join(A.folder, "*.png"))) if "_shadow" not in os.path.basename(f).lower()]
print(len(cars), "cars; loading wheel models ...", flush=True)
print("tyre finder:", "YOUR TRAINED MODEL (models/tyres.pt)" if WH._yolo() is not None else "old detector (OWLv2 + SAM)", flush=True)
tiles = []
for f in cars:
    stem = os.path.splitext(f)[0]
    sf = next((stem + x for x in ("_shadow.png", "_shadows.png") if os.path.exists(stem + x)), None)
    car = Image.open(f).convert("RGBA")
    sh = Image.open(sf).convert("RGBA") if sf else None
    if sh is not None and sh.size != car.size:
        sh = None
    carL, shL = place(bg, car, sh)
    c = cv2.cvtColor(np.asarray(carL), cv2.COLOR_RGBA2BGRA)
    t = time.time()
    ws = WH.detect(c)
    res = bgc.copy()
    if shL is not None:
        res = over(res, cv2.cvtColor(np.asarray(shL), cv2.COLOR_RGBA2BGRA))
    res = WH.flip(res, c, ws, A.strength, A.length, A.fade, 0.0, A.far, A.hidden, A.detail)   # reflection over the shadow
    res = over(res, c)
    res = np.clip(res, 0, 255).astype(np.uint8)
    dbg = WH.debug(res, ws)
    ys, xs = np.where(c[..., 3] > 128)
    H_, W_ = res.shape[:2]
    pw = int(0.04 * W_)
    y0, y1 = max(0, ys.min() - pw), min(H_, ys.max() + int(0.25 * (ys.max() - ys.min())))
    x0, x1 = max(0, xs.min() - pw), min(W_, xs.max() + pw)
    pair = np.hstack([res[y0:y1, x0:x1], dbg[y0:y1, x0:x1]])
    pair = cv2.resize(pair, (1400, int(1400 * pair.shape[0] / pair.shape[1])), interpolation=cv2.INTER_AREA)
    n = os.path.basename(stem)
    cv2.putText(pair, f"{n}  wheels: {sum(w['near'] for w in ws)} near, {sum(not w['near'] for w in ws)} far",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    tiles.append(pair)
    print(f"{n}: {len(ws)} wheels ({time.time() - t:.1f} s)", flush=True)

# contact sheet, 2 per row
Hm = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.full((Hm - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
if len(tiles) % 2:
    tiles.append(np.full_like(tiles[0], 255))
sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
os.makedirs(os.path.dirname(A.out), exist_ok=True)
cv2.imwrite(A.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("->", A.out)
