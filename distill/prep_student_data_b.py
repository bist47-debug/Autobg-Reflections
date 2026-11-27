"""Plan B data: teacher pairs -> strips (bottom half, 1024x384) with the mirror hint in the input.
usage: python prep_student_data_b.py /workspace/distill/pairs /workspace/distill/backdrops /workspace/distill/student_data_b"""
import os, sys, json, glob, hashlib, re
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hint

PROMPT = "realistic floor reflection of the car on a glossy grey showroom floor"
src, bdir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
for d in ("train_A", "train_B", "test_A", "test_B", "check"):
    os.makedirs(os.path.join(dst, d), exist_ok=True)
car_id = lambda s: re.sub(r"_\d+$", "", s)
stems = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(src, "*_in.png")))
stems = [s for s in stems if os.path.exists(os.path.join(src, s + "_tgt.png"))]
cars = sorted({car_id(s) for s in stems})
test_cars = set(sorted(cars, key=lambda c: hashlib.md5(c.encode()).hexdigest())[:max(5, int(0.04 * len(cars)))])
bgs = {}
tr, te, bad = {}, {}, 0
for i, s in enumerate(stems, 1):
    meta = json.load(open(os.path.join(src, s + ".json")))
    a = np.asarray(Image.open(os.path.join(src, s + "_in.png")).convert("RGB"))
    b = np.asarray(Image.open(os.path.join(src, s + "_tgt.png")).convert("RGB"))
    key = (meta["backdrop"], a.shape)
    if key not in bgs:
        bp = os.path.join(bdir, meta["backdrop"])
        bgs[key] = np.asarray(Image.open(bp).convert("RGBA").resize((a.shape[1], a.shape[0]), Image.LANCZOS).convert("RGB"))
    m = hint.car_mask(a, bgs[key])
    if m.sum() < 0.02 * m.size:                          # car not found against the backdrop -> skip
        bad += 1; continue
    ia, y0 = hint.strip(hint.mirror_hint(a, m))
    ib, _ = hint.strip(b)
    n = s.replace(" ", "_") + ".png"
    sp = "test" if car_id(s) in test_cars else "train"
    Image.fromarray(ia).save(os.path.join(dst, sp + "_A", n)); Image.fromarray(ib).save(os.path.join(dst, sp + "_B", n))
    (te if sp == "test" else tr)[n] = PROMPT
    if sp == "train":
        mn = n[:-4] + "_mirror.png"
        Image.fromarray(ia[:, ::-1].copy()).save(os.path.join(dst, "train_A", mn)); Image.fromarray(ib[:, ::-1].copy()).save(os.path.join(dst, "train_B", mn))
        tr[mn] = PROMPT
    if i <= 12:                                          # a few to eyeball: input-with-hint over target
        Image.fromarray(np.vstack([ia, ib])).save(os.path.join(dst, "check", n.replace(".png", ".jpg")), quality=90)
    if i % 250 == 0:
        print(f"{i}/{len(stems)}", flush=True)
json.dump(tr, open(os.path.join(dst, "train_prompts.json"), "w"))
json.dump(te, open(os.path.join(dst, "test_prompts.json"), "w"))
print(f"pairs {len(stems)} -> train {len(tr)} (incl. mirrored), test {len(te)}, skipped {bad}")
