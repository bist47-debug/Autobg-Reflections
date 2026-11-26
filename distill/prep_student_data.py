"""Distillation step 2a: teacher pairs (/workspace/distill/pairs: <car>_in.png / <car>_tgt.png / <car>.json)
-> pix2pix-turbo dataset folder (train_A/B, test_A/B, *_prompts.json).
- resized to 768x576 (the student works at this size; the reflection is soft, finish() upsamples it)
- ~200 cars held out for testing, split by CAR (stock number), so photos of the same car never sit on both sides
- every training pair also added mirrored left-right (x2 data)
usage: python prep_student_data.py /workspace/distill/pairs /workspace/distill/student_data
"""
import os, sys, json, glob, hashlib, re
from PIL import Image, ImageOps

PROMPT = "realistic floor reflection of the car on a glossy grey showroom floor"
W, H = 768, 576
src, dst = sys.argv[1], sys.argv[2]
for d in ("train_A", "train_B", "test_A", "test_B"):
    os.makedirs(os.path.join(dst, d), exist_ok=True)

def car_id(stem):                      # "004408_03" -> "004408", "v60 d2 943_02" -> "v60 d2 943"
    return re.sub(r"_\d+$", "", stem)

stems = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(src, "*_in.png")))
stems = [s for s in stems if os.path.exists(os.path.join(src, s + "_tgt.png"))]
cars = sorted({car_id(s) for s in stems})
test_cars = {c for c in cars if int(hashlib.md5(c.encode()).hexdigest(), 16) % 1000 < int(1000 * 200 / max(1, len(cars)))}
tr, te = {}, {}
for i, s in enumerate(stems, 1):
    a = Image.open(os.path.join(src, s + "_in.png")).convert("RGB").resize((W, H), Image.LANCZOS)
    b = Image.open(os.path.join(src, s + "_tgt.png")).convert("RGB").resize((W, H), Image.LANCZOS)
    n = s.replace(" ", "_") + ".png"
    if car_id(s) in test_cars:
        a.save(os.path.join(dst, "test_A", n)); b.save(os.path.join(dst, "test_B", n)); te[n] = PROMPT
    else:
        a.save(os.path.join(dst, "train_A", n)); b.save(os.path.join(dst, "train_B", n)); tr[n] = PROMPT
        m = n[:-4] + "_mirror.png"
        ImageOps.mirror(a).save(os.path.join(dst, "train_A", m)); ImageOps.mirror(b).save(os.path.join(dst, "train_B", m)); tr[m] = PROMPT
    if i % 250 == 0:
        print(f"{i}/{len(stems)}", flush=True)
json.dump(tr, open(os.path.join(dst, "train_prompts.json"), "w"))
json.dump(te, open(os.path.join(dst, "test_prompts.json"), "w"))
print(f"pairs {len(stems)} from {len(cars)} cars -> train {len(tr)} (incl. mirrored), test {len(te)} ({len(test_cars)} cars held out)")
