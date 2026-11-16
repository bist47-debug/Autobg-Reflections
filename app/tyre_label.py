r"""
tyre_label.py - label the tyres of a folder of car cut-outs (for training your own tyre model).

    cd /d D:\reflections
    python tyre_label.py

Opens in your browser. For each car the tyres are already marked (coloured). Then:
  * click a tyre that is NOT marked      -> it gets added
  * click a mark that is wrong           -> it gets removed (spare wheel, bumper, tow hitch ...)
  * Next  -> saves and opens the next car     (Skip = bad image, not used for training)
Mark every tyre you can SEE, also far / half-hidden ones. Not the spare wheel.
It remembers where you stopped. Labels go to tyre_data\ds (ready for training).
"""
import argparse, glob, json, os
import numpy as np, cv2, torch
from PIL import Image
import gradio as gr
import wheels as WH

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--images", default=os.path.join(HERE, "tyre_data", "images"))
ap.add_argument("--out", default=os.path.join(HERE, "tyre_data", "ds"))
ap.add_argument("--port", type=int, default=7861)
A = ap.parse_args()
IMG_DIR, LAB_DIR = os.path.join(A.out, "images"), os.path.join(A.out, "labels")
os.makedirs(IMG_DIR, exist_ok=True); os.makedirs(LAB_DIR, exist_ok=True)
STATUS = os.path.join(A.out, "status.json")
FILES = sorted(glob.glob(os.path.join(A.images, "*.png")),
               key=lambda f: (len(os.path.basename(f)), os.path.basename(f)))
FILES = [f for f in FILES if "_shadow" not in os.path.basename(f).lower()]
DISP = 1400                                  # display width
COLS = [(0, 255, 0), (255, 160, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
S = {}


def load_status():
    try:
        return json.load(open(STATUS))
    except Exception:
        return {}


def save_status(st):
    json.dump(st, open(STATUS, "w"), indent=0)


def composite(f):
    im = np.asarray(Image.open(f).convert("RGBA")).astype(np.float32)
    a = im[..., 3:] / 255
    rgb = im[..., :3] * a + 128 * (1 - a)                # same mid grey the tyre finder uses
    return np.clip(rgb, 0, 255).astype(np.uint8), (im[..., 3] > 128)


@torch.no_grad()
def sam_embed(rgb):
    M = WH._models()
    inp = M["sp"](Image.fromarray(rgb), return_tensors="pt").to(WH.DEV)
    S["emb"] = M["sm"].get_image_embeddings(inp["pixel_values"])
    S["sizes"] = (inp["original_sizes"], inp["reshaped_input_sizes"])


@torch.no_grad()
def sam_point(x, y):
    """whole wheel (tyre + rim) under the click: the largest of SAM's 3 answers that is still wheel-sized"""
    M = WH._models()
    inp = M["sp"](Image.fromarray(S["rgb"]), input_points=[[[float(x), float(y)]]], return_tensors="pt")
    o = M["sm"](image_embeddings=S["emb"], input_points=inp["input_points"].to(WH.DEV), multimask_output=True)
    ms = M["sp"].image_processor.post_process_masks(o.pred_masks.cpu(), S["sizes"][0].cpu(), S["sizes"][1].cpu())[0][0].numpy()
    car = S["alpha"]
    ys, xs = np.where(car)
    lim = 0.12 * (xs.max() - xs.min()) * (ys.max() - ys.min())
    best = None
    for m in ms:
        m = m & car
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        if n < 2 or not m[int(y), int(x)]:
            continue
        m = lab == lab[int(y), int(x)]
        if m.sum() <= lim and (best is None or m.sum() > best.sum()):
            best = m
    if best is None:
        best = min((m & car for m in ms), key=lambda m: m.sum())
    return fill(best)


def fill(m):
    m = m.astype(np.uint8)
    ff = np.pad(1 - m, 1, constant_values=1).astype(np.uint8)
    cv2.floodFill(ff, None, (0, 0), 0)
    return (m | ff[1:-1, 1:-1]).astype(bool)


def view():
    rgb = S["rgb"].copy()
    over = rgb.astype(np.float32)
    for i, m in enumerate(S["masks"]):
        c = np.array(COLS[i % len(COLS)], np.float32)
        over[m] = over[m] * 0.5 + c * 0.5
    out = over.astype(np.uint8)
    for i, m in enumerate(S["masks"]):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, COLS[i % len(COLS)], 3)
    # crop to the car (+ margin) so the tyres are big on screen
    ys, xs = np.where(S["alpha"])
    H_, W_ = out.shape[:2]
    p = int(0.03 * W_)
    x0, x1, y0, y1 = max(0, xs.min() - p), min(W_, xs.max() + p), max(0, ys.min() - p), min(H_, ys.max() + p)
    crop = out[y0:y1, x0:x1]
    sc = DISP / crop.shape[1]
    S["view"] = (x0, y0, sc)
    return cv2.resize(crop, (DISP, int(crop.shape[0] * sc)), interpolation=cv2.INTER_AREA)


def info():
    st = load_status()
    done = sum(1 for v in st.values() if v in ("ok", "skip"))
    return f"{os.path.basename(S['file'])}   |   {len(S['masks'])} tyres marked   |   {done} / {len(FILES)} done"


def open_index(i):
    i = max(0, min(len(FILES) - 1, i))
    S.update(i=i, file=FILES[i])
    rgb, alpha = composite(FILES[i])
    S.update(rgb=rgb, alpha=alpha)
    stem = os.path.splitext(os.path.basename(FILES[i]))[0]
    lab = os.path.join(LAB_DIR, stem + ".txt")
    masks = []
    if os.path.exists(lab):                                  # already labelled -> show saved labels
        H_, W_ = alpha.shape
        for line in open(lab):
            v = line.split()
            if len(v) > 6:
                pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [W_, H_]).astype(np.int32)
                m = np.zeros((H_, W_), np.uint8); cv2.fillPoly(m, [pts], 1); masks.append(m.astype(bool))
    else:
        car = np.dstack([rgb[..., ::-1], (alpha * 255).astype(np.uint8)])
        for w in WH.detect(car, raw=True):
            X0, Y0, X1, Y1 = w["box"]
            m = np.zeros(alpha.shape, bool); m[Y0:Y1, X0:X1] = w["m"]
            masks.append(fill(m))
    S["masks"] = masks
    sam_embed(rgb)
    return view(), info()


def first_open():
    st = load_status()
    todo = [i for i, f in enumerate(FILES) if st.get(os.path.basename(f)) not in ("ok", "skip")]
    return open_index(todo[0] if todo else 0)


def click(evt: gr.SelectData):
    x0, y0, sc = S["view"]
    x, y = x0 + evt.index[0] / sc, y0 + evt.index[1] / sc
    xi, yi = int(x), int(y)
    hit = [k for k, m in enumerate(S["masks"]) if m[yi, xi]]
    if hit:
        S["masks"].pop(hit[0])                                # click on a mark -> remove
    elif S["alpha"][yi, xi]:
        S["masks"].append(sam_point(x, y))                    # click on an unmarked tyre -> add
    return view(), info()


def save_current(state):
    st = load_status()
    name = os.path.basename(S["file"]); stem = os.path.splitext(name)[0]
    if state == "ok":
        H_, W_ = S["alpha"].shape
        lines = []
        for m in S["masks"]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
            if len(c) < 3:
                continue
            lines.append("0 " + " ".join(f"{px / W_:.5f} {py / H_:.5f}" for px, py in c))
        open(os.path.join(LAB_DIR, stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(os.path.join(IMG_DIR, stem + ".jpg"), S["rgb"][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        for p in (os.path.join(LAB_DIR, stem + ".txt"), os.path.join(IMG_DIR, stem + ".jpg")):
            if os.path.exists(p):
                os.remove(p)
    st[name] = state
    save_status(st)


def nxt():
    save_current("ok")
    return open_index(S["i"] + 1)


def skip():
    save_current("skip")
    return open_index(S["i"] + 1)


def back():
    return open_index(S["i"] - 1)


def redetect():
    lab = os.path.join(LAB_DIR, os.path.splitext(os.path.basename(S["file"]))[0] + ".txt")
    S["masks"] = []
    car = np.dstack([S["rgb"][..., ::-1], (S["alpha"] * 255).astype(np.uint8)])
    for w in WH.detect(car, raw=True):
        X0, Y0, X1, Y1 = w["box"]
        m = np.zeros(S["alpha"].shape, bool); m[Y0:Y1, X0:X1] = w["m"]
        S["masks"].append(fill(m))
    return view(), info()


def clear():
    S["masks"] = []
    return view(), info()


print(f"{len(FILES)} cars in {A.images}; loading tyre finder ...", flush=True)
WH._models()
with gr.Blocks(title="Tyre labelling") as demo:
    gr.Markdown("### Tyre labelling - click an unmarked tyre to add it, click a wrong mark to remove it, then **Next**.  "
                "Mark every tyre you can see (also far / half-hidden ones), never the spare wheel.")
    txt = gr.Textbox(label="", interactive=False)
    with gr.Row():
        b_back = gr.Button("< Back")
        b_clear = gr.Button("Clear all marks")
        b_re = gr.Button("Re-detect")
        b_skip = gr.Button("Skip (bad image)")
        b_next = gr.Button("Next >  (save)", variant="primary")
    im = gr.Image(label="", type="numpy", interactive=False)
    im.select(click, None, [im, txt])
    b_next.click(nxt, None, [im, txt])
    b_skip.click(skip, None, [im, txt])
    b_back.click(back, None, [im, txt])
    b_re.click(redetect, None, [im, txt])
    b_clear.click(clear, None, [im, txt])
    demo.load(first_open, None, [im, txt])

demo.launch(server_name="127.0.0.1", server_port=A.port, inbrowser=True)
