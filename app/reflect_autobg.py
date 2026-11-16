r"""
reflect_autobg.py - AutoBG floor reflections: v2 reflection LoRA (Qwen-Image-Edit-2511) + own YOLO tyre model
+ AutoBG shadows, with automatic view handling (side / front-back / 3-4). Locked settings: see BATCH below
and docs/REPLICATION.md.

  Team page (upload only, locked settings, password; login user "autobg"):
    python reflect_autobg.py --full --share --team --password XXXX --lora /workspace/v2.safetensors --backdrop floor_satin_light.png
  Full page with all sliders:
    python reflect_autobg.py --full --share --lora /workspace/v2.safetensors --backdrop floor_satin_light.png
  Batch (folder of name.png + name_shadow.png):
    python reflect_autobg.py --full --lora /workspace/v2.safetensors --backdrop floor_satin_light.png --batch cars
  On a 16 GB PC: leave out --full (compressed GGUF model; --quant Q3_K_M if memory is short).

Needs: pip install -r requirements.txt
"""
import argparse, os, re, sys, time
import numpy as np, cv2, torch
from PIL import Image
import gradio as gr
import wheels as WH       # your trained tyre model (models/tyres.pt) + tyre reflections

G = C = sys.modules[__name__]        # (the finishing code used to live in two helper files)

# ------------------------------------------------------------------------------- finishing helpers
W, H = 1024, 768

def rd(p, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flags)

def wr(p, im):
    cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(p)

def over(base, rgba):
    al = rgba[..., 3:4].astype(np.float32) / 255
    return base * (1 - al) + rgba[..., :3].astype(np.float32) * al


def prepare(bg, car, sh, v2):
    """Everything that doesn't depend on the sliders."""
    opa = car[..., 3] > 128
    ys, xs = np.where(car[..., 3] > 12)
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    cols = np.where(opa.any(0))[0]
    yb = np.full(W, -1, np.int32)
    yb[cols] = H - 1 - np.argmax(opa[::-1, cols], axis=0)       # lowest car pixel per column

    # ground line under the car (follows the wheels), extended flat to the sides
    line = yb[cols].astype(np.float32)
    line = cv2.dilate(line[None], np.ones((1, max(3, int(0.08 * bw)) | 1), np.uint8))[0]
    line = cv2.GaussianBlur(line[None], (0, 0), max(2, 0.03 * bw))[0]
    ybf = np.interp(np.arange(W), cols, line).astype(np.float32)

    # floor starts at the wall/floor line of the backdrop: the strongest horizontal edge between
    # 30% of the height and the top of the car's ground line (searched outside the car)
    g = np.abs(np.diff(cv2.GaussianBlur(bg.mean(-1), (0, 0), 2), axis=0))
    g[cv2.dilate(opa.astype(np.uint8), np.ones((15, 15), np.uint8))[1:] > 0] = 0
    prof = g.mean(1)
    lo, hi = int(0.3 * H), int(max(0.3 * H + 2, ybf.min() - 0.05 * bh))
    horizon = lo + int(np.argmax(prof[lo:hi])) + 2
    yy = np.arange(H, dtype=np.float32)[:, None]
    floor = np.clip((yy - horizon) / 12.0, 0, 1) * np.ones((1, W), np.float32)   # soft 12 px start

    # which columns show a real tyre touching the ground (for the hidden-wheel rule)
    lum = car[..., :3].astype(np.float32).mean(-1)
    win = (yy >= (yb[None, :] - 0.12 * bh)) & (yy <= yb[None, :]) & (yb[None, :] >= 0)
    vis = (win & (car[..., 3] > 200) & (lum < 70)).sum(0) / np.maximum(win.sum(0), 1)
    vis = np.clip((vis - 0.25) / 0.35, 0, 1).astype(np.float32)
    d_ = max(3, int(0.05 * bw)); ybp = np.pad(yb.astype(np.float32), d_, mode="edge")
    bump = np.maximum(ybp[2 * d_:], ybp[:-2 * d_])
    vis *= np.clip((yb - bump) / (0.02 * bh), 0, 1) * (yb >= 0)
    vis = cv2.dilate(vis[None], np.ones((1, 2 * d_ + 1), np.uint8))[0]
    vis = cv2.GaussianBlur(vis[None], (0, 0), max(3, 0.015 * bw))[0]
    # zone right under/below the car where a wheel reflection would be drawn
    zone = ((yy >= ybf[None, :] - 0.05 * bh) & (yy <= ybf[None, :] + 0.6 * bh)).astype(np.float32)
    band = np.zeros(W, np.float32); band[cols.min():cols.max() + 1] = 1
    zone *= cv2.GaussianBlur(band[None], (0, 0), max(4, 0.03 * bw))[0][None, :]
    zone = cv2.GaussianBlur(zone, (0, 0), 4)

    # v2's slightly offset car edge (light ghost line) right next to the real car -> keep it dark
    ring = cv2.dilate((car[..., 3] > 20).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ring = cv2.GaussianBlur((ring & (car[..., 3] < 250)).astype(np.float32), (0, 0), 2.0)

    # AutoBG shadow: keep only its dark pixels outside the car
    sh = sh.copy()
    outside = car[..., 3] < 128
    sh[..., 3][outside & (sh[..., :3].max(-1) > 45)] = 0

    # see-through gaps inside the car outline (rims, soft tyre edges) -> fill from v2
    solid = (car[..., 3] > 40).astype(np.uint8)
    ff = np.pad(1 - solid, 1, constant_values=1); cv2.floodFill(ff, None, (0, 0), 2)
    inside = ((ff[1:-1, 1:-1] == 1) | (solid == 1)).astype(np.uint8)
    inside = cv2.GaussianBlur(cv2.erode(inside, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)

    return dict(bg=bg, car=car, sh=sh, v2=v2, bw=bw, bh=bh, ybf=ybf, floor=floor, vis=vis,
                zone=zone, ring=ring, inside=inside[..., None])




def clean(P, hidden=0.3, lift=10, origcar=False, opacity=1.0, length=0.4, fade=1.0, shadow=True, even=0.6, straight=1.0, paste_car=True, color=1.0, fstart=0.0, tstyle=True, tlen=0.6, far_blur=0.4, far_op=0.9):
    v2, bg, car = P["v2"], P["bg"], P["car"]
    R = v2.copy()
    carmask = cv2.dilate((car[..., 3] > 20).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    carmask = cv2.GaussianBlur(carmask, (0, 0), 1.5)[..., None]
    zone = P["zone"][..., None]                       # floor right under / below the car

    # 1. hidden wheels -> soft, faint
    force = P.get("force_hidden")
    if force is not None or hidden < 1 or P.get("hide") is not None:
        # keep everything between the visible wheels (sills, body) at full detail;
        # the rule only softens wheels v2 drew outside them (far-side / hidden wheels)
        vis = P["vis"].copy()
        on = np.where(vis > 0.5)[0]
        if len(on) and not P.get("vis_strict"):
            vis[on.min():on.max() + 1] = 1.0
            # wide, smooth hand-over between visible and hidden wheels (a narrow one drew an outline)
            vis = cv2.GaussianBlur(vis[None], (0, 0), max(6, 0.04 * P["bw"]))[0]
        if force is not None:
            # front / rear view: ALL of v2's tyre shapes go (they sit beside the real tyres), the floor keeps
            # their tone; the real tyres are mirrored exactly under themselves later
            k = np.full((1, vis.shape[0], 1), float(force), np.float32)
        else:
            k = hidden + (1 - hidden) * vis[None, :, None]
        k = 1 - (1 - k) * zone
        # replace a hidden wheel's reflection with the plain floor tone around it (heavy blur that
        # ignores the car itself), so it fades away instead of turning into a soft blob with an edge
        m = (car[..., 3] < 10).astype(np.float32)
        sg = max(15, 0.12 * P["bw"])
        soft = cv2.GaussianBlur(R * m[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(m, (0, 0), sg), 1e-3)[..., None]
        if force is None and hidden > 0.1:
            # the 40 % of v2's hidden-tyre reflection that is kept: bright alloy parts removed (small opening),
            # only in the hidden-wheel columns / floor zone, so nothing else changes shape
            ko_ = int(max(5, 0.08 * P.get("wheel_h", 0.1 * P["bw"]))) | 1
            opn_ = np.minimum(cv2.morphologyEx(R, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko_, ko_))), R)
            hw_ = np.clip((1 - k) / max(1e-3, 1 - hidden), 0, 1)      # 1 where a hidden wheel, 0 at visible ones
            R = R * (1 - hw_) + opn_ * hw_
        R = R * k + soft * (1 - k)
        if P.get("hide") is not None and force is None:
            # tyres that are only a sliver in the photo: v2 draws a whole alloy there. Keep its dark tyre
            # reflection but blur away the rim detail (medium blur: the dark shape stays, spokes go)
            sg2 = max(1.5, 0.006 * P["bw"]) * far_blur
            med = cv2.GaussianBlur(v2 * m[..., None], (0, 0), sg2) / np.maximum(cv2.GaussianBlur(m, (0, 0), sg2), 1e-3)[..., None]
            # NO alloy in a far tyre's reflection (the inside tyre never mirrors its rim that brightly): a
            # morphological opening removes every bright part smaller than ~the rim (spokes, whole rim disc)
            # and leaves only the dark tyre tone
            kk_ = int(min(max(3, P.get("hide_k", 5)), 0.15 * P.get("wheel_h", 0.1 * P["bw"]))) | 1
            med = cv2.morphologyEx(med, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk_, kk_)))
            med = cv2.GaussianBlur(med, (0, 0), 1.0)
            hz = (P["hide"][None, :, None] * zone)
            if P.get("hide_mode") == "flip":
                # third tyre: v2's version (drawn offset, with alloy) goes -> plain floor tone; the real tyre is
                # mirrored under itself in finish()
                R = R * (1 - 0.9 * hz) + soft * (0.9 * hz)
            else:
              R = R * (1 - hz) + np.minimum(med, R) * hz
            # ... and a bit fainter (60 %), so the alloy is not prominent - v2's tyre reflection stays
              R = bg + (R - bg) * (1 - (1 - far_op) * hz)

    # 2. no white patch: under the car the floor can only get darker than the backdrop (+ lift)
    # (brightness only, per pixel - capping each colour channel separately turned light / coloured car
    #  bodies into grey reflections)
    over_ = np.maximum(R.mean(-1, keepdims=True) - (bg.mean(-1, keepdims=True) + lift), 0)
    # light cars (silver / white): keep part of their light reflection - soft cap, at most ~25 levels brighter
    # than the floor (a hard cap removed the whole bumper / body reflection of silver cars)
    KB = float(globals().get("BRIGHT_KEEP", 25.0))
    if KB > 0:
        over_ = over_ - KB * (1 - np.exp(-over_ / KB))
    capped = R - over_
    R = R * (1 - zone) + capped * zone

    # even: bright spots are compressed and the reflection's patchy contrast is smoothed into an even tone
    if even > 0:
        D = R - bg
        # bright spots (rims, lights) are compressed, not removed, so light-coloured cars still reflect
        Dp = np.maximum(D, 0)
        D = np.minimum(D, 0) + Dp / (1 + Dp / 30.0)
        Ds = cv2.GaussianBlur(D, (0, 0), max(3, 0.02 * P["bw"]))  # broad, even darkening
        Dd = D - Ds                                               # fine detail (wheel spokes, edges)
        Re = bg + Ds + (1 - 0.6 * even) * Dd
        R = R * (1 - even) + Re * even

    # reflection colour: strengthen the car's colour in the reflection (1 = as v2 drew it)
    if color != 1.0:
        D = R - bg
        Dm = D.mean(-1, keepdims=True)
        R = bg + Dm + color * (D - Dm)

    # opacity + fade: full reflection strength x opacity at the tyres, fading to nothing after
    # `length` x car height; floor behind the car keeps the same opacity, walls stay your backdrop
    yy = np.arange(G.H, dtype=np.float32)[:, None]
    # fade: 0 = follows each wheel's contact (curved); 1 = PERSPECTIVE fade: the car's footprint outline on the
    # floor (the straight lines under the front / rear and along the side, through the lowest points: bumper
    # corner, front tyre, rear tyre) and the reflection fades at right angles away from each of those lines
    L = max(1.0, length * P["bh"])
    L = max(1.0, min(L, 0.9 * (G.H - float(P["ybf"].max()))))       # fade ends inside the frame, never cut by its edge
    t = (yy - P["ybf"][None, :]) / L
    if straight > 0:
        cols = np.where((car[..., 3] > 128).any(0))[0]
        if len(cols) > 10:
            yb_ = P["ybf"][cols].astype(np.float64)
            # lowest outline (convex hull from below) of the car's bottom edge, image y pointing down
            hull = []
            for x_, y_ in zip(cols.astype(np.float64), yb_):
                while len(hull) >= 2:
                    (x1, y1), (x2, y2) = hull[-2], hull[-1]
                    if (x2 - x1) * (y_ - y1) - (y2 - y1) * (x_ - x1) >= 0:   # keep it convex towards the floor
                        hull.pop()
                    else:
                        break
                hull.append((x_, y_))
            hx, hy = np.array([h[0] for h in hull]), np.array([h[1] for h in hull])
            yh = np.interp(cols, hx, hy)
            Z = np.ones((G.H, G.W), np.uint8)
            for x_, y_ in zip(cols, yh):
                Z[:int(np.floor(y_)) + 1, x_] = 0           # on / above the footprint line
            if tstyle and P.get("wheel_h"):
                # like the side-shot tyre reflection: straight DOWN from the footprint line through the tyre
                # contacts, over `tlen` x wheel height - every tyre starts equally strong and fades the same
                L = max(1.0, tlen * P["wheel_h"])
                yh_all = np.interp(np.arange(G.W), cols, yh)
                L = max(1.0, min(L, 0.9 * (G.H - float(yh_all[cols].max()))))   # fade ends inside the frame
                t = (yy - yh_all[None, :]) / L
            else:
                dist = cv2.distanceTransform(Z, cv2.DIST_L2, 5)
                t = t * (1 - straight) + (dist / L) * straight
    if fstart > 0:
        # the fade starts ABOVE the footprint line: the reflection is already partly faded where the tyres
        # touch the floor, so the tyres' own reflection fades too, not only the body's
        t = np.maximum(t, 0) + fstart * P["bh"] / L
    prof = np.where(t <= 0, 1.0, np.clip(1 - t, 0, 1) ** fade).astype(np.float32)
    if P.get("hide") is not None and P.get("force_hidden") is None:
        # far tyres that are only a sliver: their reflection fades out faster than the others
        hc_ = P["hide"][None, :]
        prof = prof * (1 - hc_) + (prof ** 1.15) * hc_
    alpha = (opacity * prof * P["floor"])[..., None]
    out = bg * (1 - alpha) + R * alpha

    if shadow:
        out = G.over(out, P["sh"])                 # AutoBG shadow, unchanged
    # the car itself stays fully solid
    if paste_car:                                  # (the app puts your own car on top instead)
        out = out * (1 - carmask) + v2 * carmask
    if origcar:
        out = G.over(out, car)
    return np.clip(out, 0, 255).astype(np.uint8)




# ------------------------------------------------------------------------------------------ app
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--lora", default=os.path.join(HERE, "models", "autobg_reflection_v2.safetensors"))
ap.add_argument("--backdrop", default=os.path.join(HERE, "floor_satin_light.png"))
ap.add_argument("--quant", default="Q4_K_M", help="GGUF size: Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_M ...")
ap.add_argument("--lowvram", action="store_true", help="sequential offload: slow, minimal GPU memory")
ap.add_argument("--full", action="store_true", help="full-quality model on a big GPU (48 GB+, e.g. Vast): no compression")
ap.add_argument("--share", action="store_true", help="public gradio.live link (use on Vast)")
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--batch", help="process every car cut-out (.png) in this folder and exit, no web page")
ap.add_argument("--shift", type=int, default=None, help="--batch: move the reflection up by this many pixels")
ap.add_argument("--front", type=float, default=None, help="--batch: reflection in front of the shadow (0-1)")
ap.add_argument("--autobg_shadow", action="store_true", help="--batch: use the AutoBG shadow file instead of the generated shadow")
ap.add_argument("--set", nargs="*", default=[], help="--batch: override settings, e.g. --set detrend=1 opacity=0.6")
ap.add_argument("--batch_out", default=None, help="output folder for --batch (default: <batch folder>_out)")
ap.add_argument("--team", action="store_true", help="simple page for the team: upload cars, get results (locked settings, no sliders)")
ap.add_argument("--password", default=None, help="--team: protect the page, login user 'autobg' + this password")
ap.add_argument("--bright", type=float, default=25.0, help="light cars: how much lighter than the floor their reflection may be (0 = old hard cap)")
ap.add_argument("--shadow_dark", type=float, default=1.2, help="AutoBG shadow strength x this (1 = original)")
A = ap.parse_args()
SH_DARK = A.shadow_dark
BRIGHT_KEEP = A.bright
os.makedirs(A.out, exist_ok=True)

PROMPT = "Add a realistic floor reflection of the car on a glossy grey showroom floor."
BASE = "Qwen/Qwen-Image-Edit-2511"
W, H = G.W, G.H

# ---------------------------------------------------------------------------------------------- model
print("loading model (first time downloads ~25 GB) ...", flush=True)
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

if A.full:
    pipe = QwenImageEditPlusPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16)
else:
    gguf = hf_hub_download("unsloth/Qwen-Image-Edit-2511-GGUF", f"qwen-image-edit-2511-{A.quant}.gguf")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, config=BASE, subfolder="transformer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    pipe = QwenImageEditPlusPipeline.from_pretrained(BASE, transformer=transformer, text_encoder=text_encoder,
                                                     torch_dtype=torch.bfloat16)
pipe.load_lora_weights(A.lora, adapter_name="refl")
# 4-step distilled LoRA (lightx2v): ~15x fewer model passes than 30 steps with CFG
pipe.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning",
                       weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", adapter_name="light")
if A.full:
    pipe.to("cuda")
elif A.lowvram:
    pipe.enable_sequential_cpu_offload()
else:
    pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=False)
print("model ready", flush=True)

# ------------------------------------------------------------------------------------------ placement
def align(car, sh):
    """AutoBG's shadow canvas can be larger than the cut-out: find where the car sits in it."""
    if car.size == sh.size:
        return car
    cw, ch = car.size; sw, shh = sh.size
    if sw < cw or shh < ch:
        return None
    c = np.asarray(car, np.float32)[::4, ::4]; s = np.asarray(sh, np.float32)
    m = c[..., 3] > 200
    best = None
    for dy in range(0, shh - ch + 1, 2):
        for dx in range(0, sw - cw + 1, 2):
            win = s[dy:dy + ch:4, dx:dx + cw:4]
            if win.shape[:2] != m.shape:
                continue
            err = np.abs(win[..., :3][m] - c[..., :3][m]).mean()
            if best is None or err < best[0]:
                best = (err, dx, dy)
    if best is None or best[0] > 25:
        return None
    pad = Image.new("RGBA", sh.size, (0, 0, 0, 0)); pad.paste(car, (best[1], best[2]))
    return pad


def place(bg_path, car, sh, bottom, width, height):
    """Place the car on the backdrop at the backdrop's FULL resolution. Returns full-res layers;
    the model works on a downscaled copy and only its floor change is scaled back up."""
    bg = Image.open(bg_path).convert("RGBA")
    BW, BH = bg.size
    if sh is None:
        sh = car
    if car.size != sh.size:
        car = align(car, sh)
        if car is None:
            raise gr.Error("Could not line up the cut-out with the shadow image - are they the same shot?")
    box = car.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    bw, bh = box[2] - box[0], box[3] - box[1]
    s = min(width * BW / bw, height * BH / bh)
    size = (max(1, round(car.width * s)), max(1, round(car.height * s)))
    x0 = round(BW / 2 - (box[0] + bw / 2) * s)
    y0 = round(bottom * BH - box[3] * s)

    def layer(img):
        L = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
        im = img.resize(size, Image.LANCZOS); L.paste(im, (x0, y0), im)
        return L
    return bg, layer(car), layer(sh)


def defringe(c, choke=True):
    """AutoBG cut-outs have a wide soft edge that still carries the ORIGINAL photo's light ground /
    sky colour -> shows as a light outline on a darker floor. Re-colour the edge pixels from the
    solid car just inside them and tighten the edge a little."""
    c = c.copy()
    a = c[..., 3].astype(np.float32)
    solid = (a > 250).astype(np.float32)
    rgb = c[..., :3].astype(np.float32)
    r = max(3, int(0.004 * max(c.shape[:2])))
    num = cv2.GaussianBlur(rgb * solid[..., None], (0, 0), r)
    den = cv2.GaussianBlur(solid, (0, 0), r)[..., None]
    inner = num / np.maximum(den, 1e-3)
    edge = ((a < 250) & (a > 0))[..., None]
    c[..., :3] = np.where(edge & (den > 1e-3), np.clip(inner, 0, 255), rgb).astype(np.uint8)
    # choke the outer edge by a couple of pixels: AutoBG cut-outs often keep a thin opaque strip of the
    # original (light) ground under the tyres and along the bumper, which shows as a light rim
    if not choke:                                      # side views: keep the cut-out's own crisp edge
        return c
    k = max(2, int(0.0012 * max(c.shape[:2])))
    core = cv2.erode((a > 128).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), k / 2) * 255
    c[..., 3] = np.minimum(a, core).astype(np.uint8)
    return c


def shadow_only(sh, car):
    """AutoBG's shadow file = car + shadow in one image. Keep only the shadow, and continue it in
    under the car's soft edge. (Cutting it at the car's edge left a thin band without shadow,
    which read as a light outline around the tyres and bumpers.)"""
    a = sh[..., 3].astype(np.float32) / 255
    dark = sh[..., :3].max(-1) <= 45
    # pixels that belong to the car copy inside the shadow file (light, not shadow) must be FILLED,
    # not read as "no shadow" - after the BiRefNet clean-up removes a strip of the cut-out, those
    # pixels lie outside the car and used to leave a light rim between car and shadow
    valid = (car[..., 3] <= 5) & (dark | (sh[..., 3] < 5))
    sa = np.where(valid & dark, a, 0).astype(np.float32)
    vm = valid.astype(np.float32)
    fill, got = sa.copy(), valid.astype(np.float32)
    for sg in (2, 6, 18):
        num = cv2.GaussianBlur(sa * vm, (0, 0), sg)
        den = cv2.GaussianBlur(vm, (0, 0), sg)
        ok = (den > 0.05) & (got == 0)
        fill = np.where(ok, num / np.maximum(den, 1e-4), fill)
        got = np.maximum(got, ok.astype(np.float32))
    col = np.median(sh[..., :3][valid & dark & (a > 0.3)], axis=0) if (valid & dark & (a > 0.3)).sum() > 50 else np.array([8, 8, 8])
    out = np.zeros_like(sh)
    out[..., :3] = col.astype(np.uint8)
    out[..., 3] = np.clip(fill * 255, 0, 255).astype(np.uint8)
    return out


def align_v2(v2, cond, alpha):
    """v2 redraws the car slightly shifted / scaled, and draws its reflection under ITS car. Line v2's
    image up with the real car (matched on the car itself) so the reflection sits under your tyres."""
    m = cv2.dilate((alpha > 128).astype(np.uint8), np.ones((15, 15), np.uint8))
    a = cv2.GaussianBlur(cv2.cvtColor(cond.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    b = cv2.GaussianBlur(cv2.cvtColor(v2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), m, 5)
    except cv2.error:
        return v2, "0 px (match failed)"
    sc = float(np.sqrt(abs(np.linalg.det(warp[:, :2]))))
    if abs(sc - 1) > 0.08 or np.abs(warp[:, 2]).max() > 0.08 * v2.shape[1]:
        return v2, "0 px (implausible match skipped)"
    out = cv2.warpAffine(v2, warp, (v2.shape[1], v2.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT)
    return out, f"{warp[0, 2]:+.1f},{warp[1, 2]:+.1f} px, scale {sc:.3f}"


_BIREF = {}

def birefnet_alpha(img_rgb):
    """BiRefNet foreground matte (0..1) for a PIL RGB image, at that image's size."""
    if "m" not in _BIREF:
        from transformers import AutoModelForImageSegmentation
        m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        _BIREF["m"] = m.to("cuda").eval().half()
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((1024, 1024)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(img_rgb).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        pred = _BIREF["m"](x)[-1].sigmoid()[0, 0].float().cpu().numpy()
    return cv2.resize(pred, img_rgb.size, interpolation=cv2.INTER_LINEAR)


def recut(car, photo=None):
    """Tighten the AutoBG cut-out's LOWER edge (tyres, bumper, sills) with BiRefNet, which separates
    car from ground better. Only ever removes pixels (min of both mattes), never adds; the top of
    the car (roof, glass) is left exactly as AutoBG cut it. Returns (car, info)."""
    try:
        a = np.asarray(car)[..., 3].astype(np.float32) / 255
        if photo is not None and abs(photo.width / photo.height - car.width / car.height) < 0.02:
            src = photo.convert("RGB").resize(car.size, Image.LANCZOS); how = "photo"
        else:
            grey = Image.new("RGBA", car.size, (128, 128, 128, 255))
            src = Image.alpha_composite(grey, car).convert("RGB"); how = "cut-out"
        b = birefnet_alpha(src)
        ys = np.where((a > 0.5).any(1))[0]
        if len(ys) < 10:
            return car, "re-cut skipped (empty cut-out)"
        y_top, y_bot = ys.min(), ys.max()
        w = np.clip((np.arange(car.height) - (y_top + 0.55 * (y_bot - y_top))) / (0.1 * (y_bot - y_top)), 0, 1)[:, None]
        new = a * (1 - w) + np.minimum(a, b) * w           # tighten only in the lower part
        lost = 1 - (new > 0.5).sum() / max(1, (a > 0.5).sum())
        if lost > 0.05:                                     # BiRefNet disagreed a lot -> don't trust it
            return car, f"re-cut skipped (would remove {lost:.0%} of the car)"
        out = np.asarray(car).copy()
        out[..., 3] = np.clip(new * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA"), f"re-cut from {how}: edge tightened ({lost:.1%} of car area)"
    except Exception as e:
        print("BiRefNet re-cut failed:", e, flush=True)
        return car, f"re-cut failed ({type(e).__name__}), AutoBG cut-out used"


def put_car(res, car, v2_low, edge_w=1.0):
    """Your original car on top - except a thin band along its LOWER edge (tyres, bumper, sills),
    which is taken from v2's own (aligned) image. v2 draws that edge together with its reflection,
    so the car-to-floor join is seamless whatever the cut-out's edge looks like (fringe, ground strip)."""
    a = car[..., 3].astype(np.float32) / 255
    H2, W2 = a.shape
    if edge_w <= 0:
        return G.over(res, car)
    k = max(3, int(0.004 * W2 * edge_w))
    inner = cv2.erode((a > 0.9).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), k / 2)
    ys = np.where((a > 0.5).any(1))[0]
    y0, y1 = ys.min(), ys.max()
    lower = np.clip((np.arange(H2) - (y0 + 0.35 * (y1 - y0))) / (0.06 * (y1 - y0)), 0, 1)[:, None]
    use_car = 1 - (1 - inner) * lower                     # 1 = original pixel, 0 = v2 pixel
    v2u = cv2.resize(v2_low, (W2, H2), interpolation=cv2.INTER_CUBIC)
    col = car[..., :3].astype(np.float32) * use_car[..., None] + v2u * (1 - use_car[..., None])
    # in the lower band also use v2's coverage, so the car ends where v2's car ends (no gap, no rim)
    a2 = a * use_car + np.maximum(a, inner) * (1 - use_car)
    return res * (1 - a2[..., None]) + col * a2[..., None]


def gen_shadow(car, dark=0.55, soft=1.0, contact=0.8):
    """Studio shadow built from the car's own outline (no AutoBG shadow file needed), returned as an
    RGBA layer like the AutoBG one: (1) soft ambient shadow under the whole car along its ground line,
    (2) tight dark contact patch under each tyre. It starts INSIDE the car outline, so there is never
    a gap or rim between car and shadow."""
    a = car[..., 3] > 128
    H2, W2 = a.shape
    cols = np.where(a.any(0))[0]
    ys = np.where(a.any(1))[0]
    out = np.zeros((H2, W2, 4), np.uint8)
    if len(cols) < 10:
        return out
    bw, bh = cols.max() - cols.min(), ys.max() - ys.min()
    bot = np.full(W2, -1.0, np.float32)
    bot[cols] = H2 - 1 - np.argmax(a[::-1, cols], axis=0)
    # ground line: follows the lowest points (tyres), bridged flat between them
    win = max(3, int(0.12 * bw)) | 1
    gl = cv2.dilate(np.where(bot < 0, 0, bot)[None], np.ones((1, win), np.float32))[0]
    gl = cv2.GaussianBlur(gl[None], (0, 0), max(2, 0.02 * bw))[0]
    gl[:cols.min()] = gl[cols.min()]; gl[cols.max():] = gl[cols.max()]
    yy = np.arange(H2, dtype=np.float32)[:, None]
    span = np.zeros(W2, np.float32); span[cols.min():cols.max() + 1] = 1
    span = cv2.GaussianBlur(span[None], (0, 0), max(3, 0.02 * bw))[0]
    # 1. ambient: a band from a little above the ground line to a little below it, softly blurred
    up_, dn_ = 0.10 * bh, 0.025 * bh * soft
    amb = ((yy > gl[None] - up_) & (yy < gl[None] + dn_)).astype(np.float32) * span[None]
    amb = cv2.GaussianBlur(amb, (0, 0), sigmaX=max(3, 0.025 * bw * soft), sigmaY=max(2, 0.012 * bh * soft))
    amb = amb / max(amb.max(), 1e-3) * dark
    # 2. contact: columns where the car really touches the line (tyres), a thin dark patch under them
    tyre = (bot >= 0) & (bot >= gl - 0.012 * bh)
    tyre = cv2.dilate(tyre.astype(np.uint8)[None], np.ones((1, max(3, int(0.01 * bw)) | 1), np.uint8))[0].astype(np.float32)
    con = ((yy > gl[None] - 0.03 * bh) & (yy < gl[None] + 0.008 * bh)).astype(np.float32) * tyre[None]
    con = cv2.GaussianBlur(con, (0, 0), sigmaX=max(2, 0.006 * bw), sigmaY=max(1.5, 0.004 * bh)) * contact
    # 3. under the car: the gap between the car's bottom edge (sills, bumpers) and the ground line is the
    #    floor seen UNDER the car - in reality it is in deep shadow. Without this the light backdrop floor
    #    showed through between the wheels.
    botf = np.where(bot < 0, gl, bot)
    und = ((yy > botf[None] - 2) & (yy < gl[None] + 0.01 * bh)).astype(np.float32) * span[None]
    und = cv2.GaussianBlur(und, (0, 0), sigmaX=max(2, 0.008 * bw), sigmaY=max(2, 0.006 * bh)) * min(0.9, dark + 0.3)
    sa = np.clip(1 - (1 - amb) * (1 - con) * (1 - und), 0, 0.95)
    out[..., :3] = 6
    out[..., 3] = (sa * 255).astype(np.uint8)
    return out


def soft_tyre(res, car, wheels, strength=0.7):
    """faint reflection for a far tyre that is only a sliver in the photo: soft dark oval below its lowest point"""
    H_, W_ = res.shape[:2]
    a = car[..., 3] > 128
    out = res.copy()
    for w_ in wheels:
        X0, Y0, X1, Y1 = [int(v) for v in w_["box"]]
        bots = []
        for x in range(max(0, X0), min(W_, X1)):
            c = np.where(a[max(0, Y0):min(H_, Y1 + 20), x])[0]
            bots.append(max(0, Y0) + c.max() if len(c) else -1)
        bots = np.array(bots)
        if not len(bots) or bots.max() < 0:
            continue
        yb = bots.max()
        cx = max(0, X0) + int(np.mean(np.where(bots >= yb - 2)[0]))
        rx = 0.45 * (X1 - X0); ry = 0.6 * rx
        y0_, y1_ = int(yb - 2), int(min(H_, yb + ry * 2))
        x0_, x1_ = int(max(0, cx - rx * 1.5)), int(min(W_, cx + rx * 1.5))
        if y1_ <= y0_ or x1_ <= x0_:
            continue
        Y, X = np.mgrid[y0_:y1_, x0_:x1_].astype(np.float32)
        e = np.clip(1 - ((X - cx) / rx) ** 2 - ((Y - yb) / ry) ** 2, 0, 1) ** 0.5 * (Y >= yb - 1)
        e = cv2.GaussianBlur(e.astype(np.float32), (0, 0), max(2.0, 0.12 * rx))
        lin = np.clip(1 - (Y - yb) / (ry * 1.1), 0, 1)
        k = (strength * e * lin * (1 - car[y0_:y1_, x0_:x1_, 3] / 255.0))[..., None]
        out[y0_:y1_, x0_:x1_] = out[y0_:y1_, x0_:x1_] * (1 - k) + np.array([22, 22, 22], np.float32) * k
    return out


_TMPL = {}

def tyre_template(res, car, ws, strength=0.225):
    """front / back tyres that are mostly hidden behind the bumper: mirroring the visible sliver gives a cut-off
    block -> paste a ready-made reflection taken from a perfect, fully visible front / back tyre (Toyota),
    scaled to this tyre's width, at the floor contact. Darken-only, never over the car."""
    if "T" not in _TMPL:
        p_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tyre_refl.png")
        t_ = cv2.imread(p_, cv2.IMREAD_UNCHANGED)
        _TMPL["T"] = None if t_ is None else t_.astype(np.float32) / 65535.0
    T = _TMPL["T"]
    if T is None:
        return None
    tw0 = T.shape[1] / 1.6                       # template = tyre width + 30 % each side, starts 0.15 x width above
    out = res.copy()
    H, W = out.shape[:2]
    for w in ws:
        X0, Y0, X1, Y1 = [float(v) for v in w["box"]]
        f = (X1 - X0) / tw0
        Tm = cv2.resize(T, (max(2, int(T.shape[1] * f)), max(2, int(T.shape[0] * f))), interpolation=cv2.INTER_AREA)
        x0 = int(round(X0 - 0.3 * tw0 * f)); y0 = int(round(float(w["yb"]) - 0.15 * tw0 * f))
        h_, w_ = Tm.shape
        ya, xa, yb_, xb_ = max(0, y0), max(0, x0), min(H, y0 + h_), min(W, x0 + w_)
        if yb_ <= ya or xb_ <= xa:
            continue
        k = Tm[ya - y0:yb_ - y0, xa - x0:xb_ - x0] * strength * (1 - car[ya:yb_, xa:xb_, 3].astype(np.float32) / 255.0)
        out[ya:yb_, xa:xb_] *= (1 - k)[..., None]
    return out


def is_frontrear(ws, car):
    """straight front / rear shot: a narrow car (width < 1.8 x height) whose two lowest tyres sit at the
    left and right ends on the same ground line"""
    if len(ws) < 2:
        return False
    a = car[..., 3] > 128
    ys, xs = np.where(a)
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    w2 = sorted(ws, key=lambda w: -w["yb"])[:2]
    c = sorted((w["box"][0] + w["box"][2]) / 2 for w in w2)
    return (bw < 1.8 * bh and abs(w2[0]["yb"] - w2[1]["yb"]) < 0.06 * bh
            and c[0] < xs.min() + 0.35 * bw and c[1] > xs.max() - 0.35 * bw)


def is_side(ws, car):
    """side view: a long car (width > 1.7 x height; short hatchbacks are ~2.0) with two round, full tyres far apart (front + rear).
    Checked on the 20 test cars: finds the 4 side shots, none of the front / rear / 3-4 shots."""
    if len(ws) < 2:
        return False
    a = car[..., 3] > 128
    ys, xs = np.where(a)
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    w2 = sorted(ws, key=lambda w: -(w["box"][3] - w["box"][1]))[:2]          # the two biggest tyres
    c = [(w["box"][0] + w["box"][2]) / 2 for w in w2]
    round_ = all(0.75 < (w["box"][2] - w["box"][0]) / max(1, w["box"][3] - w["box"][1]) < 1.35 for w in w2)
    return bw > 1.7 * bh and round_ and abs(c[0] - c[1]) > 0.5 * bw and abs(w2[0]["yb"] - w2[1]["yb"]) < 0.12 * bh


def model_size(BW, BH):
    """Working size for the model: about 1 megapixel, same shape as the backdrop, multiple of 16."""
    k = (1024 * 768 / (BW * BH)) ** 0.5
    return max(256, round(BW * k / 16) * 16), max(256, round(BH * k / 16) * 16)


def to_cv(im, rgba=False):
    a = np.asarray(im)
    return cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA if rgba else cv2.COLOR_RGB2BGR)


STATE = {}

def generate(car_f, sh_f, bg_f, bottom, width, height, steps, seed, fast, light_w, photo_f=None, do_recut=True, skip_side=False):
    if car_f is None:
        raise gr.Error("Drop the car cut-out (transparent PNG) first.")
    car = Image.open(car_f).convert("RGBA")
    sh = Image.open(sh_f).convert("RGBA") if sh_f else None
    cut_info = "AutoBG cut-out as is"
    if do_recut:
        car, cut_info = recut(car, Image.open(photo_f) if photo_f else None)
    bg_hi, car_hi, sh_hi = place(bg_f or A.backdrop, car, sh, bottom, width, height)
    w, h = model_size(*bg_hi.size)
    G.W, G.H = w, h                                    # finishing works at the model's size
    bg = bg_hi.resize((w, h), Image.LANCZOS)
    carL, shL = car_hi.resize((w, h), Image.LANCZOS), sh_hi.resize((w, h), Image.LANCZOS)
    cond = Image.alpha_composite(bg, carL).convert("RGB")
    car_raw = to_cv(car_hi, True)
    ws = WH.detect(car_raw)
    side = is_side(ws, car_raw)
    frontrear = (not side) and is_frontrear(ws, car_raw)
    if not side and not frontrear:
        # straight front / rear shots are near mirror-symmetric (0.89-0.99 on the test cars, 3-4 views <= 0.83):
        # treat them as front / rear even when the tyre model found only one (or two odd) tyres
        am_ = car_raw[..., 3] > 128
        yy_, xx_ = np.where(am_)
        cr_ = am_[yy_.min():yy_.max() + 1, xx_.min():xx_.max() + 1]
        sym_ = (cr_ & cr_[:, ::-1]).sum() / max(1, (cr_ | cr_[:, ::-1]).sum())
        if sym_ >= 0.88 and ws:
            ws = sorted(ws, key=lambda q: -q["yb"])[:2]
            if len(ws) == 1:                            # the other tyre: the same, mirrored about the car's centre
                ws.append(WH.mirror_wheel(ws[0], (xx_.min() + xx_.max()) / 2.0))
            frontrear = True
    t = time.time()
    if skip_side and side:
        # side view gets the tyre reflection - v2 isn't used, so don't spend ~35 s running it
        out = cond
    elif fast:
        pipe.set_adapters(["refl", "light"], adapter_weights=[1.0, float(light_w)])
        steps, cfg = 4, 1.0
    else:
        pipe.set_adapters(["refl"], adapter_weights=[1.0])
        cfg = 4.0
    if not (skip_side and side):
        out = pipe(image=[cond], prompt=PROMPT, negative_prompt=" ", true_cfg_scale=cfg,
                   num_inference_steps=int(steps), height=h, width=w,
                   generator=torch.Generator("cpu").manual_seed(int(seed))).images[0]
    secs = time.time() - t
    v2 = to_cv(out.resize((w, h), Image.LANCZOS)).astype(np.float32)
    v2, shift = align_v2(v2, to_cv(cond).astype(np.float32), np.asarray(carL)[..., 3])
    P = G.prepare(to_cv(bg.convert("RGB")).astype(np.float32), to_cv(carL, True), to_cv(shL, True), v2)
    # which columns really have a tyre: from YOUR tyre model (the old dark-rubber guess put v2's invented
    # far-side wheels in the wrong place). Only there v2's wheel reflections are kept as drawn; anything
    # wheel-like elsewhere (a hidden wheel v2 invented) is faded by the "Hidden wheels" slider.
    if ws:
        sc_ = w / car_raw.shape[1]
        vis_ = np.zeros(w, np.float32)
        hmax_ = max(wh_["box"][3] - wh_["box"][1] for wh_ in ws)
        hide_ = np.zeros(w, np.float32)
        main_ = sorted(ws, key=lambda q: -(q["box"][3] - q["box"][1]))[:2]     # the two main tyres
        for wh_ in ws:
            x0_, y0_, x1_, y1_ = wh_["box"]
            c0_, c1_ = max(0, int(x0_ * sc_)), min(w, int(np.ceil(x1_ * sc_)) + 1)
            if not any(wh_ is q for q in main_):     # the THIRD tyre (hardly visible, beside the two main ones)
                # tyre only slightly visible (far side, behind the body): v2 draws a full alloy there -> its
                # reflection becomes the smooth floor tone, no rim (widened a bit: v2 places it loosely)
                pad_ = int(0.5 * (c1_ - c0_)) + 2
                hide_[max(0, c0_ - pad_):min(w, c1_ + pad_)] = 1.0
                P["hide_k"] = max(P.get("hide_k", 0), 0.6 * (c1_ - c0_))
            vis_[c0_:c1_] = 1.0
        if hide_.any():
            P["hide_mode"] = "flip"
            P["hide"] = np.clip(cv2.GaussianBlur(hide_[None], (0, 0), max(2.0, 0.01 * w))[0] * 1.2, 0, 1)
        P["vis"] = cv2.GaussianBlur(vis_[None], (0, 0), max(2.0, 0.01 * w))[0]
        P["wheel_h"] = float(max(wh_["box"][3] - wh_["box"][1] for wh_ in ws)) * sc_
        P["vis_strict"] = True
        if frontrear:
            P["force_hidden"] = 0.0                    # v2's tyres are misplaced there: remove them completely
    # full-resolution layers for the final image
    car_c = defringe(car_raw)
    sh_c = shadow_only(to_cv(sh_hi, True), car_raw) if sh is not None else None
    hm_ = max([wh_["box"][3] - wh_["box"][1] for wh_ in ws], default=0)
    main2_ = sorted(ws, key=lambda q: -(q["box"][3] - q["box"][1]))[:2]
    slivers = [wh_ for wh_ in ws if not any(wh_ is q for q in main2_)]
    STATE.update(slivers=slivers, frontrear=frontrear, car_sharp=defringe(car_raw, False), wheels=ws, side=side, P=P, wh=(w, h), bg_hi=to_cv(bg_hi.convert("RGB")).astype(np.float32), car_hi=car_c, sh_hi=sh_c,
                 name=os.path.splitext(os.path.basename(car_f))[0].replace(" ", "_"), v2=out, cond=cond)
    return cond, out, f"view: {'SIDE -> tyre reflections' if side else ('FRONT/REAR -> v2 reflection + exact tyre reflections' if frontrear else '3-4 -> v2 reflection')} ({len(ws)} tyres); model: {secs:.1f} s ({int(steps)} steps, {'fast' if fast else 'full'}), output {bg_hi.size[0]}x{bg_hi.size[1]}, v2 car re-aligned by {shift}; {cut_info}"


def finish(opacity, length, fade, even, hidden, lift, shadow, origcar, straight, detrend=1.0, edge_w=1.0, tone=0.0, shift_px=0, refl_front=0.0, gen_sh=False, sh_dark=0.55, sh_soft=1.0, sh_contact=0.8, mode="Auto", t_str=0.6, t_len=0.55, t_det=0.1, t_shift=0.0, t_con=0.4, t_gloss=0.5, t_gblur=0.0, refl_col=1.5, fstart=0.0, tstyle=True, tlen=0.6):
    if "P" not in STATE:
        return None
    P = STATE["P"]
    G.W, G.H = STATE["wh"]
    # no v2-car paste: it put v2's pixels at FULL strength in a margin around the car, while the
    # reflection elsewhere is at the chosen opacity -> a dark outline hugging tyres and bumper
    low = C.clean(P, hidden, lift, False, opacity, length, fade, False, even, straight, paste_car=False, color=refl_col, fstart=fstart, tstyle=tstyle, tlen=tlen).astype(np.float32)
    # only the floor change (reflection) is taken from the model, scaled up to full resolution
    delta = low - P["bg"]
    # v2 also re-tones the whole floor (usually a bit lighter). With the fade starting at the car that
    # shift is strongest right next to the car -> a light glow/halo around it. Remove v2's floor-tone
    # change (measured on floor away from the car) so only the car's own reflection is kept.
    if detrend > 0:
        far = (P["floor"] > 0.5) & (P["zone"] < 0.05) & (P["car"][..., 3] < 5)
        if far.sum() > 500:
            sg = 0.08 * P["bg"].shape[1]
            m = far.astype(np.float32)
            num = cv2.GaussianBlur(delta * m[..., None], (0, 0), sg)
            den = cv2.GaussianBlur(m, (0, 0), sg)[..., None]
            glob_ = np.median(delta[far], axis=0)
            trend = np.where(den > 0.05, num / np.maximum(den, 1e-3), glob_)
            delta = delta - detrend * trend * P["floor"][..., None]
    # inside the car outline the model's change is v2's redrawn car - irrelevant (your car goes on top)
    # but it must not leak out when scaled up, and it must not be zeroed either (that left a light
    # line of bare floor between the car and its reflection). Fill it from the floor just outside.
    # (the finishing step pastes v2's car with a few px margin, so skip that margin as well)
    # (v2 is now aligned to the real car, so keep its pixels right up to the tyres: skipping a margin
    #  there removed v2's dark tyre contact and left a light gap between tyre and reflection)
    out_m = (P["car"][..., 3] < 10).astype(np.float32)
    fill, got = np.zeros_like(delta), np.zeros(out_m.shape + (1,), np.float32)
    for sg in (2, 6, 20, 60):                          # nearest floor values first, wider if needed
        num = cv2.GaussianBlur(delta * out_m[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(out_m, (0, 0), sg)[..., None]
        ok = (den > 0.05) & (got == 0)
        fill = np.where(ok, num / np.maximum(den, 1e-4), fill)
        got = np.maximum(got, ok.astype(np.float32))
    delta = delta * out_m[..., None] + fill * (1 - out_m[..., None])
    # v2 redraws the car a few pixels bigger/offset; right around the real car its white body edge
    # would show as a light outline. In that ring the reflection may only darken, never lighten.
    rk = max(5, int(0.015 * P["bw"])) | 1
    ring = cv2.dilate((P["car"][..., 3] > 10).astype(np.uint8), np.ones((rk, rk), np.uint8)).astype(np.float32)
    ring = cv2.GaussianBlur(ring, (0, 0), rk / 3)[..., None]
    delta = delta * (1 - ring) + np.minimum(delta, 0) * ring
    # tone down LIGHT parts of the reflection (e.g. a white car body mirrored in the floor) under and
    # below the car. ("Max brightening" can't do this when v2 has darkened the floor around it.)
    if tone > 0:
        Rl = P["bg"] + delta
        L = Rl.mean(-1, keepdims=True) / 255
        mapped = Rl * (1 - 0.75 * tone * np.clip((L - 0.15) / 0.6, 0, 1))
        w = cv2.GaussianBlur(P["zone"] * P["floor"], (0, 0), 6)[..., None]
        delta = delta + (mapped - Rl) * w
    bg_hi = STATE["bg_hi"]
    H2, W2 = bg_hi.shape[:2]
    up = cv2.resize(delta, (W2, H2), interpolation=cv2.INTER_CUBIC)
    if shift_px:                                       # move the reflection up (+) / down (-)
        M = np.float32([[1, 0, 0], [0, 1, -float(shift_px)]])
        up = cv2.warpAffine(up, M, (W2, H2), borderMode=cv2.BORDER_REPLICATE)
    use_t = mode == "Tyre reflection" or (mode == "Auto" and STATE.get("side"))
    if use_t:
        # side view: plain backdrop floor + ONLY the real tyres mirrored (your trained tyre model)
        res = bg_hi.copy()                              # tyre reflection goes on AFTER the shadow (below)
        up = np.zeros_like(up)
        edge_w = 0.0
    else:
        if not STATE.get("frontrear"):
            # 3-4 views: the reflection stopped in a hard vertical edge straight down from the car's outermost point
            # (front bumper corner). Fade it out smoothly over a band at the car's left / right ends that widens
            # with depth below the car, so the edge melts away instead of sticking out
            ca_ = STATE["car_hi"][..., 3] > 128
            cols_ = np.where(ca_.any(0))[0]
            if len(cols_) > 10:
                x0c, x1c = float(cols_.min()), float(cols_.max())
                botc = np.full(W2, -1.0, np.float32)
                botc[cols_] = H2 - 1 - np.argmax(ca_[::-1, cols_], 0)
                botc = np.interp(np.arange(W2), cols_, botc[cols_]).astype(np.float32)   # car's bottom per column
                # smooth it (lowest points win, then blurred): the jump between bumper and tyre bottoms made the
                # band jump too -> a vertical line in the reflection
                kd_ = max(3, int(0.1 * (x1c - x0c))) | 1
                botc = cv2.dilate(botc[None], np.ones((1, kd_), np.uint8))[0]
                botc = cv2.GaussianBlur(botc[None], (0, 0), max(1.0, 0.05 * (x1c - x0c)))[0]
                yy_ = np.arange(H2, dtype=np.float32)[:, None]
                xx_ = np.arange(W2, dtype=np.float32)[None, :]
                dep_ = np.maximum(yy_ - botc[None, :], 0)
                b_ = 0.06 * (x1c - x0c) + 0.8 * dep_
                wl_ = np.clip((xx_ - (x0c - 0.25 * b_)) / np.maximum(b_, 1), 0, 1)
                wr_ = np.clip(((x1c + 0.25 * b_) - xx_) / np.maximum(b_, 1), 0, 1)
                wx_ = np.minimum(wl_, wr_)
                wx_ = wx_ * wx_ * (3 - 2 * wx_)                 # smoothstep
                # vertical lines INSIDE the reflection (mirrored vertical edges) melt too: blurred sideways, more
                # the further below the car (like a real floor reflection getting softer with distance)
                smax_ = max(2.0, 0.03 * W2)
                sigs_ = [0.0] + [smax_ / 2 ** i for i in range(4, -1, -1)]
                lv_ = [up] + [cv2.GaussianBlur(up, (0, 0), sigmaX=s_, sigmaY=0.01) for s_ in sigs_[1:]]
                sd_ = np.clip(0.3 * dep_, 0, smax_)
                ub_ = np.zeros_like(up)
                for i_ in range(len(sigs_) - 1):
                    a_, b2_ = sigs_[i_], sigs_[i_ + 1]
                    t_ = np.clip((sd_ - a_) / (b2_ - a_), 0, 1)[..., None]
                    m_ = ((sd_ >= a_) if i_ == 0 else (sd_ > a_)) & (sd_ <= b2_)
                    ub_ = np.where(m_[..., None], lv_[i_] * (1 - t_) + lv_[i_ + 1] * t_, ub_)
                up = ub_ * wx_[..., None]
        res = bg_hi + up
    if shadow and gen_sh:
        STATE["sh_use"] = gen_shadow(STATE["car_hi"], sh_dark, sh_soft, sh_contact)
    else:
        STATE["sh_use"] = STATE["sh_hi"]
    if shadow and STATE["sh_use"] is not None:
        sh_ = STATE["sh_use"]
        if True:                                       # all views (3-4 views showed the same light line)
            # AutoBG's shadow stops with a hard straight edge at the tyre's contact row; under a semi-clear tyre
            # reflection that step showed as a light line across tyre + reflection -> soften it vertically
            sh_ = sh_.copy()
            ys_ = np.where(STATE["car_hi"][..., 3] > 128)[0]
            sgy = max(2.0, 0.012 * (ys_.max() - ys_.min()))
            a0_ = sh_[..., 3].astype(np.float32)
            ab_ = cv2.GaussianBlur(a0_, (0, 0), sigmaX=1.0, sigmaY=sgy)
            # the softening only under the tyres: elsewhere (along the sills, bumpers) it thinned the shadow right at
            # the car's edge -> a light line of floor between car and shadow. There the shadow is never made lighter.
            tc_ = np.zeros(a0_.shape[1], np.float32)
            for w_ in STATE.get("wheels") or []:
                x0_, _, x1_, _ = [int(v) for v in w_["box"]]
                tc_[max(0, x0_):max(0, x1_)] = 1.0
            tc_ = cv2.GaussianBlur(tc_[None], (0, 0), 8)[0][None, :]
            sh_[..., 3] = np.clip(ab_ * tc_ + np.maximum(a0_, ab_) * (1 - tc_), 0, 255).astype(np.uint8)
            # AutoBG shadow: never dimmed; made darker by --shadow_dark (default 1.2 = 20 % stronger), all views
            sh_[..., 3] = np.clip(sh_[..., 3].astype(np.float32) * float(globals().get("SH_DARK", 1.2)), 0, 255).astype(np.uint8)
        under = G.over(res, sh_)                        # shadow on top of the reflection (as before)
        res = under
        if refl_front > 0:
            # reflection in front of the shadow: the shadow stays, but the reflection's DETAIL (tyre and
            # body shapes) that the shadow covered is put back on top, so it stays readable inside it.
            # (Replacing the shadow with the reflection removed the shadow completely on side views.)
            a_sh = STATE["sh_use"][..., 3:4].astype(np.float32) / 255
            sg = max(4.0, 0.012 * W2)
            detail = up - cv2.GaussianBlur(up, (0, 0), sg)
            res = under + refl_front * detail * a_sh
    if use_t:
        # reflection ON TOP of the AutoBG shadow: under it, the shadow's thin edge right at the tyre showed
        # as a light line between tyre and reflection
        # 11-16 (evening): same look as the front / back shots - plain matte column the tyre's own width, its round
        # bottom mirrored at the top, soft sides, long linear fade; at the side-shot strength (t_str)
        # (before: the real tyre + rim mirrored at 40 %, short fade:
        #  WH.flip(res, car_sharp, wheels, t_str, t_len, fade, t_gblur, 1, 1, 0.4 * t_det, t_shift, t_con, 0.4 * t_gloss, edge=0.012))
        res = WH.tyre_column(res, STATE["car_sharp"], STATE["wheels"], strength=t_str, ellipse=True)   # smooth stretched oval, no square shoulders
    elif STATE.get("frontrear"):
        # front / rear view: v2 put its tyre reflections beside the tyres -> those are removed (above) and the
        # real tyres are mirrored exactly under themselves, same look as the side views
        # the reflection is moved up until its top is as wide as the tyre (the round bottom of the tyre would
        # otherwise make it start narrow), and it runs further down than on side shots
        # this morning's version (real tyre mirrored with detail + gloss, moved up until the reflection is as wide
        # as the tyre, filled to full width above the round bottom), only STRETCHED VERTICALLY x1.5 (not a uniform
        # scale) with the fade lengthened to match
        wl_ = []
        ys_c = np.where(STATE["car_sharp"][..., 3] > 128)[0]
        bh_c = float(ys_c.max() - ys_c.min()) if len(ys_c) else 0.0
        for w_ in STATE["wheels"]:
            m_ = w_["m"].copy()
            wid = m_.sum(1)
            th_ = float(w_["yb"] - w_["top"] + 1)
            if wid.max() > 0:
                rb = np.where(wid >= 0.9 * wid.max())[0].max()
                cl = np.where(m_[rb])[0]
                r0 = np.where(m_.any(1))[0].min()
                m_[r0:rb + 1, cl.min():cl.max() + 1] = True
                # no bumps: the outline is replaced by its smooth convex shape (dents / notches in the tyre mask from
                # mudflaps, bumper fins, tow eyes made bumpy reflection edges)
                pts_ = cv2.findNonZero(m_.astype(np.uint8))
                if pts_ is not None and len(pts_) >= 3:
                    hm_ = np.zeros(m_.shape, np.uint8)
                    cv2.fillConvexPoly(hm_, cv2.convexHull(pts_), 1)
                    m_ = hm_.astype(bool)
                rows_ = np.where(wid >= 0.75 * wid.max())[0]
                lift = float(m_.shape[0] - 1 - rows_.max()) if len(rows_) else 0.0
                w_ = dict(w_, m=m_, full=m_, ma=cv2.GaussianBlur(m_.astype(np.float32), (0, 0), 1.5), lift=lift - 2.0,
                          vstretch=max(2.5, max(0.45 * th_, 0.20 * bh_c) / max(1.0, 0.85 * (m_.shape[0] - lift))),
                          flen=max(0.45 * th_, 0.20 * bh_c))   # never a stub on half-hidden tyres
                # tyre half hidden by the bumper: the mirrored tyre ran out before the fade ended -> a hard cut.
                # Now the stretch grows until the mirrored tyre is longer than the fade, so it always fades out first
            wl_.append(w_)
        # 11-16: replaced by the reference look - plain matte column exactly the tyre's width, round top corners,
        # soft sides, long linear fade (the mirrored-tyre version above is no longer drawn)
        res = WH.tyre_column(res, STATE["car_sharp"], STATE["wheels"])
    else:
        sl_ = STATE.get("slivers") or []
        full_ = [w_ for w_ in STATE.get("wheels", []) if not any(w_ is x_ for x_ in sl_)]
        if full_:
            # 3-4 views, visible tyres: v2 draws the reflection; add only the thin dark contact line where the
            # tyre meets the floor, so tyre and v2's reflection join without a light seam
            res = WH.flip(res, STATE["car_sharp"], full_, 0.001, t_len, fade, 0.0, 1.0, 1.0, 0.0, t_shift, t_con, 0.0)
        if sl_:
            # far tyre that is only a sliver: v2 draws no usable reflection there (an alloy or nothing), and
            # mirroring the sliver gives a flat slab. Treat it as an invisible tyre: a soft dark oval under it
            # that fades away - no rim, no shape detail
            # now: the real third tyre mirrored exactly under itself (v2's offset version is removed in clean)
            # same recipe as the approved front / back shots (filled to full width, 2.5x vertical stretch, faint)
            sl2_ = []
            for w_ in sl_:
                m_ = w_["m"].copy()
                wid = m_.sum(1)
                if wid.max() > 0:
                    rb = np.where(wid >= 0.9 * wid.max())[0].max()
                    cl = np.where(m_[rb])[0]
                    r0 = np.where(m_.any(1))[0].min()
                    m_[r0:rb + 1, cl.min():cl.max() + 1] = True
                    rows_ = np.where(wid >= 0.75 * wid.max())[0]
                    lift = float(m_.shape[0] - 1 - rows_.max()) if len(rows_) else 0.0
                    sl2_.append(dict(w_, m=m_, full=m_, ma=cv2.GaussianBlur(m_.astype(np.float32), (0, 0), 0.8),
                                     lift=lift - 2.0, vstretch=2.5, soft_sides=0.12))
            if sl2_:
                # darken-only: never a lighter patch inside the AutoBG shadow beside it; soft sides: no straight edges
                res = WH.flip(res, STATE["car_sharp"], sl2_, 0.25, 0.45, fade, t_gblur, 1.0, 1.0, t_det, t_shift, 0.3 * t_con, t_gloss, darken=True)

    # the cut-out's own crisp edge in ALL views: the tightened, softened edge let the floor show through at the
    # bottom of the tyres -> a light line between tyre and reflection
    res = put_car(res, STATE["car_sharp"], P["v2"], edge_w)

    return cv2.cvtColor(np.clip(res, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def save(*s):
    if "P" not in STATE:
        raise gr.Error("Generate first.")
    img = finish(*s)
    n = STATE["name"]
    Image.fromarray(img).save(os.path.join(A.out, f"{n}_final.jpg"), quality=95)
    STATE["v2"].save(os.path.join(A.out, f"{n}_gloss_grey_cond__v2.jpg"), quality=95)
    STATE["cond"].save(os.path.join(A.out, f"{n}_gloss_grey_cond.jpg"), quality=95)
    return f"saved {n}_final.jpg ({img.shape[1]}x{img.shape[0]}) in {A.out}"


# settings chosen on 2026-11-09 (Polestar CRB05Z test) - used by --batch
BATCH = dict(fast=False, light_w=1.15, steps=20, seed=42, bottom=0.88, width=0.55, height=0.55, do_recut=True,
             opacity=0.4, length=0.2, fade=1.0, even=0.6, hidden=0.4, lift=0, shadow=True, straight=1.0,
             detrend=0.0, edge_w=0.0, tone=0.1, shift_px=2, refl_front=0.0, gen_sh=False, sh_dark=0.8, sh_soft=1.0, sh_contact=0.9,
             mode="Auto", t_str=0.184, t_len=0.231, t_det=0.5, t_shift=4, t_con=1.0, t_gloss=0.65, t_gblur=0.0, refl_col=1.0, fstart=0.0, tstyle=True, tlen=0.45)

if A.batch:
    import glob
    out_dir = A.batch_out or A.batch.rstrip("/\\") + "_out"
    os.makedirs(out_dir, exist_ok=True)
    B = dict(BATCH)
    if A.shift is not None:
        B["shift_px"] = A.shift
    for kv in A.set:
        k, v = kv.split("=")
        B[k] = v if isinstance(B[k], str) else type(B[k])(float(v)) if not isinstance(B[k], bool) else v.lower() in ("1", "true", "yes")
    print("batch settings:", B, flush=True)
    if A.autobg_shadow:
        B["gen_sh"] = False
    if A.front is not None:
        B["refl_front"] = A.front
    cuts = [f for f in sorted(glob.glob(os.path.join(A.batch, "*.png"))) if "_shadow" not in os.path.basename(f).lower()]
    print(f"{len(cuts)} cars -> {out_dir}", flush=True)
    for i, cf in enumerate(cuts, 1):
        stem = os.path.splitext(cf)[0]
        sf = next((stem + x for x in ("_shadow.png", "_shadows.png") if os.path.exists(stem + x)), None)
        try:
            _, raw_out, msg = generate(cf, sf, None, B["bottom"], B["width"], B["height"], B["steps"], B["seed"],
                                       B["fast"], B["light_w"], None, B["do_recut"], B["mode"] == "Auto")
            img = finish(B["opacity"], B["length"], B["fade"], B["even"], B["hidden"], B["lift"], B["shadow"],
                         True, B["straight"], B["detrend"], B["edge_w"], B["tone"], B["shift_px"], B["refl_front"],
                         B["gen_sh"], B["sh_dark"], B["sh_soft"], B["sh_contact"], B["mode"], B["t_str"], B["t_len"], B["t_det"], B["t_shift"], B["t_con"], B["t_gloss"], B["t_gblur"], B["refl_col"], B["fstart"], B["tstyle"], B["tlen"])
            n = os.path.basename(stem)
            Image.fromarray(img).save(os.path.join(out_dir, f"{n}_final.jpg"), quality=95)
            raw_out.save(os.path.join(out_dir, f"{n}_v2raw.jpg"), quality=92)
            print(f"[{i}/{len(cuts)}] {n}: {msg}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(cuts)}] {os.path.basename(cf)} FAILED: {e}", flush=True)
    print("done ->", out_dir)
    raise SystemExit(0)

def run_cars(files, bg_file=None, progress=gr.Progress()):
    """team page: every uploaded car (+ its _shadow.png if uploaded) -> final image with the locked settings"""
    import zipfile, tempfile, shutil
    if not files:
        raise gr.Error("Upload the car PNGs (and their _shadow.png files).")
    paths = [f if isinstance(f, str) else f.name for f in files]
    def key(p_):                                       # "Car 01_Shadows.PNG" -> ("car01", True)
        n_ = os.path.splitext(os.path.basename(p_))[0].lower()
        is_sh = re.search(r"[\s_\-.]*shadows?$", n_) is not None
        return re.sub(r"[^a-z0-9]", "", re.sub(r"[\s_\-.]*shadows?$", "", n_)), is_sh
    shadows = {key(p_)[0]: p_ for p_ in paths if key(p_)[1]}
    cars = [p_ for p_ in paths if p_.lower().endswith(".png") and not key(p_)[1]]
    if not cars:
        raise gr.Error("No car PNGs found (files named ..._shadow.png are only the shadows).")
    out_dir = tempfile.mkdtemp(prefix="autobg_")
    B = dict(BATCH)
    done, gallery, notes = [], [], []
    for i, cf in enumerate(cars):
        stem = os.path.splitext(os.path.basename(cf))[0]
        progress(i / len(cars), desc=f"{stem} ({i + 1}/{len(cars)})")
        sf = shadows.get(key(cf)[0])
        try:
            bgp = (bg_file if isinstance(bg_file, str) else bg_file.name) if bg_file else None
            _, raw_v2, _ = generate(cf, sf, bgp, B["bottom"], B["width"], B["height"], B["steps"], B["seed"],
                     B["fast"], B["light_w"], None, B["do_recut"], B["mode"] == "Auto")
            img = finish(B["opacity"], B["length"], B["fade"], B["even"], B["hidden"], B["lift"], B["shadow"],
                         True, B["straight"], B["detrend"], B["edge_w"], B["tone"], B["shift_px"], B["refl_front"],
                         B["gen_sh"], B["sh_dark"], B["sh_soft"], B["sh_contact"], B["mode"], B["t_str"], B["t_len"],
                         B["t_det"], B["t_shift"], B["t_con"], B["t_gloss"], B["t_gblur"], B["refl_col"], B["fstart"], B["tstyle"], B["tlen"])
            op = os.path.join(out_dir, f"{stem}_reflection.jpg")
            Image.fromarray(img).save(op, quality=95)
            raw_v2.save(os.path.join(out_dir, f"{stem}_v2raw.jpg"), quality=90)     # for checking what v2 drew
            done.append(op); gallery.append((op, stem))
            if sf is None:
                notes.append(f"{stem}: no shadow file uploaded - made without AutoBG shadow")
        except Exception as e:
            notes.append(f"{stem}: FAILED ({e})")
    zp = os.path.join(out_dir, "autobg_reflections.zip")
    with zipfile.ZipFile(zp, "w") as z:
        for op in done:
            z.write(op, os.path.basename(op))
    msg = f"{len(done)} of {len(cars)} cars done, {sum(1 for c in cars if key(c)[0] in shadows)} with AutoBG shadow." + ("\n" + "\n".join(notes) if notes else "")
    return gallery, zp, msg


if A.team:
    with gr.Blocks(title="AutoBG reflections") as demo:
        gr.Markdown("## AutoBG reflections\nUpload the car cut-outs (PNG) **and** their `_shadow.png` files, then press **Generate**.  "
                    "About 40 seconds per car.")
        up = gr.File(label="Car PNGs + _shadow.png files", file_count="multiple", file_types=[".png"])
        bg_t = gr.File(label=f"Background (optional - empty = standard studio {os.path.basename(A.backdrop)})",
                       file_count="single", file_types=[".png", ".jpg", ".jpeg"])
        go_t = gr.Button("Generate", variant="primary")
        msg_t = gr.Textbox(label="Status", interactive=False)
        gal = gr.Gallery(label="Results", columns=3, height="auto")
        zip_t = gr.File(label="Download all (zip)")
        go_t.click(run_cars, [up, bg_t], [gal, zip_t, msg_t])
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0" if A.share else "127.0.0.1", server_port=A.port, share=A.share,
        auth=("autobg", A.password) if A.password else None)
    raise SystemExit(0)

with gr.Blocks(title="AutoBG reflection (v2, local)") as demo:
    gr.Markdown("### AutoBG reflection - v2 gloss, locked look - runs on this PC")
    with gr.Row():
        with gr.Column(scale=1):
            car_f = gr.File(label="Car cut-out (transparent PNG)", type="filepath")
            sh_f = gr.File(label="Same car with AutoBG shadow (_shadows.png)", type="filepath")
            photo_f = gr.File(label="Original photo (optional - best edges)", type="filepath")
            do_recut = gr.Checkbox(True, label="Clean the cut-out's lower edge with BiRefNet")
            bg_f = gr.File(label=f"Backdrop (empty = {os.path.basename(A.backdrop)})", type="filepath")
            with gr.Accordion("Placement", open=False):
                bottom = gr.Slider(0.7, 0.99, 0.95, step=0.01, label="Tyre line (fraction of height)")
                width = gr.Slider(0.3, 0.9, 0.55, step=0.01, label="Max car width")
                height = gr.Slider(0.3, 0.9, 0.55, step=0.01, label="Max car height")
            steps = gr.Slider(10, 40, 20, step=1, label="Model steps (30 = the locked results; 20 = faster)")
            seed = gr.Number(42, label="Seed")
            fast = gr.Checkbox(False, label="Fast mode (4 steps, Lightning) - grainier; unticked = full 30-step look")
            light_w = gr.Slider(0.5, 1.4, 1.15, step=0.05, label="Lightning strength (fast mode)")
            go = gr.Button("Generate", variant="primary")
            info = gr.Textbox(label="Time", interactive=False)
            gr.Markdown("**Finish** (live)")
            mode = gr.Radio(["Auto", "v2 reflection", "Tyre reflection"], value="Auto",
                            label="Reflection: Auto = tyre reflection for side views, v2 for front / rear / 3-4")
            t_str = gr.Slider(0, 1, 0.3, step=0.05, label="Tyre reflection strength (side views)")
            t_len = gr.Slider(0.1, 1.5, 0.22, step=0.05, label="Tyre reflection fade length (x wheel height)")
            t_det = gr.Slider(0, 0.5, 0.5, step=0.05, label="Real tyre detail in the tyre reflection (fades out halfway)")
            t_shift = gr.Slider(-10, 10, 4, step=1, label="Tyre reflection up (+) / down (-), px")
            t_con = gr.Slider(0, 1, 1.0, step=0.05, label="Tyre contact darkness (thin line where tyre meets floor)")
            t_gloss = gr.Slider(0, 1, 0.65, step=0.05, label="Tyre reflection gloss (0 = matte dark shape, 1 = mirror of the real tyre + rim)")
            t_gblur = gr.Slider(0, 6, 0.0, step=0.5, label="Gloss softness (0 = sharp mirror, higher = satin)")
            with gr.Row():
                p_raw = gr.Button("Start from raw v2 gloss")
                p_lock = gr.Button("Start from locked look")
            opacity = gr.Slider(0, 1, 0.4, step=0.05, label="Opacity")
            refl_col = gr.Slider(0, 3, 1.0, step=0.1, label="Reflection colour (1 = as v2, higher = more of the car's colour)")
            length = gr.Slider(0.1, 4, 0.2, step=0.05, label="Length (x car height; 4 = no fade)")
            fade = gr.Slider(0.3, 3, 1.0, step=0.1, label="Fade curve (1 = even)")
            straight = gr.Slider(0, 1, 1.0, step=0.1, label="Fade (0 = follows wheels / curved, 1 = perspective: away from the car's footprint lines)")
            fstart = gr.Slider(0, 0.4, 0.0, step=0.02, label="Fade start above the tyres (x car height; 0 = full reflection at the tyres, higher = tyre reflections fade too)")
            tstyle = gr.Checkbox(True, label="Fade like the side shots: straight down from each tyre's contact, same for every tyre")
            tlen = gr.Slider(0.1, 2, 0.45, step=0.05, label="Fade length when fading like side shots (x wheel height)")
            refl_front = gr.Slider(0, 1, 0.0, step=0.05, label="Reflection in front of the AutoBG shadow (0 = shadow on top)")
            shift_px = gr.Slider(-40, 40, 2, step=1, label="Move reflection up (+) / down (-), pixels of the final image")
            tone = gr.Slider(0, 1, 0.1, step=0.05, label="Darken light parts of the reflection under the car (white bodies)")
            edge_w = gr.Slider(0, 3, 0.0, step=0.25, label="Car-floor join drawn by v2 (0 = off, 1 = thin band)")
            detrend = gr.Slider(0, 1, 0.0, step=0.1, label="Remove v2's floor tone (1 = only the car's reflection, no glow)")
            even = gr.Slider(0, 1, 0.6, step=0.05, label="Even out")
            hidden = gr.Slider(0, 1, 0.1, step=0.05, label="Hidden wheels (0 = faint outline)")
            lift = gr.Slider(0, 255, 0, step=1, label="Max brightening under the car")
            shadow = gr.Checkbox(True, label="Shadow")
            gen_sh = gr.Checkbox(False, label="Generated shadow (unticked = AutoBG shadow file)")
            sh_dark = gr.Slider(0, 1, 0.8, step=0.05, label="Shadow darkness")
            sh_soft = gr.Slider(0.3, 3, 1.0, step=0.1, label="Shadow softness / spread")
            sh_contact = gr.Slider(0, 1, 0.9, step=0.05, label="Tyre contact darkness")
            origcar = gr.Checkbox(True, label="(always on now: original car, full resolution)", visible=False)
            sv = gr.Button("Save")
            msg = gr.Textbox(label="", interactive=False)
        with gr.Column(scale=2):
            final = gr.Image(label="Final (locked look)", type="numpy")
            with gr.Row():
                cond_im = gr.Image(label="Model input", type="pil")
                raw = gr.Image(label="Raw v2 gloss output", type="pil")
    fin = [opacity, length, fade, even, hidden, lift, shadow, origcar, straight, detrend, edge_w, tone, shift_px, refl_front, gen_sh, sh_dark, sh_soft, sh_contact, mode, t_str, t_len, t_det, t_shift, t_con, t_gloss, t_gblur, refl_col, fstart, tstyle, tlen]
    go.click(generate, [car_f, sh_f, bg_f, bottom, width, height, steps, seed, fast, light_w, photo_f, do_recut], [cond_im, raw, info]) \
      .then(finish, fin, final)
    for c in fin:
        c.change(finish, fin, final)
    # presets: [opacity, length, fade, even, hidden, lift, straight, detrend]
    pv = [opacity, length, fade, even, hidden, lift, straight, detrend]
    p_raw.click(lambda: [1.0, 4.0, 1.0, 0.0, 1.0, 255, 0.0, 0.0], None, pv).then(finish, fin, final)
    p_lock.click(lambda: [0.6, 0.5, 1.0, 0.6, 0.3, 10, 1.0, 1.0], None, pv).then(finish, fin, final)
    sv.click(save, fin, msg)

demo.launch(server_name="0.0.0.0" if A.share else "127.0.0.1", server_port=A.port,
            inbrowser=not A.share, share=A.share)
