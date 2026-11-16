"""
make_pairs_v2.py - v2 training set: real backdrops kept unchanged + a physically correct reflection.

    python make_pairs_v2.py RENDERS_DIR BACKDROPS_DIR OUT_DIR [--per_view 2] [--holdout 6]

RENDERS_DIR   output of  render_cars.py --v2  (input_rgba + floorcar/floorempty passes per view)
BACKDROPS_DIR your AutoBG backdrops (jpg/png). Plain, gradient and tiled studios are generated
              on top of these, so the model also sees floors it will meet later.

For every view and backdrop:
  control = backdrop + (soft AutoBG-style shadow) + car                 -> model input
  target  = backdrop + reflection + (same shadow) + car                 -> model output
The reflection is the difference between the mirror-floor passes with and without the car, using
the bright studio for light backdrops (the car darkens the floor) and the dark studio for dark ones.
Writes OUT_DIR/target, OUT_DIR/control (same file names), captions, held-out samples and
train_reflection_v2.yaml (continues from the v1 LoRA).
"""
import argparse, glob, os, random
import numpy as np
import cv2

CAPTION = "Add a realistic floor reflection of the car. Keep the floor, background and car unchanged."
W, H = 1184, 896                                  # the model's working size (4:3, ~1 MP)


def floor_line(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h = g.shape[0]
    rows = cv2.GaussianBlur(np.median(g, axis=1).reshape(1, -1), (0, 0), 2)[0]
    d = np.abs(np.diff(rows))
    lo, hi = int(0.3 * h), int(0.9 * h)
    return (lo + int(np.argmax(d[lo:hi]))) / h


def synthetic_backdrops(rng, n):
    """Plain / gradient / tiled studios at many brightness levels, 1600x1200."""
    out = []
    for i in range(n):
        h, w = 1200, 1600
        seam = rng.uniform(0.45, 0.65)
        wall = np.array([rng.uniform(0.15, 0.97)] * 3) * 255 * np.array([1, rng.uniform(0.97, 1.0), rng.uniform(0.94, 1.0)])
        floor = np.clip(wall * rng.uniform(0.55, 1.05), 0, 255)
        img = np.zeros((h, w, 3), np.float32)
        ys = np.arange(h)[:, None]
        sy = int(seam * h)
        img[:sy] = wall * (0.9 + 0.1 * (ys[:sy] / sy))[..., None] if rng.random() < 0.5 else wall
        grad = (1.0 - 0.15 * (ys[sy:] - sy) / (h - sy))[..., None] if rng.random() < 0.6 else 1.0
        img[sy:] = floor * grad
        if rng.random() < 0.5:                      # tiled floor in perspective
            line = floor * rng.uniform(0.82, 0.95)
            vp = (w / 2 + rng.uniform(-300, 300), sy - rng.uniform(150, 600))
            for k in range(-12, 13):
                x_bottom = w / 2 + k * rng.uniform(180, 260)
                cv2.line(img, (int(vp[0] + (x_bottom - vp[0]) * (sy - vp[1]) / (h - vp[1]) * 0 + (x_bottom - vp[0]) * ((sy - vp[1]) / (h - vp[1])) + vp[0] - vp[0]), sy),
                         (int(x_bottom), h), line.tolist(), 2, cv2.LINE_AA)
            y = sy
            step = 18.0
            while y < h:
                cv2.line(img, (0, int(y)), (w, int(y)), line.tolist(), 2, cv2.LINE_AA)
                step *= 1.35
                y += step
        img += np.random.default_rng(i).normal(0, 1.2, img.shape)
        out.append((f"synth{i:02d}", np.clip(img, 0, 255).astype(np.uint8)))
    return out


def fit_backdrop(bd, seam_frac, seam_y, rng):
    """Scale/crop the backdrop to W x H so its wall/floor line lands at seam_y."""
    bh, bw = bd.shape[:2]
    for _ in range(20):
        sc = max(W / bw, H / bh) * rng.uniform(1.0, 1.35)
        sh, sw = int(bh * sc), int(bw * sc)
        oy = seam_frac * sh - seam_y                 # crop top
        if oy < 0:
            continue
        if oy + H <= sh:
            ox = rng.uniform(0, sw - W)
            M = np.array([[sc, 0, -ox], [0, sc, -oy]], np.float32)
            return cv2.warpAffine(bd, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
    # backdrop too short below the floor line: extend it by mirroring the floor
    sc = max(W / bw, H / bh)
    M = np.array([[sc, 0, -(bw * sc - W) / 2], [0, sc, -(seam_frac * bh * sc - seam_y)]], np.float32)
    return cv2.warpAffine(bd, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)


def soft_shadow(alpha, rng):
    """A soft dark footprint under the car, like the AutoBG shadow (multiplier 0..1)."""
    a = (alpha > 127).astype(np.uint8)
    ys = np.where(a.any(1))[0]
    y0, y1 = ys.min(), ys.max()
    band = np.zeros_like(a)
    cut = int(y1 - 0.18 * (y1 - y0))
    band[cut:y1 + 1] = a[cut:y1 + 1]
    band = cv2.dilate(band, np.ones((5, 31), np.uint8))
    sh = cv2.GaussianBlur(band.astype(np.float32), (0, 0), rng.uniform(8, 18))
    return 1.0 - rng.uniform(0.25, 0.45) * np.clip(sh / max(sh.max(), 1e-3), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("renders"); ap.add_argument("backdrops"); ap.add_argument("out")
    ap.add_argument("--per_view", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=6)
    ap.add_argument("--synthetic", type=int, default=24)
    ap.add_argument("--v1_lora", default="/workspace/lora_v1.safetensors")
    a = ap.parse_args()
    rng = random.Random(11)
    bds = []
    for p in sorted(glob.glob(os.path.join(a.backdrops, "*"))):
        im = cv2.imread(p)
        if im is not None:
            bds.append((os.path.splitext(os.path.basename(p))[0].replace(" ", "_"), cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    print(len(bds), "backdrops +", a.synthetic, "generated")
    bds += synthetic_backdrops(rng, a.synthetic)
    info = []
    for name, im in bds:
        s = floor_line(im)
        fl = im[int(s * im.shape[0]):].mean() / 255
        info.append((name, im, s, "light" if fl > 0.42 else "dark"))

    views = sorted(v for v in glob.glob(os.path.join(a.renders, "*", "v*")) if os.path.exists(os.path.join(v, "floorcar_light.png")))
    cars = sorted({os.path.basename(os.path.dirname(v)) for v in views})
    rng.shuffle(cars)
    hold = set(cars[:a.holdout])
    for d in ("target", "control", "samples"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    n, samples = 0, []
    for vi, v in enumerate(views):
        car_id = os.path.basename(os.path.dirname(v))
        rgba = cv2.imread(os.path.join(v, "input_rgba.png"), cv2.IMREAD_UNCHANGED)
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
        alpha = rgba[..., 3]
        if (alpha > 127).sum() < 500:
            continue
        A = alpha.astype(np.float32)[..., None] / 255
        deltas = {}
        for env in ("light", "dark"):
            fc = cv2.imread(os.path.join(v, f"floorcar_{env}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            fe = cv2.imread(os.path.join(v, f"floorempty_{env}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            scale = 65535.0 if fc.max() > 255 else 255.0
            d = cv2.cvtColor((fc - fe) / scale, cv2.COLOR_BGR2RGB)
            deltas[env] = cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
        # where the floor line should go: just above the car's highest ground contact
        car = alpha > 127
        ys = np.where(car.any(1))[0]
        cols = np.where(car.any(0))[0]
        bottoms = np.array([np.where(car[:, x])[0].max() for x in cols])
        contact_top = np.percentile(bottoms, 8)
        for k in range(a.per_view):
            name, bd, seam, env = info[rng.randrange(len(info))]
            seam_y = contact_top - rng.uniform(0.04, 0.12) * (ys.max() - ys.min())
            seam_y = float(np.clip(seam_y, 0.2 * H, 0.75 * H))
            bg = fit_backdrop(bd, seam, seam_y, rng).astype(np.float32) / 255
            strength = rng.uniform(0.45, 0.7)          # floor glossiness
            refl = deltas[env] * strength
            refl[: int(seam_y)] = 0                     # reflections only on the floor
            shadow = soft_shadow(alpha, rng)[..., None] if rng.random() < 0.8 else 1.0
            car_rgb = rgba[..., :3].astype(np.float32) / 255
            ctrl = (bg * shadow) * (1 - A) + car_rgb * A
            tgt = (np.clip(bg + refl, 0, 1) * shadow) * (1 - A) + car_rgb * A
            key = f"{car_id}_{os.path.basename(v)}_{name}_{k}"
            if car_id in hold:
                if len(samples) < 12 and k == 0:
                    cv2.imwrite(os.path.join(a.out, "samples", key + "_cond.jpg"), cv2.cvtColor((ctrl * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(a.out, "samples", key + "_target.jpg"), cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    samples.append(key)
                continue
            cv2.imwrite(os.path.join(a.out, "control", key + ".jpg"), cv2.cvtColor((ctrl * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, "target", key + ".jpg"), cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(os.path.join(a.out, "target", key + ".txt"), "w", encoding="utf-8").write(CAPTION)
            n += 1
        if (vi + 1) % 100 == 0:
            print(f"{vi + 1}/{len(views)} views, {n} pairs", flush=True)
    print(f"training pairs: {n}, test samples: {len(samples)}")

    pod = "/workspace/reflection_v2"
    sample_yaml = "".join(f'          - prompt: "{CAPTION}"\n            ctrl_img_1: "{pod}/samples/{k}_cond.jpg"\n' for k in samples[:6])
    yaml = f"""---
job: extension
config:
  name: "autobg_reflection_v2"
  process:
    - type: "diffusion_trainer"
      training_folder: "/workspace/output"
      device: cuda:0
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        pretrained_lora_path: "{a.v1_lora}"
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 6
      datasets:
        - folder_path: "{pod}/target"
          control_path:
            - "{pod}/control"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [ 768, 1024 ]
      train:
        batch_size: 1
        cache_text_embeddings: true
        steps: 2500
        gradient_accumulation: 1
        timestep_type: "weighted"
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
      model:
        name_or_path: "Qwen/Qwen-Image-Edit-2511"
        arch: "qwen_image_edit_plus"
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
        model_kwargs:
          match_target_res: false
      sample:
        sampler: "flowmatch"
        sample_every: 500
        width: 1184
        height: 896
        samples:
{sample_yaml}        neg: ""
        seed: 42
        walk_seed: false
        guidance_scale: 3
        sample_steps: 25
meta:
  name: "[name]"
  version: "2.0"
"""
    open(os.path.join(a.out, "train_reflection_v2.yaml"), "w", encoding="utf-8").write(yaml)
    print("wrote", os.path.join(a.out, "train_reflection_v2.yaml"))


if __name__ == "__main__":
    main()
