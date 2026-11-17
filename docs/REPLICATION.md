# AutoBG Reflections — replication guide

How the current AutoBG floor-reflection results are produced, and how to rebuild every part
from scratch. Status: **final settings locked 2026-11-13**, with front/back strength and
shadow tweaks applied on 2026-11-16.

---

## 1. Overview

The system takes an **AutoBG car cut-out** (transparent PNG) and its **AutoBG shadow** (`<car>_shadow.png`),
places the car on the studio backdrop `floor_satin_light.png`, and adds a floor reflection.

Three models are used:

| # | Model | Role | Where it comes from |
|---|---|---|---|
| 1 | **Qwen-Image-Edit-2511** + **v2 reflection LoRA** (`v2.safetensors`) | Draws the floor reflection of the whole car (used on 3/4 and front/back shots) | Base: Hugging Face `Qwen/Qwen-Image-Edit-2511`. LoRA: trained by us on Blender renders (section 4) |
| 2 | **Tyre model** (`tyres.pt`, YOLO11s-seg) | Finds every tyre, with its exact outline | Trained by us on 300 labelled AutoBG cut-outs (section 5) |
| 3 | **BiRefNet** (`ZhengPeng7/BiRefNet`) | Re-cuts the car edge cleanly before processing | Hugging Face, used as-is |

Helper models used only for labelling: **OWLv2** (`google/owlv2-base-patch16-ensemble`) and
**SAM** (`facebook/sam-vit-huge`).

The final image is a **hybrid**: the AI reflection is used where it looks good, and exact mirrored
tyres are used where the AI was unreliable (tyre positions, alloys on hidden tyres). Which recipe
runs depends on the camera view, detected automatically.

```
cut-out + AutoBG shadow
      │
      ├─ BiRefNet re-cut ─► place on backdrop (bottom 95 %, max 55 % of width/height)
      │
      ├─ tyre model (YOLO) ─► tyres + outlines ─► view: SIDE / FRONT-BACK / 3-4
      │
      ├─ Qwen-Image-Edit-2511 + v2 LoRA (20 steps, CFG 4)  ── skipped on side views
      │        └─ align to the real car (ECC) ─► keep only the floor change (the reflection)
      │
      ├─ per-view finishing (section 3)
      ├─ AutoBG shadow on top (original strength × 1.25, never dimmed)
      └─ original car on top, full resolution  ─►  <car>_reflection.jpg
```

---

## 2. Running it

### 2.1 Hardware

- **Production / team page:** Vast.ai, **RTX PRO 6000 (96 GB)**, PyTorch template, single GPU,
  about $1.5/hr. Any 48 GB+ GPU works with `--full`.
- **16 GB PC:** works without `--full` (compressed GGUF model, `--quant Q4_K_M`, or `Q3_K_M` if memory is short). Slower.
- Roughly 20–30 s per image at 20 steps on the RTX PRO 6000 (30 steps measured ~35 s). Side shots skip the AI model and take about 1 s.

### 2.2 Files you need (not in git)

| File | Where it lives | Put it at (Vast) |
|---|---|---|
| `v2.safetensors` (reflection LoRA, ~590 MB) | Google Drive (ask Manu for the file id); also `D:\reflections\models\autobg_reflection_v2.safetensors` | `/workspace/v2.safetensors` |
| `tyres.pt` (tyre model, ~20 MB) | `D:\reflections\models\tyres.pt` | `/workspace/Autobg-Reflections/app/models/tyres.pt` |
| `floor_satin_light.png` (backdrop) | `D:\reflections\floor_satin_light.png` | `/workspace/floor_satin_light.png` |

Qwen-Image-Edit-2511, BiRefNet and the YOLO base weights download automatically on first run.

### 2.3 Setup and start (Vast)

```bash
cd /workspace && git clone https://github.com/bist47-debug/Autobg-Reflections.git
V2_DRIVE_ID=<file id> bash Autobg-Reflections/vast/setup.sh
# upload tyres.pt and floor_satin_light.png (table above)
bash Autobg-Reflections/vast/start_team.sh <password>
```

Open the `https://….gradio.live` link it prints. Log in with user **`autobg`** and the password.
The first start downloads Qwen-Image-Edit-2511 (tens of GB), which takes several minutes.

### 2.4 Three ways to run `app/reflect_autobg.py`

| Mode | Command (from `app/`) | Use |
|---|---|---|
| Team page | `python reflect_autobg.py --full --share --team --password XXXX --port 7870 --lora /workspace/v2.safetensors --backdrop /workspace/floor_satin_light.png` | Upload cars + shadows, get results. Locked settings, no sliders. |
| Full page | same without `--team --password` | All sliders, for tuning. |
| Batch | `python reflect_autobg.py --full --lora … --backdrop … --batch FOLDER` | Every `name.png` + `name_shadow.png` in a folder → `FOLDER_out/`. |

Useful flags: `--shadow_dark 1.25` (AutoBG shadow strength multiplier; 1 = original),
`--bright 25` (how much lighter than the floor the reflection of white/silver cars may be).

### 2.5 Inputs

- `name.png`: AutoBG cut-out, transparent background.
- `name_shadow.png`: the AutoBG shadow file, the same size as the cut-out. The team page pairs files by name.
- **Check the shadow file:** some exported `_shadow.png` files turned out to be copies of the car
  (no shadow). Then no shadow is drawn.

### 2.6 Outputs

- `<name>_reflection.jpg`: the final image, backdrop size.
- `<name>_v2raw.jpg`: the AI model's raw output. Send this when reporting a problem shot.
- Team-page results are kept in `/tmp/autobg_*` on the instance. Collect them with
  `bash vast/collect_team_results.sh` **before destroying the instance**.

---

## 3. Locked settings, per view

Defaults live in `BATCH` in `app/reflect_autobg.py` (team page and batch) and in `finish()`.

### 3.1 View detection

| View | Rule (`is_side`, `is_frontrear`) |
|---|---|
| Side | Car width > 1.7 × height, two round tyres (w/h 0.75–1.35) more than half the car width apart, on the same ground line |
| Front/back | Car width < 1.8 × height, two lowest tyres at the left and right ends on the same line; **or** the car outline is mirror-symmetric ≥ 0.88 (then a missing tyre is mirrored from the found one) |
| 3/4 | Everything else |

### 3.2 Side shots: v2 AI reflection by default since 2026-11-17 (same as 3/4); `--side_tyres` gives the tyre-only version below

- `WH.tyre_column(..., ellipse=True, length=0.68, up=5)`: each tyre's round bottom mirrored and stretched
  into a smooth oval (full tyre width, no square shoulders), soft sides, plain matte tone, no rim detail.
- Linear fade over **0.443 × tyre width**, oval stretch **0.5**, strength **0.164**, pushed up 8 px; the real tyre + rim mirrored inside it at **4.25 %** (11-17), strength **0.193**. Both tyres get the same fade length and tone.
- Nothing between the tyres (the sill mirror was removed 2026-11-16).

### 3.3 Front/back shots: AI reflection of the body + tyre columns (frozen 2026-11-16)

- The AI's own tyre reflections are removed (it places them wrongly on these views).
- `WH.tyre_column(..., strength=0.236, mirror=False)`: each tyre's reflection is a plain matte column
  **exactly that tyre's width at the floor**, top corners = the tyre's round corners mirrored (one shared,
  symmetric corner shape for all tyres), soft straight sides, **pushed up 8 px**.
- Linear fade over **0.75 × tyre width** (11-17; same length and tone for every tyre), always ending inside the frame.
- The dark AutoBG shadow right under the tyres can hide the top of the reflection; if it looks too low,
  lengthen the fade, never weaken the shadow.

### 3.4 3/4 shots: AI reflection (this morning's lock, frozen 2026-11-16)

- AI reflection opacity **0.42** (0.4 + 5 %), fade **0.45 × wheel height** ("fade like side shots": straight
  down from each tyre's contact, the same for every tyre), even-out 0.6, reflection colour 1, moved 2 px.
- Hidden wheels (not visible in the photo): the AI's reflection is kept at **40 %**, with the bright alloy removed.
- **Third tyre** (hardly visible, beside the two main ones): the AI's version is removed and the real
  tyre is mirrored like on front/back shots, at strength **0.2625**.
- No bumper-edge softening (tried 11-16, reverted).
- Light cars (white/silver): the reflection may be up to 25 levels lighter than the floor (`--bright 25`), soft-capped.

### 3.5 Shadows (all views)

- **AutoBG shadows are never dimmed** (decision 2026-11-13). They are drawn at **1.25 × the original strength**
  (`--shadow_dark 1.25`), with a slight vertical softening of the hard edge at the tyre contact.
- The shadow goes on top of the reflection. If a dark shadow hides the reflection, lengthen the
  fade; never weaken the shadow.

### 3.6 Model inference settings

- Prompt: `Add a realistic floor reflection of the car on a glossy grey showroom floor.`
- 20 steps, `true_cfg_scale` 4.0, seed 42, negative prompt `" "`.
- Working size 1024 px wide; the car is placed with the tyre line at 88 % of the frame height (every reflection fades out inside the frame), max 55 % of the width/height.
- The AI image is aligned back onto the real car (`align_v2`, OpenCV ECC affine); only the **floor change** is used, upscaled to full resolution.
- Lightning 4-step LoRA (`lightx2v/Qwen-Image-Edit-2511-Lightning`) is available as "fast" mode (~3 s),
  but it was judged too grainy and is **off** in the locked setup.

---

## 4. Reflection model (v2 LoRA): how it was trained

### 4.1 Data: synthetic Blender renders

1. **Car models:** about 160 cars from **Objaverse** (`objaverse` Python package), filtered to clean,
   correctly scaled cars (a `keep_list.csv` of uids). Licences per model are stored in each render's `meta.json`.
2. **Rendering:** `training/render_cars.py` (Blender, Cycles, GPU). Example (view count and camera range were varied between runs):
   ```bash
   blender -b -P training/render_cars.py -- --models keep_list.csv --glb_dir glb/ --out renders_v2 \
     --v2 --gpu --views 24 --res 1024 --samples 48 --elev 3,22
   ```
   For each car and camera view (`--v2` mode) it renders:
   - the car cut-out (`input_rgba.png`, like an AutoBG cut-out);
   - pure-mirror floor passes **with and without the car**, under a bright and a dark studio.

   The reflection layer is the difference between the with-car and without-car passes.
   Renders are kept in `D:\reflections\training_renders\` (`renders`, `renders_v2`, `renders_v2_angles`, …).

### 4.2 Training pairs

- **v1** (`training/prep_dataset_v1.py`): four floor looks (mirror dark, gloss grey, satin light,
  polished concrete), one caption per look. 4000 steps.
- **v2** (`training/make_pairs_v2.py`), the model in use:
  ```bash
  python training/make_pairs_v2.py renders_v2 backdrops/ reflection_v2 --per_view 2 --holdout 6 --synthetic 24
  ```
  - `control` = backdrop + soft AutoBG-style shadow + car (model input).
  - `target` = backdrop + physically correct reflection + the same shadow + car (model output).
  - The backdrops are real AutoBG studio backdrops **kept unchanged**, plus 24 synthetic plain,
    gradient and tiled studios, so the model learns to change only the floor reflection.
  - Caption: `Add a realistic floor reflection of the car. Keep the floor, background and car unchanged.`
  - 6 held-out cars are never trained on (used as samples).

### 4.3 Training

- Trainer: **ostris/ai-toolkit** (`python run.py train_reflection_v2.yaml`), on a rented 48 GB+ GPU (Vast/RunPod).
- Base: `Qwen/Qwen-Image-Edit-2511`, arch `qwen_image_edit_plus`, quantized qfloat8 (transformer and text encoder).
- LoRA rank/alpha **32/32**, AdamW8bit, lr **1e-4**, batch 1, bf16, flowmatch, weighted timesteps,
  resolution 768–1024, caption dropout 0.05, cached text embeddings.
- v1: 4000 steps from scratch. **v2: 2500 steps, continued from the v1 LoRA** (`pretrained_lora_path`).
  Checkpoints every 500 steps; the chosen checkpoint is `v2.safetensors`.

### 4.4 What did not work (don't repeat)

- **v3** (reflection + shadow in one model): results were far off. v2 + AutoBG shadows was better.
- **v4** (Autofox-style contact shadow): paused.
- Satin / mirror prompts: satin blurs too much; mirror darkens the whole floor. The gloss-grey prompt is used.
- Generated shadows: worse than AutoBG's. AutoBG shadows are used everywhere.
- Lightning 4-step: fast, but grainy.

---

## 5. Tyre model: how it was trained

1. **Images:** 300 AutoBG cut-outs, a mix of side, front/back and 3/4 views.
2. **Labelling** (`app/tyre_label.py`, on a local GPU):
   ```bash
   python tyre_label.py --images tyre_data/images
   ```
   Each car opens pre-marked by OWLv2 + SAM. Click an unmarked tyre to add it (SAM outline), or click
   a wrong mark to remove it. Mark every tyre you can see, including far and half-hidden ones,
   but never the spare wheel. Labels are saved in YOLO segmentation format in `tyre_data/ds/`.
3. **Training** (`app/train_tyres.py`):
   ```bash
   python -m pip install -U ultralytics
   python train_tyres.py
   ```
   - 90/10 train/val split.
   - Base model: `yolo11s-seg.pt`; `imgsz=1024`, 120 epochs, batch 6, patience 30.
   - About 20–40 min on an RTX 5060 Ti. Output: `models/tyres.pt`.
   - Result: mask precision 1.0, recall 0.983, **mAP50 0.985**, mAP50-95 0.917.
4. **At inference** (`app/wheels.py`): the image is cropped to the car first, the model runs at 1024 px,
   and again at 1280 px if fewer than 2 tyres are found. Confidence 0.35. Flat strips under the
   bumper (tow eyes, diffusers) are dropped.
5. **Quick check without the GPU model:** `python wheel_test.py FOLDER` writes `debug/wheels_test.jpg`
   (tyre reflections plus what was found).

---

## 6. Testing and handover checklist

- Test set: `D:\reflections\test_cars\` (`test20`: 20 cars with shadows; `batch2_100`: about 100 cars, including the Toyotas and the white Volvo).
- Check every new setting on at least one side shot, one front/back shot with fully visible tyres,
  one with half-hidden tyres (Audi/Mercedes), one 3/4 shot of a dark car, and one of a white car.
- Report problems with both `<name>_reflection.jpg` and `<name>_v2raw.jpg`.
- **Destroy the Vast instance after every session.**

## 7. Licences (check before commercial use)

| Component | Licence |
|---|---|
| Qwen-Image-Edit-2511 | Apache-2.0 |
| Ultralytics YOLO11 (tyre model) | AGPL-3.0. Commercial/SaaS use needs an Ultralytics Enterprise licence |
| BiRefNet | MIT |
| OWLv2, SAM (labelling only) | Apache-2.0 |
| Objaverse car models | Per model (mostly Creative Commons), in `meta.json` |
| ai-toolkit | MIT |

Confirm each on its own page; this table is a starting point, not legal advice.
