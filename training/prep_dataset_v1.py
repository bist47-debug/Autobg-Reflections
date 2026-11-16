"""
prep_dataset.py - package pairs_train for LoRA training with ai-toolkit (Qwen-Image-Edit-2511).

    python prep_dataset.py pairs_train reflection_dataset

Makes:
  reflection_dataset/target/<key>.jpg + <key>.txt   car WITH reflection + instruction caption
  reflection_dataset/control/<key>.jpg              same car WITHOUT reflection (same file name)
  reflection_dataset/samples/                       a few held-out cars, never trained on
  reflection_dataset/train_reflection.yaml          ai-toolkit config, paths set for RunPod
Then zip the reflection_dataset folder and upload it.
"""
import glob, os, random, shutil, sys

src, out = sys.argv[1], sys.argv[2]
POD_DIR = "/workspace/reflection_dataset"          # where it will live on RunPod
HOLDOUT_CARS = 6

CAPTIONS = {
    "mirror_dark": "Add a realistic floor reflection of the car on a dark mirror-polished studio floor.",
    "gloss_grey": "Add a realistic floor reflection of the car on a glossy grey showroom floor.",
    "satin_light": "Add a soft, blurred floor reflection of the car on a light satin studio floor.",
    "polished_concrete": "Add a faint, diffuse floor reflection of the car on a polished concrete floor.",
}

conds = sorted(glob.glob(os.path.join(src, "*_cond.jpg")))
pairs = []
for c in conds:
    key = os.path.basename(c)[:-9]
    t = os.path.join(src, key + "_target.jpg")
    look = next((l for l in CAPTIONS if key.endswith("_" + l)), None)
    if os.path.exists(t) and look:
        car = key.split("_v")[0]
        pairs.append((key, car, look, c, t))
cars = sorted({p[1] for p in pairs})
random.Random(7).shuffle(cars)
hold = set(cars[:HOLDOUT_CARS])
print(f"{len(pairs)} pairs from {len(cars)} cars; {HOLDOUT_CARS} cars held out for testing")

for d in ("target", "control", "samples"):
    os.makedirs(os.path.join(out, d), exist_ok=True)
n = 0
samples = []
for key, car, look, c, t in pairs:
    if car in hold:
        # keep a few held-out views (one per look) as test inputs
        if sum(1 for s in samples if s[1] == look) < 2:
            shutil.copy(c, os.path.join(out, "samples", key + "_cond.jpg"))
            shutil.copy(t, os.path.join(out, "samples", key + "_target.jpg"))
            samples.append((key, look))
        continue
    shutil.copy(t, os.path.join(out, "target", key + ".jpg"))
    shutil.copy(c, os.path.join(out, "control", key + ".jpg"))
    open(os.path.join(out, "target", key + ".txt"), "w", encoding="utf-8").write(CAPTIONS[look])
    n += 1
    if n % 500 == 0:
        print(n, "copied", flush=True)
print(f"training pairs: {n}, test samples: {len(samples)}")

sample_yaml = "".join(
    f'          - prompt: "{CAPTIONS[look]}"\n'
    f'            ctrl_img_1: "{POD_DIR}/samples/{key}_cond.jpg"\n'
    for key, look in samples[:6])

yaml = f"""---
job: extension
config:
  name: "autobg_reflection_v1"
  process:
    - type: "diffusion_trainer"
      training_folder: "/workspace/output"
      device: cuda:0
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 6
      datasets:
        - folder_path: "{POD_DIR}/target"
          control_path:
            - "{POD_DIR}/control"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [ 768, 1024 ]
      train:
        batch_size: 1
        cache_text_embeddings: true
        steps: 4000
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
        width: 1024
        height: 768
        samples:
{sample_yaml}        neg: ""
        seed: 42
        walk_seed: false
        guidance_scale: 3
        sample_steps: 25
meta:
  name: "[name]"
  version: "1.0"
"""
open(os.path.join(out, "train_reflection.yaml"), "w", encoding="utf-8").write(yaml)
print("wrote", os.path.join(out, "train_reflection.yaml"))
print("Next: right-click the folder -> Send to -> Compressed (zipped) folder, then upload the zip to RunPod.")
