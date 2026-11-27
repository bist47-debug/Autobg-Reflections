# AutoBG Reflections — Distillation (fast student model)

Full write-up (with tables): https://claude.ai/artifact/Mtm5eYUDj1VHfsveVggSJd — this file is the short version for the repo.
State: 27 Nov 2026.

## Why
Qwen-Image-Edit 2511 + v2 LoRA (20 steps) needs ~58 GB GPU memory and ~35 s per car, so it cannot run on the
RTX 5060 Ti (16 GB) servers. A one-step **student** (pix2pix-turbo, SD-Turbo based, ~1B parameters) is trained to copy
the **teacher**. Only the AI call changes; all finishing (opacity, fades, tyre reflections, shadow x1.25, placement)
stays as approved.

## Result
- Student AI step: 0.12 s (idle RTX PRO 6000), 0.2–0.9 s in the pipeline; ~5 GB GPU memory.
- Finished images looked like v2 on the first cars tested. 5060 Ti timing still to be confirmed.
- CPU threads capped at 8 (`AUTOBG_THREADS`): rented machines made the CPU steps 20–50x slower.
- Cost: ~$160 on Vast (pairs ~$90, plan A ~$25, plan B ~$45).

## Step 1 — teacher pairs
`bash vast/make_pairs.sh <drive_id_part1> <drive_id_part2>` — downloads the cars (5063 cut-outs + `_shadow.png`),
runs `reflect_autobg.py --pairs` on every GPU (resumable). Backdrops in `/workspace/distill/backdrops/` are mixed in.
Output per car: `_in.png`, `_tgt.png` (aligned v2), `.json`. 4148 pairs (915 side shots skipped).
`tyres.pt` must be in `app/models/` first. ~11 h on 4x RTX PRO 6000.

## Step 2 — student training (plan B)
`bash vast/train_student_b.sh` — `distill/prep_student_data_b.py` makes 1024x384 bottom-half strips with a
**mirror hint** (`distill/hint.py`: car flipped under its ground line, 45 % fading), 7974 train / 161 test (43 cars held
out); then img2img-turbo trains in `/workspace/i2ienv` (diffusers 0.25.1, transformers 4.35.2, setuptools<70),
LoRA 32/16, lr 1e-5, ~10 passes, ~5.5 h on 4 GPUs. Quality was reached by ~step 2000 — check early next time.
Plan A (`prep_student_data.py`, `train_student.sh`: whole frame, no hint) gave dark puddles; kept for reference.

## Step 3 — check and run
- Raw check: `i2ienv/bin/python distill/student_eval.py <model.pkl> <student_data_b> <out>` → `*_cmp.jpg`
  (input | v2 | student) + speed.
- Pipeline: `python app/reflect_autobg.py ... --student <model_9001.pkl> --i2i <img2img-turbo/src>` (Qwen not loaded).
  The log prints each stage's time. Judge finished images, not raw strips.

## Files
- Student: Google Drive `autobg_distill/model_9001.pkl` (139.5 MB)
- Pairs: Google Drive `autobg_distill/pairs_backup.tar` (7.72 GB)
- Cars: Google Drive `cars_ part1.zip`, `cars_ part2.zip`; PC `D:\reflections\distill\cars\`

## Next
1. Test 15–20 cars on the 5060 Ti (target 2–4 s).
2. API / batch entry point for AutoBG.
3. If CPU part >2 s: finishing on GPU, maybe skip BiRefNet re-cut.
