# AutoBG Reflections

Realistic floor reflections for AutoBG car cut-outs on a studio backdrop.
Upload a car cut-out (plus its AutoBG shadow) and get the car on the studio floor with a
matching reflection: tyre reflections on side shots, an AI reflection on 3/4 shots, and
stretched tyre reflections on straight front/back shots.

**Full replication guide:** [docs/REPLICATION.md](docs/REPLICATION.md) — every model, dataset,
training run and locked setting used to get the current results.

**Fast version (distilled student, runs on a 16 GB GPU):** [docs/DISTILLATION.md](docs/DISTILLATION.md).

## Quick start (Vast.ai)

```bash
# new instance: RTX PRO 6000 (96 GB) or any 48 GB+ GPU, PyTorch template
cd /workspace && git clone https://github.com/bist47-debug/Autobg-Reflections.git
V2_DRIVE_ID=<id from Manu> bash Autobg-Reflections/vast/setup.sh
# upload (not in git):  tyres.pt -> /workspace/Autobg-Reflections/app/models/
#                       floor_satin_light.png -> /workspace/
bash Autobg-Reflections/vast/start_team.sh <password>     # team page, login user "autobg"
```

Results the team produced: `bash Autobg-Reflections/vast/collect_team_results.sh`, then download
`/workspace/team`. **Destroy the instance after use.**

## Repository

| Path | What it is |
|---|---|
| `app/reflect_autobg.py` | The app: team page, full page with sliders, batch mode. Locked settings in `BATCH`. |
| `app/wheels.py` | Tyre finder (own YOLO model) + tyre reflection drawing. |
| `app/tyre_label.py`, `app/train_tyres.py` | Label tyres and train the tyre model. |
| `app/wheel_test.py` | Quick tyre-reflection contact sheet, no GPU model needed. |
| `training/render_cars.py` | Blender renders of Objaverse cars (training data for the reflection LoRA). |
| `training/prep_dataset_v1.py`, `training/make_pairs_v2.py` | Build the v1 / v2 LoRA training sets + ai-toolkit configs. |
| `vast/` | Setup, start and result-collection scripts for Vast.ai. |
| `legacy/` | The earlier Colab flip-based version. |

Model weights are **not** in git: `v2.safetensors` (reflection LoRA, Google Drive) and
`tyres.pt` (tyre model). See the guide for where they live and how to rebuild them.

## Licences to check before commercial use

Qwen-Image-Edit-2511 (Apache-2.0), Ultralytics YOLO11 (AGPL-3.0 — commercial/SaaS use needs an
Ultralytics Enterprise licence), BiRefNet (MIT), OWLv2 / SAM (Apache-2.0), Objaverse models
(per-model Creative Commons licences, recorded in each render's `meta.json`). Confirm each on its page.
