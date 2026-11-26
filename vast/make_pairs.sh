#!/bin/bash
# Distillation step 1: run the teacher (Qwen + v2, 20 steps) over the 5000 AutoBG cars -> input/target pairs.
# usage:  bash vast/make_pairs.sh <drive_id_part1> <drive_id_part2>
# Uses every GPU on the machine (one process per GPU). Safe to re-run: finished cars are skipped.
set -e
cd /workspace
pip install -q gdown
mkdir -p distill/cars
for id in "$@"; do
  f=distill/$(echo $id | cut -c1-8).zip
  [ -f "$f.ok" ] || { gdown --fuzzy "https://drive.google.com/file/d/$id/view" -O "$f" && unzip -q -o "$f" -d distill/cars && touch "$f.ok" && rm -f "$f"; }
done
echo "cars: $(ls distill/cars | grep -vc _shadow)"
N=$(nvidia-smi -L | wc -l)
cd /workspace/Autobg-Reflections/app
for g in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$g nohup python reflect_autobg.py --full --lora /workspace/v2.safetensors \
     --backdrop /workspace/floor_satin_light.png --pairs /workspace/distill/cars \
     --pairs_out /workspace/distill/pairs --shard $g/$N > /workspace/distill/log_gpu$g.txt 2>&1 &
done
echo "started $N GPU worker(s). Progress:  tail -f /workspace/distill/log_gpu0.txt"
