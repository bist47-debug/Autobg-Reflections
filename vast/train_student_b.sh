#!/bin/bash
# Distillation PLAN B: the student gets the bottom half of the frame (1024x384) with a faint MIRROR HINT
# (car flipped under its ground line) and learns v2's reflection from there. Bigger LoRA, higher LR.
# Reuses the teacher pairs and the student environment from plan A.   usage: bash vast/train_student_b.sh [max_steps]
set -e
STEPS=${1:-8000}
cd /workspace
pkill -f train_pix2pix_turbo || true; sleep 3        # stop plan A if it is still running
[ -f distill/student_data_b/train_prompts.json ] || i2ienv/bin/python Autobg-Reflections/distill/prep_student_data_b.py \
     distill/pairs distill/backdrops distill/student_data_b
N=$(nvidia-smi -L | wc -l)
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
cd img2img-turbo
nohup /workspace/i2ienv/bin/accelerate launch --num_processes $N $( [ $N -gt 1 ] && echo --multi_gpu ) --mixed_precision no \
  src/train_pix2pix_turbo.py \
  --pretrained_model_name_or_path=stabilityai/sd-turbo \
  --output_dir=/workspace/distill/student_b \
  --dataset_folder=/workspace/distill/student_data_b \
  --train_image_prep no_resize --test_image_prep no_resize --resolution 1024 \
  --lora_rank_unet 32 --lora_rank_vae 16 \
  --train_batch_size 2 --max_train_steps $STEPS --checkpointing_steps 1000 \
  --eval_freq 500 --num_samples_eval 50 --viz_freq 200 \
  --learning_rate 1e-5 --allow_tf32 --dataloader_num_workers 4 \
  --report_to wandb --tracker_project_name autobg_student_b \
  > /workspace/distill/train_b_log.txt 2>&1 &
echo "plan B training started on $N GPU(s), $STEPS steps.  Progress:  tail -f /workspace/distill/train_b_log.txt"
echo "check the hint first: /workspace/distill/student_data_b/check/*.jpg (top = input with hint, bottom = v2 target)"
