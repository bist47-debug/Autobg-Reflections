#!/bin/bash
# Distillation step 2: train the small one-step student (pix2pix-turbo, SD-Turbo based) on the teacher pairs.
# Run on the same Vast machine after make_pairs.sh finished.  usage: bash vast/train_student.sh [max_steps]
set -e
STEPS=${1:-8000}
cd /workspace
[ -d img2img-turbo ] || git clone -q https://github.com/GaParmar/img2img-turbo.git
# separate environment: img2img-turbo needs older diffusers / transformers than the Qwen pipeline
if [ ! -d i2ienv ]; then
  python -m venv --system-site-packages i2ienv
  i2ienv/bin/pip install -q "diffusers==0.25.1" "transformers==4.35.2" "huggingface_hub==0.25.2" "peft==0.7.1" \
      "accelerate==0.27.2" lpips clean-fid vision_aided_loss open-clip-torch wandb "git+https://github.com/openai/CLIP.git"
fi
[ -f distill/student_data/train_prompts.json ] || i2ienv/bin/python Autobg-Reflections/distill/prep_student_data.py distill/pairs distill/student_data
N=$(nvidia-smi -L | wc -l)
export WANDB_MODE=offline
cd img2img-turbo
nohup /workspace/i2ienv/bin/accelerate launch --num_processes $N $( [ $N -gt 1 ] && echo --multi_gpu ) --mixed_precision no \
  src/train_pix2pix_turbo.py \
  --pretrained_model_name_or_path=stabilityai/sd-turbo \
  --output_dir=/workspace/distill/student \
  --dataset_folder=/workspace/distill/student_data \
  --train_image_prep no_resize --test_image_prep no_resize --resolution 768 \
  --train_batch_size 2 --max_train_steps $STEPS --checkpointing_steps 1000 \
  --eval_freq 500 --num_samples_eval 50 --viz_freq 200 \
  --learning_rate 5e-6 --allow_tf32 --dataloader_num_workers 4 \
  --report_to wandb --tracker_project_name autobg_student \
  > /workspace/distill/train_log.txt 2>&1 &
echo "training started on $N GPU(s), $STEPS steps. Progress:  tail -f /workspace/distill/train_log.txt"
echo "checkpoints every 1000 steps in /workspace/distill/student/checkpoints/"
