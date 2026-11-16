#!/bin/bash
# One-time setup on a new Vast.ai instance (RTX PRO 6000 96 GB or any 48 GB+ GPU), PyTorch template.
#   V2_DRIVE_ID=xxxx bash vast/setup.sh
# Then upload models/tyres.pt and floor_satin_light.png (not in git) - see docs/REPLICATION.md.
set -e
cd /workspace
git clone https://github.com/bist47-debug/Autobg-Reflections.git 2>/dev/null || (cd Autobg-Reflections && git pull)
pip install -q opencv-python-headless
pip install -q -U gradio peft transformers accelerate timm kornia einops gdown ultralytics
pip install -q -U git+https://github.com/huggingface/diffusers
mkdir -p /workspace/Autobg-Reflections/app/models
# v2 reflection LoRA (Google Drive)
V2_ID="${V2_DRIVE_ID:?set V2_DRIVE_ID to the Google Drive file id of the v2 LoRA (ask Manu)}"
[ -f /workspace/v2.safetensors ] || gdown "$V2_ID" -O /workspace/v2.safetensors
echo "SETUP DONE. Now upload: tyres.pt -> /workspace/Autobg-Reflections/app/models/ ; floor_satin_light.png -> /workspace/"
