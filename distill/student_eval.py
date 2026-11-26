"""Distillation step 3: run a trained student checkpoint on the held-out test cars.
Saves <name>_cmp.jpg = input | teacher (v2) | student, and prints the speed per image.
usage: python student_eval.py <model_XXXX.pkl> /workspace/distill/student_data /workspace/distill/eval_XXXX
"""
import os, sys, json, time, torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
sys.path.insert(0, "/workspace/img2img-turbo/src")
from pix2pix_turbo import Pix2Pix_Turbo

ckpt, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
prompts = json.load(open(os.path.join(data, "test_prompts.json")))
net = Pix2Pix_Turbo(pretrained_path=ckpt); net.set_eval(); net.half()
ts = []
for i, (n, p) in enumerate(sorted(prompts.items())):
    a = Image.open(os.path.join(data, "test_A", n)).convert("RGB")
    b = Image.open(os.path.join(data, "test_B", n)).convert("RGB")
    x = F.to_tensor(a).unsqueeze(0).cuda().half()
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        y = net(x, p)
    torch.cuda.synchronize(); ts.append(time.time() - t)
    s = Image.fromarray(((y[0].float().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    s.save(os.path.join(out, n.replace(".png", "_student.png")))
    cmp_ = Image.new("RGB", (a.width * 3, a.height)); cmp_.paste(a, (0, 0)); cmp_.paste(b, (a.width, 0)); cmp_.paste(s, (2 * a.width, 0))
    cmp_.save(os.path.join(out, n.replace(".png", "_cmp.jpg")), quality=90)
print(f"{len(ts)} test cars -> {out}   speed: {np.median(ts[3:] or ts) * 1000:.0f} ms / image (fp16, this GPU)")
