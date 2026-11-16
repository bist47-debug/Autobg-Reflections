r"""
train_tyres.py - train your own tyre finder (YOLO segmentation) on the labels from tyre_label.py.

    cd /d D:\reflections
    python -m pip install -U ultralytics
    python train_tyres.py

Runs on your GPU (~20-40 min on a 5060 Ti). Result: models\tyres.pt  (wheels.py uses it automatically).
"""
import glob, os, random, shutil
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "tyre_data", "ds")
SPLIT = os.path.join(HERE, "tyre_data", "split")

def main():
    # 90 % train / 10 % check
    imgs = sorted(glob.glob(os.path.join(DS, "images", "*.jpg")))
    random.seed(0); random.shuffle(imgs)
    n_val = max(10, len(imgs) // 10)
    for part, files in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(SPLIT, part, sub), exist_ok=True)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            shutil.copy(f, os.path.join(SPLIT, part, "images", stem + ".jpg"))
            shutil.copy(os.path.join(DS, "labels", stem + ".txt"), os.path.join(SPLIT, part, "labels", stem + ".txt"))
    yaml = os.path.join(SPLIT, "tyres.yaml")
    open(yaml, "w").write(f"path: {SPLIT}\ntrain: train/images\nval: val/images\nnames:\n  0: tyre\n")
    print(f"{len(imgs) - n_val} train, {n_val} check images")

    model = YOLO("yolo11s-seg.pt")
    model.train(data=yaml, imgsz=1024, epochs=120, batch=6, patience=30, workers=2,
                fliplr=0.5, mosaic=0.5, degrees=0, scale=0.3, hsv_v=0.3,
                project=os.path.join(HERE, "tyre_data", "runs"), name="tyres", exist_ok=True)
    best = os.path.join(HERE, "tyre_data", "runs", "tyres", "weights", "best.pt")
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    shutil.copy(best, os.path.join(HERE, "models", "tyres.pt"))
    print("done -> models\\tyres.pt")


if __name__ == "__main__":      # needed on Windows (data loader workers)
    main()
