import os
import shutil
import random

BASE_DIR = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312"
IMAGE_DIR = os.path.join(BASE_DIR, "Dataset", "images")
LABEL_DIR = os.path.join(BASE_DIR, "Dataset", "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset", "splitted")

# Remove previous split if any
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Make dirs
for split in ("train","val"):
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Collect images
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
images = []
for root, _, files in os.walk(IMAGE_DIR):
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_exts:
            images.append(os.path.join(root, f))

print("Total images found:", len(images))
if not images:
    raise SystemExit("No images found in Dataset/images — check extraction and path.")

random.shuffle(images)
split_idx = int(0.8 * len(images))
train_files = images[:split_idx]
val_files = images[split_idx:]

def copy_pair(files, split):
    img_out = os.path.join(OUTPUT_DIR, split, "images")
    lbl_out = os.path.join(OUTPUT_DIR, split, "labels")
    copied = 0
    missing_lbl = 0
    for img in files:
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(LABEL_DIR, stem + ".txt")
        shutil.copy2(img, os.path.join(img_out, os.path.basename(img)))
        copied += 1
        if os.path.exists(lbl):
            shutil.copy2(lbl, os.path.join(lbl_out, stem + ".txt"))
        else:
            missing_lbl += 1
    print(f"{split}: copied {copied}, {missing_lbl} missing labels")

copy_pair(train_files, "train")
copy_pair(val_files, "val")

print("✅ Splitting done. Check:")
print(os.path.join(OUTPUT_DIR, "train", "images"))
print(os.path.join(OUTPUT_DIR, "val", "images"))
