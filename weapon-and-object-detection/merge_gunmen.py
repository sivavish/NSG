# merge_gunmen.py
import os, shutil, uuid
from glob import glob

SRC_FOLDER = r"C:\Users\DELL\Downloads\gun detection dataset\Gunmen Dataset\All"
TARGET_BASE = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\Dataset"
IMAGES_TARGET = os.path.join(TARGET_BASE, "images")
LABELS_TARGET = os.path.join(TARGET_BASE, "labels")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

os.makedirs(IMAGES_TARGET, exist_ok=True)
os.makedirs(LABELS_TARGET, exist_ok=True)

# find files
all_files = []
for root, _, files in os.walk(SRC_FOLDER):
    for f in files:
        all_files.append(os.path.join(root, f))

img_files = [f for f in all_files if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS]
txt_files = [f for f in all_files if os.path.splitext(f)[1].lower() == ".txt"]

print(f"Found {len(img_files)} images and {len(txt_files)} txt files in source.")

# map label stems -> path
label_map = {}
for t in txt_files:
    stem = os.path.splitext(os.path.basename(t))[0]
    label_map[stem] = t

copied_imgs = 0
copied_lbls = 0

for img in img_files:
    stem = os.path.splitext(os.path.basename(img))[0]
    base_name = os.path.basename(img)
    dst = os.path.join(IMAGES_TARGET, base_name)
    if os.path.exists(dst):
        # avoid collision
        base, ext = os.path.splitext(base_name)
        newname = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
        dst = os.path.join(IMAGES_TARGET, newname)
    shutil.copy2(img, dst)
    copied_imgs += 1

    # try to copy matching label
    lbl_src = label_map.get(stem)
    if lbl_src and os.path.exists(lbl_src):
        new_stem = os.path.splitext(os.path.basename(dst))[0]
        lbl_dst = os.path.join(LABELS_TARGET, new_stem + ".txt")
        shutil.copy2(lbl_src, lbl_dst)
        copied_lbls += 1

print(f"Copied {copied_imgs} images and {copied_lbls} labels to:")
print(" images ->", IMAGES_TARGET)
print(" labels ->", LABELS_TARGET)
