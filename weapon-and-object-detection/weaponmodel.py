import os
import shutil
from glob import glob
import random

# Paths
IMAGE_DIR = "Dataset/images"
LABEL_DIR = "Dataset/labels"
OUTPUT_DIR = "Dataset/splitted"

# Create folders
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

# Load all images
images = glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png"))

# 80% train, 20% val
random.shuffle(images)
split_index = int(0.8 * len(images))

train_files = images[:split_index]
val_files = images[split_index:]

def move_files(file_list, split):
    for img_path in file_list:
        filename = os.path.basename(img_path)
        label_path = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Move image
        shutil.copy(img_path, f"{OUTPUT_DIR}/{split}/images/{filename}")

        # Move label if exists
        if os.path.exists(label_path):
            shutil.copy(label_path, f"{OUTPUT_DIR}/{split}/labels/{filename.replace('.jpg','.txt').replace('.png','.txt')}")

# Move files
move_files(train_files, "train")
move_files(val_files, "val")

print("Dataset auto-split completed!")
