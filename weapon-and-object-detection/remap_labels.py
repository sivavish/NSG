# remap_labels.py
import os, glob, shutil

LABEL_DIR = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\Dataset\labels"
BACKUP_DIR = LABEL_DIR + "_backup"
os.makedirs(BACKUP_DIR, exist_ok=True)

# EDIT this mapping based on your preview:
# mappings = { source_class_id: target_class_id, ... }
# Example mapping if 15 -> person (0) and 16 -> weapon (1):
mappings = {
    15: 0,
    16: 1
}

print("Backing up existing labels to:", BACKUP_DIR)
for f in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    shutil.copy2(f, os.path.join(BACKUP_DIR, os.path.basename(f)))

print("Remapping labels...")
count_files = 0
for path in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    count_files += 1
    with open(path, 'r', errors='ignore') as fh:
        lines = [l.strip() for l in fh if l.strip()]
    new_lines = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            # skip malformed lines
            continue
        try:
            cls = int(parts[0])
        except:
            continue
        if cls in mappings:
            parts[0] = str(mappings[cls])
            new_lines.append(" ".join(parts))
        else:
            # If label is already 0 or 1 leave as-is (or drop if it's irrelevant)
            if cls in (0,1):
                new_lines.append(" ".join(parts))
            else:
                # If other classes exist and should be ignored, skip them.
                # Alternatively you can map them to a default (like 1 for weapon) by editing mappings above.
                # Here we skip those lines.
                continue
    # overwrite file
    with open(path, 'w') as out:
        out.write("\n".join(new_lines) + ("\n" if new_lines else ""))

print("Processed", count_files, "label files. Backup kept at:", BACKUP_DIR)
print("Now run the split script (split_again.py) and then retrain/fine-tune.")
