# preview_classes.py
import os, glob, random, cv2
DATA_LABEL_DIR = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\Dataset\labels"
DATA_IMG_DIR   = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\Dataset\images"
SAMPLES_PER_CLASS = 6

def gather_for_class(class_id):
    files = glob.glob(os.path.join(DATA_LABEL_DIR, "*.txt"))
    matches = []
    for f in files:
        with open(f, 'r', errors='ignore') as fh:
            lines = [l.strip() for l in fh if l.strip()]
            for ln in lines:
                parts = ln.split()
                if parts and parts[0] == str(class_id):
                    matches.append((f, os.path.splitext(os.path.basename(f))[0]))
                    break
    return matches

def show_samples(class_id):
    matches = gather_for_class(class_id)
    if not matches:
        print(f"No examples found for class {class_id}")
        return
    print(f"Found {len(matches)} label files with class {class_id}. Showing up to {SAMPLES_PER_CLASS} samples.")
    random.shuffle(matches)
    shown = 0
    for lab_path, stem in matches[:SAMPLES_PER_CLASS]:
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = os.path.join(DATA_IMG_DIR, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            # if exact-stem image not found, try any file that contains the stem (robust)
            for p in os.listdir(DATA_IMG_DIR):
                if stem in p:
                    img_path = os.path.join(DATA_IMG_DIR, p)
                    break
        if not img_path:
            print("Image not found for", stem, "label file", lab_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print("Cannot read", img_path)
            continue
        cv2.putText(img, f"class {class_id}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow(f"Class {class_id} sample {shown+1}/{SAMPLES_PER_CLASS}", cv2.resize(img, (800,600)))
        shown += 1
        cv2.waitKey(0)  # press any key to go to next
        cv2.destroyAllWindows()
    if shown == 0:
        print("No viewable sample images found for this class.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preview_classes.py 15  (or 16)")
        print("You can pass multiple class ids: python preview_classes.py 15 16")
        sys.exit(1)
    for arg in sys.argv[1:]:
        show_samples(int(arg))
