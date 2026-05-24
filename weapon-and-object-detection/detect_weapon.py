"""
weapon_gui_murf_tts.py
Weapon detection GUI (Tkinter) + Murf.ai TTS integration (REST).

Requirements:
    pip install ultralytics opencv-python pillow requests playsound==1.2.2

This script calls:
    POST https://api.murf.ai/v1/speech/generate
with header "api-key" set to your Murf API key and encodeAsBase64=True
so the response contains encodedAudio (base64). We write that to a temp mp3 and play it.
"""

import os
import threading
import tempfile
import time
import base64
import traceback
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from playsound import playsound
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------ CONFIG ------------------
MODEL_PATH = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\runs\detect\train7\weights\best.pt"
CONF = 0.25
SPEAK_DELAY_MS = 1000  # speak after 1 second
MURF_API_KEY = "ap2_6c8e4b9d-2917-4e53-9bfc-1aaf580d3b7e"  # <-- your Murf API key (inserted)
MURF_ENDPOINT = "https://api.murf.ai/v1/speech/generate"
DEFAULT_VOICE_ID = "en-US-natalie"  # you can change to any supported voiceId from Murf voices
DEFAULT_FORMAT = "MP3"  # MP3 or WAV etc.
# -------------------------------------------

if not MURF_API_KEY:
    print("Warning: MURF_API_KEY is empty. Set the MURF_API_KEY variable in the script.")

# Load YOLO model once (preload to speed up repeated detections)
print("Loading YOLO model from:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model loaded.")

# ----------------- Helper functions -----------------
def build_sentence_from_counts(counts: dict) -> str:
    """Build a natural sentence from counts (person/weapon focused)."""
    person_count = 0
    weapon_count = 0
    for k, v in counts.items():
        kl = k.lower()
        if "person" in kl or "people" in kl or "human" in kl:
            person_count += v
        if "weapon" in kl or "gun" in kl or "knife" in kl:
            weapon_count += v

    if person_count > 0 and weapon_count > 0:
        p_word = "person" if person_count == 1 else "people"
        w_word = "weapon" if weapon_count == 1 else "weapons"
        return f"Warning: {person_count} {p_word} detected with {weapon_count} {w_word}."
    parts = []
    if person_count > 0:
        p_word = "person" if person_count == 1 else "people"
        parts.append(f"{person_count} {p_word}")
    if weapon_count > 0:
        w_word = "weapon" if weapon_count == 1 else "weapons"
        parts.append(f"{weapon_count} {w_word}")
    if parts:
        return " and ".join(parts) + " detected."
    # fallback: list classes
    if counts:
        parts = [f"{v} {k}" for k, v in counts.items()]
        if len(parts) == 1:
            return parts[0] + " detected."
        return ", ".join(parts[:-1]) + " and " + parts[-1] + " detected."
    return "No objects detected."

def murf_synthesize_to_file(text: str, voice_id: str = DEFAULT_VOICE_ID, audio_format: str = DEFAULT_FORMAT,
                            encode_base64: bool = True, rate: int = 0, pitch: int = 0) -> str:
    """
    Call Murf TTS REST API and return path to the generated audio file (mp3/wav).
    Raises RuntimeError on failures.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }
    payload = {
        "text": text,
        "voiceId": voice_id,
        "format": audio_format,
        "encodeAsBase64": encode_base64
    }
    # optional extras:
    # if rate != 0: payload["rate"] = rate
    # if pitch != 0: payload["pitch"] = pitch

    resp = requests.post(MURF_ENDPOINT, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Murf TTS API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Prefer encodedAudio if present (safer — no file retention on Murf)
    if data.get("encodedAudio"):
        audio_b64 = data["encodedAudio"]
        audio_bytes = base64.b64decode(audio_b64)
        ext = ".mp3" if audio_format.lower() == "mp3" else ".wav"
        fd, out_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return out_path
    # else if Murf returned audioFile (URL), download it
    if data.get("audioFile"):
        audio_url = data["audioFile"]
        dl = requests.get(audio_url, stream=True, timeout=30)
        dl.raise_for_status()
        ext = ".mp3" if audio_format.lower() == "mp3" else ".wav"
        fd, out_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return out_path
    raise RuntimeError("Murf TTS response missing audio content: " + str(data))

def play_audio_blocking(path: str):
    """Plays audio file using playsound (blocking call)"""
    try:
        playsound(path)
    except Exception as e:
        # fallback to OS open (best-effort)
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                import subprocess, sys
                if sys.platform.startswith("darwin"):
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
        except Exception:
            print("Playback failed:", e)

def synthesize_and_play_background(text: str, voice_id: str = DEFAULT_VOICE_ID, audio_format: str = DEFAULT_FORMAT):
    """Worker that calls Murf, writes temp file, plays it, then deletes file."""
    def worker():
        mp = None
        try:
            mp = murf_synthesize_to_file(text, voice_id=voice_id, audio_format=audio_format, encode_base64=True)
        except Exception as ex:
            print("Murf synthesis error:", ex)
            traceback.print_exc()
            return
        try:
            play_audio_blocking(mp)
        except Exception as e:
            print("Playback error:", e)
        finally:
            # cleanup
            try:
                if mp and os.path.exists(mp):
                    time.sleep(0.2)
                    os.remove(mp)
            except Exception:
                pass
    threading.Thread(target=worker, daemon=True).start()

# ------------------- GUI -------------------
root = Tk()
root.title("Weapon Detection — Murf TTS")
root.geometry("960x740")
root.configure(bg="#222222")

Label(root, text="Weapon Detection — Murf TTS", font=("Arial", 20, "bold"), fg="white", bg="#222222").pack(pady=10)

img_frame = Label(root, bg="#111111")
img_frame.pack(pady=8)

result_label = Label(root, text="", font=("Arial", 14), fg="white", bg="#222222")
result_label.pack(pady=8)

scheduled_speak_id = None

def cancel_scheduled_speak():
    global scheduled_speak_id
    if scheduled_speak_id:
        try:
            root.after_cancel(scheduled_speak_id)
        except Exception:
            pass
        scheduled_speak_id = None

def schedule_murf_speak(counts: dict):
    global scheduled_speak_id
    cancel_scheduled_speak()
    sentence = build_sentence_from_counts(counts)
    # schedule synth+play after delay (so user sees image first)
    scheduled_speak_id = root.after(SPEAK_DELAY_MS, lambda: synthesize_and_play_background(sentence, voice_id=DEFAULT_VOICE_ID, audio_format="MP3"))

def upload_image_and_detect():
    cancel_scheduled_speak()
    file_path = filedialog.askopenfilename(filetypes=[("Images","*.jpg;*.jpeg;*.png;*.bmp;*.webp"),("All files","*.*")])
    if not file_path:
        return

    result_label.config(text="Running detection...")
    root.update_idletasks()

    # run detection
    results = model.predict(file_path, conf=CONF)
    if len(results) == 0:
        result_label.config(text="No results returned.")
        return
    r = results[0]
    img = cv2.imread(file_path)
    if img is None:
        result_label.config(text="Failed to open image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # parse boxes safely
    boxes, scores, classes = [], [], []
    try:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        pass

    counts = {}
    for box, conf, cls in zip(boxes, scores, classes):
        name = model.names[int(cls)] if hasattr(model, "names") and int(cls) in model.names else str(int(cls))
        counts[name] = counts.get(name, 0) + 1
        x1,y1,x2,y2 = map(int, box)
        color = (0,255,0) if "person" in name.lower() else (0,0,255)
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img_rgb, f"{name} {conf:.2f}", (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # show image
    from PIL import Image, ImageTk
    pil = Image.fromarray(img_rgb)
    pil.thumbnail((900, 600))
    imgtk = ImageTk.PhotoImage(pil)
    img_frame.config(image=imgtk)
    img_frame.image = imgtk

    # textual summary
    if counts:
        text = "Detected:\n" + "\n".join([f"• {k}: {v}" for k,v in counts.items()])
    else:
        text = "No objects detected."
    result_label.config(text=text)

    # schedule Murf TTS speak
    schedule_murf_speak(counts)

upload_btn = Button(root, text="Upload Image & Detect", font=("Arial", 14), bg="#4caf50", fg="white", padx=16, pady=8, command=upload_image_and_detect)
upload_btn.pack(pady=14)

root.mainloop()
