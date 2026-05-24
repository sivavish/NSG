"""
Simple PyQt5 GUI for uploading an image and running YOLO detection.
Save as image_detector_gui.py and run: python image_detector_gui.py
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------- USER CONFIG --------------------
# Change this to the path of your trained best.pt
DEFAULT_MODEL_PATH = r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\runs\detect\train6\weights\best.pt"
# -----------------------------------------------------

def cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    """Convert an OpenCV image (BGR or RGB) to QImage."""
    if cv_img is None:
        return QImage()
    # If image has 3 channels
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
        # Detect whether image is BGR (common for cv2). We'll convert BGR->RGB.
        # Heuristic: assume BGR -> convert
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    # If single-channel (grayscale)
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        return QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8).copy()
    # Otherwise try safe fallback via PIL
    pil = Image.fromarray(cv_img)
    img_data = pil.convert("RGBA").tobytes("raw", "RGBA")
    h, w = pil.size[1], pil.size[0]
    return QImage(img_data, w, h, QImage.Format_RGBA8888).copy()


class DetectorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weapon Detection — Upload Image")
        self.setMinimumSize(1000, 600)
        self.model_path = DEFAULT_MODEL_PATH
        self.model = None
        self.img_path = None
        self.result_img = None

        self.build_ui()
        self.load_model_button.setFocus()

    def build_ui(self):
        # Left: original image, Right: result
        self.orig_label = QLabel("Original Image")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setFixedSize(480, 480)
        self.orig_label.setStyleSheet("background: #f6f8fa; border: 1px solid #cccccc;")

        self.result_label = QLabel("Detection Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(480, 480)
        self.result_label.setStyleSheet("background: #fffdf6; border: 1px solid #cccccc;")

        # Buttons
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.on_load_model)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.on_upload_image)

        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self.on_run_detection)
        self.detect_btn.setEnabled(False)

        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.on_save_result)
        self.save_btn.setEnabled(False)

        # Status label
        self.status_label = QLabel("Model not loaded.")
        self.status_label.setAlignment(Qt.AlignLeft)

        # Top controls layout
        controls = QHBoxLayout()
        controls.addWidget(self.load_model_button)
        controls.addWidget(self.upload_btn)
        controls.addWidget(self.detect_btn)
        controls.addWidget(self.save_btn)
        controls.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        controls.addWidget(self.status_label)

        # Images layout
        imgs = QHBoxLayout()
        imgs.addWidget(self.orig_label)
        imgs.addWidget(self.result_label)

        # Main layout
        v = QVBoxLayout()
        v.addLayout(controls)
        v.addLayout(imgs)
        self.setLayout(v)

    def on_load_model(self):
        # Allow user to choose model file (best.pt)
        path, _ = QFileDialog.getOpenFileName(self, "Select model (.pt)", str(Path.home()), "PyTorch model (*.pt *.pth)")
        if not path:
            return
        self.model_path = path
        try:
            self.status_label.setText("Loading model...")
            QApplication.processEvents()
            self.model = YOLO(self.model_path)
            self.status_label.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            self.detect_btn.setEnabled(True)
        except Exception as e:
            self.model = None
            self.status_label.setText("Failed to load model.")
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def on_upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Choose image", str(Path.home()), "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not fname:
            return
        self.img_path = fname
        pix = QPixmap(fname)
        pix = pix.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.orig_label.setPixmap(pix)
        self.status_label.setText(f"Loaded image: {os.path.basename(fname)}")
        # enable detect if model loaded
        if self.model is not None:
            self.detect_btn.setEnabled(True)

    def on_run_detection(self):
        if not self.model:
            # auto-load default model if exists
            if Path(self.model_path).exists():
                try:
                    self.status_label.setText("Loading default model...")
                    QApplication.processEvents()
                    self.model = YOLO(self.model_path)
                    self.status_label.setText(f"Model loaded: {os.path.basename(self.model_path)}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
                    return
            else:
                QMessageBox.warning(self, "Model missing", "No model loaded and default model path not found.")
                return

        if not self.img_path:
            QMessageBox.warning(self, "No image", "Please upload an image first.")
            return

        # Disable buttons while running
        self.detect_btn.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Running detection... (this may take a few seconds)")
        QApplication.processEvents()

        try:
            # Run inference using ultralytics API, returns list of results
            results = self.model.predict(source=self.img_path, imgsz=640, conf=0.25, device="cpu", save=False)

            # results[0].plot() returns an annotated image (numpy array)
            annotated = results[0].plot()  # likely BGR (OpenCV style)
            # Convert to QImage and display
            qimg = cv2_to_qimage(annotated)
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_label.setPixmap(pix)

            # store result for saving
            self.result_img = annotated
            self.save_btn.setEnabled(True)
            self.status_label.setText("Detection complete.")
        except Exception as e:
            QMessageBox.critical(self, "Detection error", f"An error occurred during detection:\n{e}")
            self.status_label.setText("Detection failed.")
        finally:
            self.detect_btn.setEnabled(True)
            self.load_model_button.setEnabled(True)
            self.upload_btn.setEnabled(True)

    def on_save_result(self):
        if self.result_img is None:
            QMessageBox.information(self, "No result", "No detection result to save.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save annotated image", "detection_result.jpg", "JPEG (*.jpg);;PNG (*.png)")
        if not fname:
            return
        # annotated is numpy array (likely BGR), save using cv2
        try:
            cv2.imwrite(fname, self.result_img)
            QMessageBox.information(self, "Saved", f"Saved annotated image to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save file:\n{e}")


def main():
    app = QApplication(sys.argv)
    w = DetectorGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
