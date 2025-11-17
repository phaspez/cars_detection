# python
import os
import cv2
import numpy as np
from difflib import get_close_matches
from PIL import Image
import io
import threading
import torch

import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QFileDialog, QScrollArea, QFrame, QMessageBox, QSizePolicy, QStatusBar
)

# Globals (lazy init)
ocr_reader = None
trocr_processor = None
trocr_model = None

model_dict = [m.lower() for m in [
    "320d", "320i", "330i", "335d", "335i", "530e", "530i", "540i",
    "M2", "M3", "M4", "M5", "M8", "X3", "X4", "X5", "Z3", "Z4",
    "Figo", "Focus", "Fusion", "Mondeo", "Ranger",
    "Civic", "Accord", "City", "CR-V", "HR-V",
    "Accent", "Elantra", "Tucson", "SantaFe", "Sonata",
    "K3", "K5", "Morning", "Seltos", "Sorento",
    "C200", "C300", "E200", "E300", "GLC200", "GLC300",
    "GLE53", "GLE450", "S450", "S500",
    "Attrage", "Mirage", "PAJERO", "OUTLANDER", "MIRAGE", "PAJERO SPORT",
    "Camry", "Corolla", "Highlander", "Prius",
    "VF3", "VF5", "VF6", "VF8", "VF9"
]]

def get_easyocr_reader():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'])
    return ocr_reader


def get_trocr_model():
    global trocr_processor, trocr_model
    if trocr_model is None:
        print("Downloading TrOCR model (this may take a few minutes)...")
        try:
            trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-large-printed",
            )
            print("Processor loaded successfully")

            trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-printed",
            )
            print("Model loaded successfully")
            print(f"Model size: {sum(p.numel() for p in trocr_model.parameters())} parameters")

        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    return trocr_processor, trocr_model


def cv2_to_qimage(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

def qimage_to_pil(qimg):
    buffer = qimg.bits().asstring(qimg.width() * qimg.height() * qimg.depth() // 8)
    mode = "RGB" if qimg.format() == QImage.Format_RGB888 else "RGBA"
    img = Image.frombytes(mode, (qimg.width(), qimg.height()), buffer)
    return img

class DetectThread(QThread):
    finished_signal = pyqtSignal(QImage, list, int)  # image with boxes, crops as QImage list, valid count
    status_signal = pyqtSignal(str)

    def __init__(self, image_path, padding=5, crop_zoom=2.0):
        super().__init__()
        self.image_path = image_path
        self.padding = padding
        self.crop_zoom = crop_zoom

    def run(self):
        try:
            self.status_signal.emit("Initializing EasyOCR...")
            reader = get_easyocr_reader()
            self.status_signal.emit("Detecting text (EasyOCR)...")
            results = reader.readtext(self.image_path, detail=1)

            img = cv2.imread(self.image_path)
            img_with_boxes = img.copy()
            h, w = img.shape[:2]
            crops_qimages = []
            valid = 0

            # Unpack each result: bbox is the polygon, text and conf are other fields
            for bbox, text, conf in results:
                box_np = np.array(bbox, dtype=np.int32)
                x_min, y_min = np.min(box_np, axis=0)
                x_max, y_max = np.max(box_np, axis=0)

                x_min = max(0, x_min - self.padding)
                y_min = max(0, y_min - self.padding)
                x_max = min(w, x_max + self.padding)
                y_max = min(h, y_max + self.padding)

                if x_min >= x_max or y_min >= y_max:
                    continue

                crop = img[y_min:y_max, x_min:x_max]
                if self.crop_zoom != 1.0 and crop.size > 0:
                    crop = cv2.resize(
                        crop,
                        (int((x_max - x_min) * self.crop_zoom), int((y_max - y_min) * self.crop_zoom)),
                        interpolation=cv2.INTER_CUBIC
                    )
                crops_qimages.append(cv2_to_qimage(crop))
                cv2.polylines(img_with_boxes, [box_np], True, (0, 255, 0), 2)
                valid += 1

            self.finished_signal.emit(cv2_to_qimage(img_with_boxes), crops_qimages, valid)
            self.status_signal.emit("Detection finished.")
        except Exception as e:
            self.status_signal.emit(f"Detection error: {e}")
            self.finished_signal.emit(QImage(), [], 0)


class RecognizeThread(QThread):
    result_signal = pyqtSignal(int, str, str)
    status_signal = pyqtSignal(str)

    def __init__(self, qimages_list):
        super().__init__()
        self.qimages_list = qimages_list

    def run(self):
        try:
            self.status_signal.emit("Initializing TrOCR...")
            processor, model = get_trocr_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self.status_signal.emit("Recognizing (TrOCR)...")

            for idx, qimg in enumerate(self.qimages_list):
                try:
                    # Convert QImage to PIL Image
                    pil_img = qimage_to_pil(qimg)

                    # Process with TrOCR
                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values, max_length=64, num_beams=1)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    # Match with car models
                    clean_text = ''.join(filter(str.isalnum, text)).lower()
                    match = get_close_matches(clean_text, model_dict, n=1, cutoff=0.7)
                    match_str = f"→ Match: {match[0]}" if match else "→ No match"

                    self.result_signal.emit(idx, text, match_str)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
                    self.result_signal.emit(idx, f"Error: {e}", "")

            self.status_signal.emit("Recognition finished.")
        except Exception as e:
            print(f"Recognition thread error: {e}")
            self.status_signal.emit(f"Recognition error: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Model OCR (EasyOCR + TrOCR)")
        self.setGeometry(100, 100, 1200, 800)

        self.image_path = None
        self.crops_qimages = []

        # Top controls
        btn_select = QPushButton("1. Select Image")
        btn_detect = QPushButton("2. Detect (EasyOCR)")
        btn_recognize = QPushButton("3. Recognize (TrOCR)")

        btn_detect.setEnabled(False)
        btn_recognize.setEnabled(False)

        btn_select.clicked.connect(self.select_image)
        btn_detect.clicked.connect(self.start_detect)
        btn_recognize.clicked.connect(self.start_recognize)

        control_layout = QHBoxLayout()
        control_layout.addWidget(btn_select)
        control_layout.addWidget(btn_detect)
        control_layout.addWidget(btn_recognize)

        # Scroll area for display
        self.scroll_area = QScrollArea()
        self.display_widget = QWidget()
        self.display_layout = QVBoxLayout()
        self.display_layout.setAlignment(Qt.AlignTop)
        self.display_widget.setLayout(self.display_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.display_widget)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.scroll_area)
        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # keep buttons references
        self.btn_detect = btn_detect
        self.btn_recognize = btn_recognize

    def update_status(self, text):
        self.status.showMessage(text)

    def clear_display(self):
        while self.display_layout.count():
            item = self.display_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def display_qimage(self, qimg, caption=None, max_size=(900,600)):
        if qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(max_size[0], max_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        frame = QFrame()
        layout = QVBoxLayout()
        if caption:
            layout.addWidget(QLabel(caption))
        layout.addWidget(lbl)
        frame.setLayout(layout)
        self.display_layout.addWidget(frame)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.image_path = path
        self.crops_qimages = []
        self.clear_display()
        self.display_qimage(cv2_to_qimage(cv2.imread(path)), caption=os.path.basename(path))
        self.update_status(f"Loaded: {os.path.basename(path)}")
        self.btn_detect.setEnabled(True)
        self.btn_recognize.setEnabled(False)

    def start_detect(self):
        if not self.image_path:
            return
        self.btn_detect.setEnabled(False)
        self.btn_recognize.setEnabled(False)
        self.clear_display()
        self.update_status("Starting detection...")
        self.detect_thread = DetectThread(self.image_path)
        self.detect_thread.status_signal.connect(self.update_status)
        self.detect_thread.finished_signal.connect(self.on_detect_finished)
        self.detect_thread.start()

    def on_detect_finished(self, img_with_boxes_qimg, crops_qimages, valid_count):
        # handle thread failure / empty image
        if img_with_boxes_qimg.isNull():
            self.clear_display()
            # show original image if available so user sees something
            if self.image_path and os.path.exists(self.image_path):
                orig = cv2.imread(self.image_path)
                if orig is not None:
                    self.display_qimage(cv2_to_qimage(orig), caption=os.path.basename(self.image_path))
            QMessageBox.warning(self, "Detection", "Detection failed or produced no output. Check status for errors.")
            self.update_status("Detection failed.")
            self.crops_qimages = []
            self.btn_detect.setEnabled(True)
            self.btn_recognize.setEnabled(False)
            return

        # normal path: show image with boxes
        self.display_qimage(img_with_boxes_qimg, caption=f"Detected {valid_count} regions")
        self.crops_qimages = crops_qimages

        if valid_count == 0:
            self.update_status("No text regions detected.")
            self.btn_recognize.setEnabled(False)
        else:
            self.update_status(f"Detection complete. {valid_count} regions found.")
            self.btn_recognize.setEnabled(True)

        self.btn_detect.setEnabled(True)

        # show crops preview if any
        for i, qimg in enumerate(self.crops_qimages):
            self.display_qimage(qimg, caption=f"Crop #{i + 1}", max_size=(400, 150))

    def start_recognize(self):
        if not self.crops_qimages:
            return
        print("Starting recognition...")
        self.btn_detect.setEnabled(False)
        self.btn_recognize.setEnabled(False)
        self.update_status("Starting recognition...")
        self.clear_display()
        label = QLabel("Recognition results (TrOCR):")
        label.setStyleSheet("font-weight:bold;")
        self.display_layout.addWidget(label)

        # placeholder labels for each crop
        self.result_labels = []
        for i, qimg in enumerate(self.crops_qimages):
            frame = QFrame()
            v = QVBoxLayout()
            v.addWidget(QLabel(f"Region #{i + 1}"))
            pix_lbl = QLabel()
            pix = QPixmap.fromImage(qimg).scaled(400, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pix_lbl.setPixmap(pix)
            v.addWidget(pix_lbl)
            res_lbl = QLabel("Recognizing...")
            res_lbl.setStyleSheet("font-family: Consolas;")
            v.addWidget(res_lbl)
            frame.setLayout(v)
            self.display_layout.addWidget(frame)
            self.result_labels.append(res_lbl)

        # Pass QImages - conversion happens in the thread
        self.rec_thread = RecognizeThread(self.crops_qimages)
        self.rec_thread.status_signal.connect(self.update_status)
        self.rec_thread.result_signal.connect(self.on_recognition_result)
        self.rec_thread.finished.connect(self.on_recognition_finished)
        self.rec_thread.start()

    def on_recognition_result(self, idx, text, match_str):
        if 0 <= idx < len(self.result_labels):
            self.result_labels[idx].setText(f"Result: {text}   {match_str}")

    def on_recognition_finished(self):
        self.update_status("Recognition finished.")
        self.btn_detect.setEnabled(True)
        self.btn_recognize.setEnabled(True)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
