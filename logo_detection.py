import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "runs/detect/train/weights/best.pt"
IOU_THRESH = 0.5
CONF_THRESH = 0.3

CLASS_NAMES = [
    "bmw","mercedes","vinfast","toyota","mitsubishi",
    "ford","honda","hyundai","kia","bien_so"
]
LOGO_CLASS_INDICES = set(range(0, 9))
BIENSO_INDEX = 9


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(path)

def gaussian_blur_region(img, x1, y1, x2, y2, k_ratio=0.12):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return img
    roi = img[y1:y2, x1:x2]
    k = int(min(max(3, int(min(roi.shape[0], roi.shape[1]) * k_ratio)), 101))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred
    return img

