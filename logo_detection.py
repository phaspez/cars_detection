import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from typing import Tuple  # <-- THÊM DÒNG NÀY

MODEL_PATH = "runs/detect/train/weights/best.pt"
IOU_THRESH = 0.5
CONF_THRESH = 0.3

CLASS_NAMES = [
    "bmw", "mercedes", "vinfast", "toyota", "mitsubishi",
    "ford", "honda", "hyundai", "kia", "bien_so"
]
LOGO_CLASS_INDICES = set(range(0, 9))
BIENSO_INDEX = 9

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model không tồn tại: {path}")
    print(f"Đang tải model từ: {path}")
    model = YOLO(path)
    print("Tải model thành công.")
    return model


def gaussian_blur_region(img, x1, y1, x2, y2, k_ratio=0.12):
    h, w = img.shape[:2]
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w - 1, int(round(x2)))
    y2 = min(h - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return img
    roi = img[y1:y2, x1:x2]
    k = int(min(max(3, int(min(roi.shape[0], roi.shape[1]) * k_ratio)), 101))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred
    return img


def process_image_logic(model: YOLO, image_bytes: bytes) -> Tuple[Image.Image, str, str]:
    # 1. Đọc ảnh từ bytes (thay vì file_path)
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Không đọc được dữ liệu ảnh.")

    orig_h, orig_w = img_bgr.shape[:2]

    # 2. Chạy model
    # Quan trọng: model.predict() có thể nhận trực tiếp mảng numpy BGR
    results = model.predict(source=img_bgr.copy(), conf=0.5, iou=IOU_THRESH, save=False)
    r = results[0]

    boxes, classes, confs = [], [], []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy()
        for b, c, cf in zip(xyxy, cls_arr, conf_arr):
            boxes.append(b.tolist())
            classes.append(int(c))
            confs.append(float(cf))

    # 3. Xử lý logic nếu không tìm thấy gì
    if len(boxes) == 0:
        pil_clean = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return (pil_clean, "Unknown", "Không phát hiện đối tượng nào")

    # 4. Tìm logo tốt nhất
    best_logo_idx = None
    best_conf = -1.0
    for i, cls in enumerate(classes):
        if cls in LOGO_CLASS_INDICES and confs[i] > best_conf:
            best_conf = confs[i]
            best_logo_idx = i

    if best_logo_idx is not None:
        best_logo_name = CLASS_NAMES[classes[best_logo_idx]]
        best_conf_text = f"{best_logo_name} ({best_conf * 100:.1f}%)"
    else:
        best_logo_name = "Unknown"
        best_conf_text = "Không tìm thấy logo"

    # 5. Áp dụng làm mờ / che
    # Chỉ cần tạo 1 ảnh 'sạch' để lưu
    img_clean = img_bgr.copy()

    for (b, cls, cf) in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, b)

        if cls in LOGO_CLASS_INDICES:
            img_clean = gaussian_blur_region(img_clean, x1, y1, x2, y2, k_ratio=0.18)
        elif cls == BIENSO_INDEX:
            img_clean[y1:y2, x1:x2] = (0, 0, 0)  # Bôi đen

    # 6. Chuyển sang PIL
    pil_clean = Image.fromarray(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB))

    # 7. Trả về kết quả
    return (pil_clean, best_logo_name, best_conf_text)