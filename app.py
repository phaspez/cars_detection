import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_PATH = "runs/detect/train/weights/best.pt" 
IOU_THRESH = 0.5

CLASS_NAMES = [
    "bmw","mercedes","vinfast","toyota","mitsubishi",
    "ford","honda","hyundai","kia","bien_so"
]
LOGO_CLASS_INDICES = set(range(0, 9)) 
BIENSO_INDEX = 9

MAX_DISPLAY_W, MAX_DISPLAY_H = 900, 700


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model không tồn tại: {path}")
    return YOLO(path)

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

def resize_for_display(pil_img, max_w=MAX_DISPLAY_W, max_h=MAX_DISPLAY_H):
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    return pil_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Detect + Blur & Save name (logo)")
        root.geometry(f"{MAX_DISPLAY_W+40}x{MAX_DISPLAY_H+160}")

        # load model
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Lỗi model", f"Không thể load model:\n{e}")
            root.destroy()
            return

        # UI
        topf = tk.Frame(root)
        topf.pack(pady=6)

        btn_choose = tk.Button(topf, text="Chọn ảnh", command=self.choose_image, width=18)
        btn_choose.pack(side=tk.LEFT, padx=6)

        btn_save = tk.Button(topf, text="Lưu ảnh & tên (TXT)", command=self.save_result, width=18)
        btn_save.pack(side=tk.LEFT, padx=6)

        self.info_label = tk.Label(root, text="Chưa có ảnh", font=("Arial", 14))
        self.info_label.pack(pady=6)

        self.canvas = tk.Label(root)
        self.canvas.pack(padx=10, pady=6)

        # lưu trạng thái
        self.last_result_pil = None   
        self.last_logo_name = None
        self.last_image_path = None

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            self.process_image(path)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh:\n{e}")

    def process_image(self, file_path):
        img_arr = np.fromfile(file_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Không đọc được ảnh. Kiểm tra đường dẫn/định dạng.")
        orig_h, orig_w = img_bgr.shape[:2]

        results = self.model.predict(source=file_path, conf=CONF_THRESH, iou=IOU_THRESH, save=False)
        r = results[0]

        boxes = []
        classes = []
        confs = []
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()
            for b, c, cf in zip(xyxy, cls_arr, conf_arr):
                boxes.append(b.tolist())
                classes.append(int(c))
                confs.append(float(cf))

        if len(boxes) == 0:
            self.info_label.config(text="Không phát hiện đối tượng nào")
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            disp = resize_for_display(pil)
            self.last_result_pil = pil
            self.last_logo_name = "Unknown"
            self.show_image(disp)
            return

        best_logo_idx = None
        best_conf = -1.0
        for i, cls in enumerate(classes):
            if cls in LOGO_CLASS_INDICES and confs[i] > best_conf:
                best_conf = confs[i]
                best_logo_idx = i

        if best_logo_idx is not None:
            best_logo_name = CLASS_NAMES[classes[best_logo_idx]]
            best_conf_text = f"{best_logo_name} ({best_conf*100:.1f}%)"
        else:
            best_logo_name = "Unknown"
            best_conf_text = "Không tìm thấy logo"

        img_proc = img_bgr.copy()
        for (b, cls, cf) in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, b)
            if cls in LOGO_CLASS_INDICES:
                img_proc = gaussian_blur_region(img_proc, x1, y1, x2, y2, k_ratio=0.18)
            elif cls == BIENSO_INDEX:
                img_proc[y1:y2, x1:x2] = (0, 0, 0)

            label_text = f"{CLASS_NAMES[cls]} {cf*100:.1f}%"
            cv2.rectangle(img_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_proc, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pil_out = Image.fromarray(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
        self.last_result_pil = pil_out
        self.last_logo_name = best_logo_name
        self.last_image_path = file_path

        disp = resize_for_display(pil_out)
        self.show_image(disp)
        self.info_label.config(text=f"Logo tốt nhất: {best_conf_text}")


    def show_image(self, pil_img):
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=tk_img)
        self.canvas.image = tk_img

    def save_result(self):
        if self.last_result_pil is None:
            messagebox.showinfo("Lưu", "Chưa có ảnh kết quả để lưu.")
            return
        # chọn nơi lưu ảnh
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG","*.jpg"), ("PNG","*.png")],
                                                 initialfile="result.jpg",
                                                 title="Lưu ảnh đã làm mờ")
        if not save_path:
            return
        # lưu ảnh
        try:
         
            self.last_result_pil.save(save_path)
        except Exception as e:
            messagebox.showerror("Lỗi lưu ảnh", f"Không lưu được ảnh:\n{e}")
            return

  
        txt_path = os.path.splitext(save_path)[0] + ".txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
        
                name_to_write = self.last_logo_name if self.last_logo_name else "Unknown"
                f.write(str(name_to_write))
        except Exception as e:
            messagebox.showerror("Lỗi lưu TXT", f"Không lưu được file TXT:\n{e}")
            return

        messagebox.showinfo("Lưu xong", f"Đã lưu ảnh: {save_path}\nVà file tên logo: {txt_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
