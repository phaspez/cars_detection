# python
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from logo_detection import *
from query_cars_info import *

from qt_material import apply_stylesheet

MAX_DISPLAY_W, MAX_DISPLAY_H = 640, 480

def cv2_to_qpixmap(bgr_img, max_w=MAX_DISPLAY_W, max_h=MAX_DISPLAY_H):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    bytes_per_line = 3 * w
    qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg)
    if w > max_w or h > max_h:
        pix = pix.scaled(max_w, max_h, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
    return pix

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car detection")
        # Make room for a right-hand vertical control panel
        self.setFixedSize(MAX_DISPLAY_W + 400, MAX_DISPLAY_H + 80)

        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Model error", f"Cannot load model:\n{e}")
            sys.exit(1)

        self.last_result_bgr = None
        self.last_logo_name = None
        self.last_image_path = None

        # Fonts: attempt to use Inter; if not installed system will fallback
        self.app_font = QtGui.QFont("Inter", 10)
        self.header_font = QtGui.QFont("Inter", 12, QtGui.QFont.Weight.Bold)

        # Image display (left)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(MAX_DISPLAY_W, MAX_DISPLAY_H)
        self.image_label.setStyleSheet("background-color: #111;")
        self.image_label.setFont(self.app_font)

        # Side panel (right) with vertical layout
        side_widget = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_widget)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(12)

        btn_choose = QtWidgets.QPushButton("Chọn ảnh")
        btn_choose.setFont(self.app_font)
        btn_choose.clicked.connect(self.choose_image)

        btn_save = QtWidgets.QPushButton("Lưu ảnh & tên (TXT)")
        btn_save.setFont(self.app_font)
        btn_save.clicked.connect(self.save_result)

        # Vertical group for buttons (stacked)
        btn_group = QtWidgets.QVBoxLayout()
        btn_group.addWidget(btn_choose)
        btn_group.addWidget(btn_save)
        btn_group.addStretch(1)

        # Info label
        self.info_label = QtWidgets.QLabel("Chưa có ảnh")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setFont(self.header_font)

        self.info_table = QtWidgets.QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.info_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.info_table.setFont(self.app_font)
        self.info_table.setFixedHeight(200)
        self.info_table.horizontalHeader().setStretchLastSection(True)
        # add label and table to the side layout (replace the previous single info_label placement)
        side_layout.addWidget(self.info_label)
        side_layout.addWidget(self.info_table)

        self.results_list = QtWidgets.QListWidget()
        self.results_list.setFont(self.app_font)
        self.results_list.setFixedHeight(100)
        self.results_list.itemClicked.connect(self.on_result_item_clicked)
        side_layout.addWidget(self.results_list)

        # Add widgets to side layout
        side_layout.addLayout(btn_group)
        side_layout.addWidget(self.info_label)
        side_layout.addStretch(2)

        # Main horizontal layout: image on left, side panel on right
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)
        main_layout.addWidget(self.image_label)
        side_widget.setFixedWidth(280)
        main_layout.addWidget(side_widget)

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image files (*.jpg *.jpeg *.png *.bmp)")
        if path:
            try:
                self.process_image(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Image processing error:\n{e}")

    def process_image(self, file_path):
        img_arr = np.fromfile(file_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot read image. Check path/format.")
        self.last_image_path = file_path

        results = self.model.predict(source=file_path, conf=CONF_THRESH, iou=IOU_THRESH, save=False)
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

        if len(boxes) == 0:
            self.info_label.setText("Không phát hiện đối tượng nào")
            self.last_result_bgr = img_bgr.copy()
            self.last_logo_name = "Unknown"
            self.info_table.clearContents()
            self.info_table.setRowCount(0)
            pix = cv2_to_qpixmap(self.last_result_bgr)
            self.image_label.setPixmap(pix)
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

        if best_logo_name != "Unknown":
            try:
                cars_response = find_cars(best_logo_name, limit=5)
                cars_results = cars_response.get('results', [])
                self.display_search_results(cars_results)
            except Exception as e:
                self.results_list.clear()
                self.info_table.clearContents()
                self.info_table.setRowCount(0)
                self.info_label.setText(f"{best_conf_text} — search failed")
        else:
            self.results_list.clear()
            self.info_table.clearContents()
            self.info_table.setRowCount(0)

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

        self.last_result_bgr = img_proc
        self.last_logo_name = best_logo_name
        self.info_label.setText(f"Logo tốt nhất: {best_conf_text}")
        pix = cv2_to_qpixmap(img_proc)
        self.image_label.setPixmap(pix)

    def display_search_results(self, results):
        """Populate the QListWidget with search results, highlight the top one and show its stats."""
        self.results_list.clear()
        if not results:
            return
        for idx, r in enumerate(results):
            brand = r.get('brand_name') or ""
            engine = r.get('engine_name') or ""
            score = r.get('score') or 0
            text = f"{brand} {engine} — {score}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, r)
            if idx == 0:
                # highlight top one
                f = item.font()
                f.setBold(True)
                item.setFont(f)
                item.setBackground(QtGui.QColor('#FFF59D'))  # pale yellow
            self.results_list.addItem(item)

        # select top item and update info label with its stats
        self.results_list.setCurrentRow(0)
        current = self.results_list.currentItem()
        if current:
            # reuse the click handler to populate info label
            self.on_result_item_clicked(current)

    def on_result_item_clicked(self, item):
        """Populate the info table when a result is clicked."""
        r = item.data(QtCore.Qt.UserRole)
        if not r:
            return

        def fmt(val, unit=""):
            return f"{int(val)}{unit}" if (val is not None and str(val) != "nan") else "N/A"

        fields = [
            ("Selected", r.get('suggestion_text') or "Unknown"),
            ("Score", str(r.get('score')) if r.get('score') is not None else "N/A"),
            ("Brand", r.get('brand_name') or "Unknown"),
            ("Engine", r.get('engine_name') or "Unknown"),
            ("Length", fmt(r.get('length_mm'), " mm")),
            ("Width", fmt(r.get('width_mm'), " mm")),
            ("Height", fmt(r.get('height_mm'), " mm")),
            ("Weight", fmt(r.get('kg'), " kg")),
            ("Rank", r.get('rank') if r.get('rank') is not None else "N/A"),
        ]

        self.info_table.clearContents()
        self.info_table.setRowCount(len(fields))
        for row, (key, val) in enumerate(fields):
            key_item = QtWidgets.QTableWidgetItem(str(key))
            val_item = QtWidgets.QTableWidgetItem(str(val))
            key_item.setFlags(key_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            val_item.setFlags(val_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.info_table.setItem(row, 0, key_item)
            self.info_table.setItem(row, 1, val_item)

        self.info_table.resizeColumnsToContents()
        # keep the small header label for status
        self.info_label.setText(f"Selected: {r.get('suggestion_text') or 'Unknown'}")

    def save_result(self):
        if self.last_result_bgr is None:
            QtWidgets.QMessageBox.information(self, "Lưu", "Chưa có ảnh kết quả để lưu.")
            return
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Lưu ảnh đã làm mờ", "", "JPEG (*.jpg);;PNG (*.png)")
        if not save_path:
            return
        try:
            # write via OpenCV to handle unicode paths on Windows
            ext = os.path.splitext(save_path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                cv2.imencode('.jpg', self.last_result_bgr)[1].tofile(save_path)
            else:
                cv2.imencode('.png', self.last_result_bgr)[1].tofile(save_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi lưu ảnh", f"Cannot save image:\n{e}")
            return

        txt_path = os.path.splitext(save_path)[0] + ".txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                name_to_write = self.last_logo_name if self.last_logo_name else "Unknown"
                f.write(str(name_to_write))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi lưu TXT", f"Cannot save TXT:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Lưu xong", f"Saved image: {save_path}\nAnd logo name file: {txt_path}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # with open("material_dark.qss", "r") as f:
    #     app.setStyleSheet(f.read())

    apply_stylesheet(app, theme='dark_teal.xml')

    font = QtGui.QFont()
    font.setFamilies(["JetBrains Mono", "Consolas", "Monaco", "Courier New"])
    font.setPointSize(10)
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
