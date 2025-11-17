from PyQt5 import QtWidgets, QtGui, QtCore
from logo_detection import *
from query_text import *
from qt_material import apply_stylesheet

from text_detection.text_detection import run_ocr_pipeline

MAX_DISPLAY_W, MAX_DISPLAY_H = 640, 480


def pil_to_qpixmap(pil_img, max_w=MAX_DISPLAY_W, max_h=MAX_DISPLAY_H):
    """Convert PIL Image to QPixmap with optional scaling."""
    img_rgba = pil_img.convert("RGBA")
    data = img_rgba.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, pil_img.width, pil_img.height, QtGui.QImage.Format_RGBA8888)
    pix = QtGui.QPixmap.fromImage(qimg)

    if pil_img.width > max_w or pil_img.height > max_h:
        pix = pix.scaled(max_w, max_h, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                         QtCore.Qt.TransformationMode.SmoothTransformation)
    return pix


def parse_ocr_results_to_string(ocr_results):
    words = []
    for item in ocr_results:
        words.extend(item.split())

    seen = set()
    unique_words = [w for w in words if not (w in seen or seen.add(w))]

    return " ".join(unique_words)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car detection")
        self.setFixedSize(MAX_DISPLAY_W + 400, MAX_DISPLAY_H + 80)

        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Model error", f"Cannot load model:\n{e}")
            sys.exit(1)

        self.last_result_pil = None
        self.last_logo_name = None
        self.last_image_path = None
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(MAX_DISPLAY_W, MAX_DISPLAY_H)
        self.image_label.setStyleSheet("background-color: #111;")

        side_widget = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_widget)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(12)

        btn_choose = QtWidgets.QPushButton("Chọn ảnh")
        btn_choose.clicked.connect(self.choose_image)

        btn_save = QtWidgets.QPushButton("Lưu ảnh & tên (TXT)")
        btn_save.clicked.connect(self.save_result)

        btn_group = QtWidgets.QVBoxLayout()
        btn_group.addWidget(btn_choose)
        btn_group.addWidget(btn_save)
        btn_group.addStretch(1)

        self.info_label = QtWidgets.QLabel("Chưa có ảnh")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.info_label.setWordWrap(True)

        self.info_table = QtWidgets.QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.info_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.info_table.setFixedHeight(200)
        self.info_table.setWordWrap(True)
        self.info_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.info_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.info_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.info_table.horizontalHeader().setStretchLastSection(True)

        side_layout.addWidget(self.info_label)
        side_layout.addWidget(self.info_table)

        self.results_list = QtWidgets.QListWidget()
        self.results_list.setFixedHeight(100)
        self.results_list.setWordWrap(True)
        self.results_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.results_list.itemClicked.connect(self.on_result_item_clicked)

        self.results_list_desc = QtWidgets.QLabel("Kết quả tìm kiếm:")

        side_layout.addWidget(self.results_list_desc)
        side_layout.addWidget(self.results_list)
        side_layout.addLayout(btn_group)
        side_layout.addStretch(2)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)
        main_layout.addWidget(self.image_label)
        side_widget.setFixedWidth(280)
        main_layout.addWidget(side_widget)

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "",
                                                        "Image files (*.jpg *.jpeg *.png *.bmp)")
        if path:
            try:
                self.process_image(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Image processing error:\n{e}")

    def process_image(self, file_path):
        self.last_image_path = file_path

        with open(file_path, 'rb') as f:
            image_bytes = f.read()

        pil_clean, logo_name, conf_text = process_image_logic(self.model, image_bytes)

        self.last_result_pil = pil_clean
        self.last_logo_name = logo_name

        pix = pil_to_qpixmap(pil_clean)
        self.image_label.setPixmap(pix)

        temp_path = "temp_ocr_image.png"
        pil_clean.save(temp_path)
        complete_string = logo_name
        try:
            ocr_results = run_ocr_pipeline(temp_path)
            complete_string += " " + parse_ocr_results_to_string(ocr_results.get('trocr_texts', []))
        except Exception as e:
            print(f"OCR Error: {e}")
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

        try:
            cars_response = find_cars(complete_string, limit=5)
            cars_results = cars_response.get('results', [])
            self.display_search_results(cars_results)
            self.info_label.setText(conf_text)
        except Exception as e:
            self.results_list.clear()
            self.info_table.clearContents()
            self.info_table.setRowCount(0)
            self.info_label.setText(f"{conf_text} — search failed")

    def display_search_results(self, results):
        self.results_list.clear()
        if not results:
            return
        for idx, r in enumerate(results):
            brand = r.get('brand_name') or ""
            score = r.get('score') or 0
            name = r.get("suggestion_text") or "Unknown"
            text = f"{brand} {name} — {score}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, r)
            if idx == 0:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
                item.setBackground(QtGui.QColor('#FFF59D'))
            self.results_list.addItem(item)

        self.results_list.setCurrentRow(0)
        current = self.results_list.currentItem()
        if current:
            self.on_result_item_clicked(current)

    def on_result_item_clicked(self, item):
        """Populate the info table when a result is clicked."""
        r = item.data(QtCore.Qt.UserRole)
        if not r:
            return

        def fmt(val, unit=""):
            return f"{int(val)}{unit}" if (val is not None and str(val) != "nan") else "N/A"

        fields = [
            ("Rank", r.get('rank') if r.get('rank') is not None else "N/A"),
            ("Score", str(r.get('score')) if r.get('score') is not None else "N/A"),
            ("Weight", fmt(r.get('kg'), " kg")),
            ("Length", fmt(r.get('length_mm'), " mm")),
            ("Width", fmt(r.get('width_mm'), " mm")),
            ("Height", fmt(r.get('height_mm'), " mm")),
            ("Brand", r.get('brand_name') or "Unknown"),
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
        self.info_label.setText(f"{r.get('suggestion_text') or 'Unknown'}")

    def save_result(self):
        if self.last_result_pil is None:
            QtWidgets.QMessageBox.information(self, "Lưu", "Chưa có ảnh kết quả để lưu.")
            return
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Lưu ảnh đã làm mờ", "", "JPEG (*.jpg);;PNG (*.png)")
        if not save_path:
            return
        try:
            self.last_result_pil.save(save_path)
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
    apply_stylesheet(app, theme='light_lightgreen.xml', css_file="themes/custom_font.qss", invert_secondary=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
