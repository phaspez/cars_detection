from ultralytics import YOLO
import cv2

# Load mô hình
model = YOLO("best.pt")

# Dự đoán
results = model("C200 (16).jpg", conf=0.1)  # hạ conf xuống 0.3 cho chắc

# Hiển thị kết quả bằng OpenCV
for r in results:
    im_array = r.plot()  # ảnh có bbox
    cv2.imshow("Prediction", im_array)
    cv2.waitKey(0)       # giữ cửa sổ, nhấn phím bất kỳ để đóng
    cv2.destroyAllWindows()
