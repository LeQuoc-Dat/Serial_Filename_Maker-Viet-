import os
import shutil
from ultralytics import YOLO

yaml_path = "Dataset/YOLO_serial_scanner_dataset/yaml"

# --- 1. Huấn luyện mô hình ---
print(" Bắt đầu huấn luyện YOLOv8...")
model = YOLO("yolov8n.pt")  # Dùng YOLOv8 nhỏ nhất (có thể đổi sang yolov8s, yolov8m, yolov8l)
model.train(data=yaml_path, epochs=50, imgsz=640, batch=8)

# --- 2. Lưu mô hình vào thư mục models ---
os.makedirs("models", exist_ok=True)
shutil.copy("runs/detect/train/weights/best.pt", "models/yolov8_best.pt")
print(" Mô hình đã được lưu vào thư mục models/yolov8_best.pt")
