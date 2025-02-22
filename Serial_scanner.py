import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np


# --- 1. Cấu hình Tesseract OCR ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --- 2. Tải mô hình YOLO ---
def load_Serial_Scanner_model(model_path):
    return YOLO(model_path)


# --- 3. Xác định vị trí số seri ---
def detect_seri(image, model):
    if image is None:
        print("Không thể đọc ảnh!")
        return None

    results = model(image)  # Chạy mô hình YOLO trên ảnh

    for result in results:
        boxes = result.boxes.xyxy  # Lấy danh sách bounding box

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # Lấy tọa độ hộp
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Hộp màu vàng
    
    return image

# --- 4. Trích xuất số seri từ ảnh đã nhận diện ---
def extract_seri(image):
    if image is None:
        return "Không có ảnh để nhận diện số seri!"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Phạm vi màu vàng trong không gian HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Tạo mask để lọc vùng màu vàng
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Tìm contours của vùng màu vàng
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Xác định contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    serial_text = "Không tìm thấy vùng màu vàng!"

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Cắt vùng bên trong hộp
        roi = image[y:y+h, x:x+w]

        # Chuyển sang ảnh xám
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Trích xuất số seri bằng OCR
        serial_text = pytesseract.image_to_string(gray_roi, config="--psm 6").strip()

    return serial_text

# --- 5. Chạy chương trình ---
if __name__== "__main__": 
   model_path = "models/yolov8_best.pt"
   model = load_Serial_Scanner_model(model_path)

