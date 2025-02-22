import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import fitz  # PyMuPDF
import cv2 
import numpy as np


# --- 1. TẢI MÔ HÌNH RESNET ---
def load_Resnet_Classifier_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)  # Phân loại 2 lớp (bìa, nội dung)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

    model = model.to(device)
    model.eval()
    return model


# --- 2. HÀM TIỀN XỬ LÝ ẢNH ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# --- 3. CHUYỂN PDF THÀNH ẢNH ---
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)  # Lưu trực tiếp ảnh vào list (không lưu file)
    return images if images else None


# --- 4. DỰ ĐOÁN ẢNH BÌA ---
def predict_cover(images, model):
    if not images:
        print("Không có ảnh để dự đoán!")
        return None
    transform = get_transform() 
    best_score = -1
    cover_image = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for img in images:
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            score = torch.softmax(output, dim=1)[0, 1].item()  # Xác suất ảnh là bìa
        
        if score > best_score:
            best_score = score
            cover_image = img  # Lưu ảnh có điểm số cao nhất
    
    return cover_image


# --- 5. Chuyển đổi ảnh PIL -> NumPy để sử dụng OpenCV ---
def convert_pil_to_opencv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# --- 6. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
   model_path = "models/resnet_cover_classifier.pth"
   model = load_Resnet_Classifier_model(model_path)


