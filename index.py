from Resnet_cover_classifier import load_Resnet_Classifier_model, pdf_to_images, predict_cover, convert_pil_to_opencv
from Serial_scanner import load_Serial_Scanner_model, detect_seri, extract_seri
import os

#---1. Khởi tạo hai mô hình cho chương trình ---
def LoadModel():
    resnet_cover_classifier_model_patch="models/resnet_cover_classifier.pth"
    serial_scanner_model_patch="models/yolov8_best.pt"
    Resnet_Cover_Classifier_model = load_Resnet_Classifier_model(resnet_cover_classifier_model_patch)
    Serial_Scanner_model = load_Serial_Scanner_model(serial_scanner_model_patch)
    return Resnet_Cover_Classifier_model, Serial_Scanner_model
    

# --- 2. Xử lý PDF ---
def process_pdf(file_path, resnet_model, yolo_model):
    if not os.path.exists(file_path):
        print("Lỗi: Đường dẫn file không tồn tại!")
        return

    List_PDF_page_images = pdf_to_images(file_path)
    if not List_PDF_page_images:
        print("Lỗi: Không thể trích xuất ảnh từ PDF!")
        return

    Pil_Cover_image = predict_cover(List_PDF_page_images, resnet_model)
    if Pil_Cover_image is None:
        print("Lỗi: Không tìm thấy ảnh bìa!")
        return

    Numpy_Cover_image = convert_pil_to_opencv(Pil_Cover_image)
    Detected_Serial_image = detect_seri(Numpy_Cover_image, yolo_model)

    File_with_Serial_number_name = extract_seri(Detected_Serial_image).strip()
    if not File_with_Serial_number_name:
        print("Lỗi: Không thể trích xuất số seri!")
        return

    # Định dạng lại tên file (loại bỏ ký tự đặc biệt nếu có)
    File_with_Serial_number_name = "".join(
        c for c in File_with_Serial_number_name if c.isalnum() or c in (" ", "_", "-")
    ) + ".pdf"

    new_path = os.path.join(os.path.dirname(file_path), File_with_Serial_number_name)

    try:
        os.rename(file_path, new_path)
        print(f"Đổi tên thành công: {file_path} -> {new_path}")
    except Exception as e:
        print(f"Lỗi khi đổi tên file: {e}")
        
if __name__ == "__main__":
    print("Đang khởi động chương trình...")
    resnet_model, yolo_model = LoadModel()

    if resnet_model and yolo_model:
        file_PDF_path = input("Hãy nhập đường dẫn của file PDF: ").strip()
        process_pdf(file_PDF_path, resnet_model, yolo_model)
    else:
        print("Lỗi: Chương trình không thể chạy do lỗi tải mô hình.")
    
    


   
    
   
    
    
