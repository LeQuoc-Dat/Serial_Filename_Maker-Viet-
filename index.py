import os
import keyboard as kb
import shutil
import threading
import time
import sys

from Resnet_cover_classifier import load_Resnet_Classifier_model, pdf_to_images, predict_cover, convert_pil_to_opencv
from Serial_scanner import load_Serial_Scanner_model, detect_seri, extract_seri

exit_flag = threading.Event()  # Cờ để kiểm tra trạng thái thoát

def on_esc():
    """Hàm xử lý khi nhấn ESC để thoát chương trình."""
    exit_flag.set()
    print("\nNhấn Enter để xác nhận thoát chương trình!")

kb.add_hotkey("esc", on_esc)

def Load_models():
    """Hàm tải mô hình nhận diện bìa và số seri."""
    resnet_cover_classifier_model_patch = "models/resnet_cover_classifier.pth"
    serial_scanner_model_patch = "models/yolov8_best.pt"
    Resnet_Cover_Classifier_model = load_Resnet_Classifier_model(resnet_cover_classifier_model_patch)
    Serial_Scanner_model = load_Serial_Scanner_model(serial_scanner_model_patch)
    return Resnet_Cover_Classifier_model, Serial_Scanner_model

def normalize_path(path):
    #"""Chuẩn hóa đường dẫn file, chuyển dấu \ thành / để tránh lỗi hệ điều hành."""
    return os.path.abspath(path).replace("\\", "/")  # Chuyển toàn bộ \ thành /

def process_pdf(file_path, resnet_model, yolo_model, output_folder):
    """Xử lý file PDF: tách ảnh, nhận diện bìa, trích xuất số seri và đổi tên file."""
    file_path = normalize_path(file_path)

    if not os.path.isfile(file_path):
        print(f"⚠ Lỗi: File không tồn tại tại {file_path}")
        return

    List_PDF_page_images = pdf_to_images(file_path)
    if not List_PDF_page_images:
        print("⚠ Lỗi: Không thể trích xuất ảnh từ PDF!")
        return

    Pil_Cover_image = predict_cover(List_PDF_page_images, resnet_model)
    if Pil_Cover_image is None:
        print("⚠ Lỗi: Không tìm thấy ảnh bìa!")
        return

    Numpy_Cover_image = convert_pil_to_opencv(Pil_Cover_image)
    Detected_Serial_image = detect_seri(Numpy_Cover_image, yolo_model)

    Serial_number = extract_seri(Detected_Serial_image).strip()
    if not Serial_number:
        print("⚠ Lỗi: Không thể trích xuất số seri!")
        return

    new_filename = Serial_number + ".pdf"
    os.makedirs(output_folder, exist_ok=True)
    new_path = normalize_path(os.path.join(output_folder, new_filename))

    try:
        if os.path.exists(new_path):
            print(f"⚠ Cảnh báo: File {new_path} đã tồn tại. Đang bỏ qua...")
            return
        shutil.move(file_path, new_path)
        print(f"✅ Đổi tên thành công: {file_path} -> {new_path}")
    except PermissionError:
        print("⚠ Lỗi: Không thể đổi tên file. Hãy kiểm tra xem file có đang mở không.")
    except Exception as e:
        print(f"⚠ Lỗi không xác định: {e}")

def Get_output_folder():
    """Nhập đường dẫn thư mục đầu ra hoặc dùng mặc định."""
    output_folder_path = input("Nhập đường dẫn thư mục bạn muốn lưu (hoặc để trống để dùng thư mục mặc định): ").strip()
    return output_folder_path if output_folder_path else "Processed_files"


def input_thread(input_queue):
    """Luồng nhập dữ liệu liên tục từ bàn phím."""
    last_input = None  
    while not exit_flag.is_set():
        try:
            path = input("Hãy nhập đường dẫn file PDF: ").strip()
            if path and path != last_input:  # Chỉ thêm nếu khác giá trị cũ
                input_queue.append(path)
                last_input = path  # Cập nhật giá trị nhập gần nhất
            time.sleep(0.5)  # Giảm tải CPU
        except EOFError:
            break  
        


def main_loop():
    """Hàm để chạy chương trình chính"""
    print("🚀 Đang khởi động chương trình...")
    resnet_model, yolo_model = Load_models()

    if not resnet_model or not yolo_model:
        print("❌ Lỗi: Chương trình không thể chạy do lỗi tải mô hình.")
        return

    output_folder = Get_output_folder()
    print("<--------- Nhấn phím 'ESC' để dừng chương trình --------->")

    input_queue = []
    thread = threading.Thread(target=input_thread, args=(input_queue,), daemon=True)
    thread.start()

    while not exit_flag.is_set():
        if input_queue:
            file_pdf_path = input_queue.pop(0)
            if not file_pdf_path.lower().endswith(".pdf"):
                print("⚠ Lỗi: Bạn phải nhập một file PDF!")
                continue

            process_pdf(file_pdf_path, resnet_model, yolo_model, output_folder)

        time.sleep(0.2)  # Giảm tải CPU


if __name__ == "__main__":
    main_loop()
