import os
import tkinter as tk
import shutil

from tkinter import filedialog, messagebox, ttk
from Resnet_cover_classifier import load_Resnet_Classifier_model, pdf_to_images, predict_cover, convert_pil_to_opencv
from Serial_scanner import load_Serial_Scanner_model, detect_seri, extract_seri

# Hiển thị thông báo hoàn tất
messagebox.showinfo("Khởi động", "Chương trình đã sẵn sàng!")

# Tiếp tục với phần còn lại của giao diện chính hoặc các tác vụ sau đó

# Biến toàn cục
Root = None
label_result_2 = None
combo_output_folder = None
entry_input= None
output_folder_list = []

OUTPUT_FILE = "output_folder_list.txt"  # File lưu danh sách thư mục OUTPUT


def Load_models():
    """Hàm tải mô hình nhận diện bìa và số seri."""
    resnet_cover_classifier_model_patch = "models/resnet_cover_classifier.pth"
    serial_scanner_model_patch = "models/yolov8_best.pt"
    Resnet_Cover_Classifier_model = load_Resnet_Classifier_model(resnet_cover_classifier_model_patch)
    Serial_Scanner_model = load_Serial_Scanner_model(serial_scanner_model_patch)
    return Resnet_Cover_Classifier_model, Serial_Scanner_model

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
             user_choice = messagebox.askyesno("Xác nhận", f"File {new_filename} đã tồn tại!\nBạn có muốn thay thế không?")
             if not user_choice:
                print(f"⚠ Người dùng chọn bỏ qua file {new_filename}.")
                return  # Bỏ qua file nếu chọn "No"
        shutil.move(file_path, new_path)
        print(f"✅ Đổi tên thành công: {file_path} -> {new_path}")
    except PermissionError:
        print("⚠ Lỗi: Không thể đổi tên file. Hãy kiểm tra xem file có đang mở không.")
    except Exception as e:
        print(f"⚠ Lỗi không xác định: {e}")

def process_input_folder(folder_path, resnet_model, yolo_model, output_folder):
    if not os.path.isdir(folder_path):
        print(f"⚠ Lỗi: Thư mục '{folder_path}' không tồn tại!")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠ Không tìm thấy file PDF nào trong thư mục!")
        return
    print(f"🔄 Đang xử lý {len(pdf_files)} file PDF trong thư mục '{folder_path}'...")

    success_count = 0
    fail_count = 0
    for file_name in pdf_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            process_pdf(file_path, resnet_model, yolo_model, output_folder)
            success_count += 1
        except Exception as e:
            print(f"⚠ Lỗi khi xử lý {file_name}: {e}")
            fail_count += 1

    print(f"✅ Hoàn thành! {success_count} file xử lý thành công, {fail_count} file bị lỗi.")
          
        
def normalize_path(path):
    #"""Chuẩn hóa đường dẫn file, chuyển dấu \ thành / để tránh lỗi hệ điều hành."""
    return os.path.abspath(path).replace("\\", "/")  # Chuyển toàn bộ \ thành /

def save_output_folder(output_folder_path):
    """Lưu đường dẫn OUTPUT vào file, tránh trùng lặp và làm mới danh sách hợp lệ"""
    global output_folder_list
    path = output_folder_path.get().strip()

    if path:
        # Cập nhật danh sách (loại bỏ thư mục không tồn tại)
        refresh_output_folders()
        
        if path not in output_folder_list:
            output_folder_list.append(path)

        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(output_folder_list))
            print("Đã lưu OUTPUT:", path)
            combo_output_folder["values"] = output_folder_list  # Cập nhật ComboBox
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lưu thư mục thất bại!\n{e}")

def load_output_folders():
    """Tải danh sách thư mục OUTPUT từ file và loại bỏ thư mục không còn tồn tại"""
    global output_folder_list
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output_folder_list = f.read().splitlines()
    
    refresh_output_folders()


def refresh_output_folders():
    """Xóa thư mục không tồn tại khỏi danh sách"""
    global output_folder_list
    output_folder_list = [folder for folder in output_folder_list if os.path.exists(folder)]

    # Cập nhật ComboBox
    combo_output_folder["values"] = output_folder_list
    if output_folder_list:
        combo_output_folder.set(output_folder_list[-1])  # Chọn thư mục cuối cùng
    else:
        combo_output_folder.set("")  # Xóa nếu không còn thư mục hợp lệ

    # Ghi lại file để xóa đường dẫn không hợp lệ
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_folder_list))

def select_output_folder(output_folder_path):
    """Chọn thư mục OUTPUT và cập nhật ComboBox"""
    path = filedialog.askdirectory(initialdir=combo_output_folder.get() or "/", title="Chọn thư mục OUTPUT")
    if path:
        output_folder_path.set(path)
        save_output_folder(output_folder_path)
        combo_output_folder.set(path)
        messagebox.showinfo("Thành công", f"Đã chọn thư mục OUTPUT:\n{path}")

def select_input_folder(input_path):
    """Chọn thư mục INPUT"""
    path = filedialog.askdirectory(initialdir="/", title="Chọn thư mục chứa file PDF")  # Luôn mới
    if path:
        input_path.set(path)  # Sử dụng input_path thay vì input_folder_path
        entry_input.delete(0, tk.END)
        entry_input.insert(0, path)

def select_input_PDF_file(input_path):
    global entry_input
    path = filedialog.askopenfilename(initialdir="/", title="Chọn file PDF", filetypes=[("PDF file", "*.pdf")])
    if path:
        input_path.set(path)
        entry_input.delete(0, tk.END)
        entry_input.insert(0, path)

def run_processing(input_path, output_folder_path, resnet_model, yolo_model):
    data_input_path = str(input_path.get()).strip()
    data_output_path = str(output_folder_path.get()).strip()

    if os.path.isdir(data_input_path):
        process_input_folder(data_input_path, resnet_model, yolo_model, data_output_path)
    else:
        process_pdf(data_input_path, resnet_model, yolo_model, data_output_path)

def tkinter_main():
    """Hàm khởi động giao diện Tkinter"""
    global Root, label_result_2, combo_output_folder, entry_input
    resnet_model, yolo_model = Load_models()
    if not resnet_model or not yolo_model:
        print("❌ Lỗi: Chương trình không thể chạy do lỗi tải mô hình.")
        return

    Root = tk.Tk()
    print("Khởi động hoàn tất")
    Root.title("Serial Filename Maker")
    Root.geometry('600x400')

    # Tạo biến
    output_folder_path = tk.StringVar()
    input_path = tk.StringVar()

    # Giao diện chọn thư mục OUTPUT
    tk.Label(Root, text="Chọn thư mục bạn muốn lưu:", font=("Arial", 12, "bold")).pack(pady=5)
    
    frame_output = tk.Frame(Root)
    frame_output.pack(pady=5)

    combo_output_folder = ttk.Combobox(frame_output, textvariable=output_folder_path, width=50, state="readonly")
    combo_output_folder.pack(side=tk.LEFT, padx=5)
    
    btn_Browser1 = tk.Button(frame_output, text="Mở thư mục", command=lambda: select_output_folder(output_folder_path))
    btn_Browser1.pack(side=tk.LEFT)

    # Giao diện chọn thư mục INPUT
    tk.Label(Root, text="Chọn thư mục chứa tệp PDF hoặc chọn tệp PDF mà bạn muốn xử lý:", font=("Arial", 9, "bold")).pack(pady=5)

    frame_input = tk.Frame(Root)
    frame_input.pack(pady=5)
    
   
    entry_input = tk.Entry(frame_input, textvariable=input_path, width=50)
    entry_input.pack(side=tk.LEFT, padx=5)
    
    btn_Browser2 = tk.Button(frame_input, text="Mở thư mục", command=lambda: select_input_folder(input_path))
    btn_Browser2.pack(side=tk.LEFT)
    btn_Browser3 = tk.Button(frame_input, text="Mở tệp", command=lambda: select_input_PDF_file(input_path))
    btn_Browser3.pack(side=tk.LEFT)
    btn_Run = tk.Button(frame_input, text="Tiến hành xử lý", command=lambda: run_processing(input_path, output_folder_path, resnet_model, yolo_model))
    btn_Run.pack(side=tk.LEFT)
    

    # Load danh sách thư mục OUTPUT đã lưu trước đó
    load_output_folders()

    # Bắt đầu vòng lặp giao diện
    Root.mainloop()

if __name__ == "__main__":
    tkinter_main()
