import os
import threading
import tkinter as tk
import shutil

from tkinter import filedialog, messagebox, ttk
from Resnet_cover_classifier import load_Resnet_Classifier_model, pdf_to_images, predict_cover, convert_pil_to_opencv
from Serial_scanner import load_Serial_Scanner_model, detect_seri, extract_seri

OUTPUT_FILE = "output_folder_list.txt"  # File lưu danh sách thư mục OUTPUT

def normalize_path(path):
    """Chuẩn hóa đường dẫn file theo hệ điều hành."""
    return os.path.abspath(path)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Serial Filename Maker")
        self.geometry("600x400")

        # Tải mô hình AI
        self.resnet_model, self.yolo_model = self.load_models()
        if not self.resnet_model or not self.yolo_model:
            messagebox.showerror("Lỗi", "Không thể tải mô hình, chương trình sẽ thoát.")
            self.destroy()
            return

        self.output_folder_list = []
        self.create_widgets()
        self.load_output_folders()

    def load_models(self):
        """Tải mô hình nhận diện bìa và số seri."""
        resnet_path = "models/resnet_cover_classifier.pth"
        yolo_path = "models/yolov8_best.pt"
        return load_Resnet_Classifier_model(resnet_path), load_Serial_Scanner_model(yolo_path)

    def create_widgets(self):
        """Tạo giao diện Tkinter"""
        # Chọn thư mục OUTPUT
        tk.Label(self, text="Chọn thư mục bạn muốn lưu:", font=("Arial", 12, "bold")).pack(pady=5)

        frame_output = tk.Frame(self)
        frame_output.pack(pady=5)

        self.output_folder_path = tk.StringVar()
        self.combo_output_folder = ttk.Combobox(frame_output, textvariable=self.output_folder_path, width=50, state="readonly")
        self.combo_output_folder.pack(side=tk.LEFT, padx=5)

        btn_output = tk.Button(frame_output, text="Mở thư mục", command=self.select_output_folder)
        btn_output.pack(side=tk.LEFT)

        # Chọn file PDF hoặc thư mục chứa PDF
        tk.Label(self, text="Chọn thư mục chứa tệp PDF hoặc chọn tệp PDF để xử lý:", font=("Arial", 9, "bold")).pack(pady=5)

        frame_input = tk.Frame(self)
        frame_input.pack(pady=5)

        self.input_path = tk.StringVar()
        self.entry_input = tk.Entry(frame_input, textvariable=self.input_path, width=50)
        self.entry_input.pack(side=tk.LEFT, padx=5)

        btn_input_folder = tk.Button(frame_input, text="Mở thư mục", command=self.select_input_folder)
        btn_input_folder.pack(side=tk.LEFT)

        btn_input_file = tk.Button(frame_input, text="Mở tệp", command=self.select_input_pdf)
        btn_input_file.pack(side=tk.LEFT)

        btn_run = tk.Button(frame_input, text="Tiến hành xử lý", command=self.run_processing_thread)
        btn_run.pack(side=tk.LEFT)

    def select_output_folder(self):
        """Chọn thư mục OUTPUT và cập nhật ComboBox"""
        path = filedialog.askdirectory(title="Chọn thư mục OUTPUT")
        if path:
            self.output_folder_path.set(path)
            self.save_output_folder(path)
            self.combo_output_folder.set(path)

    def select_input_folder(self):
        """Chọn thư mục INPUT"""
        path = filedialog.askdirectory(title="Chọn thư mục chứa file PDF")
        if path:
            self.input_path.set(path)

    def select_input_pdf(self):
        """Chọn file PDF"""
        path = filedialog.askopenfilename(title="Chọn file PDF", filetypes=[("PDF file", "*.pdf")])
        if path:
            self.input_path.set(path)

    def save_output_folder(self, path):
        """Lưu thư mục OUTPUT vào danh sách"""
        if path and path not in self.output_folder_list:
            self.output_folder_list.append(path)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(self.output_folder_list))
            self.combo_output_folder["values"] = self.output_folder_list

    def load_output_folders(self):
        """Tải danh sách thư mục OUTPUT đã lưu"""
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                self.output_folder_list = [folder for folder in f.read().splitlines() if os.path.exists(folder)]
        self.combo_output_folder["values"] = self.output_folder_list
        if self.output_folder_list:
            self.combo_output_folder.set(self.output_folder_list[-1])

    def run_processing_thread(self):
        """Chạy xử lý file trên luồng riêng để tránh đơ giao diện"""
        thread = threading.Thread(target=self.run_processing)
        thread.start()

    def run_processing(self):
        """Tiến hành xử lý file PDF"""
        input_path = self.input_path.get().strip()
        output_folder = self.output_folder_path.get().strip()

        if not os.path.exists(input_path):
            messagebox.showerror("Lỗi", "Đường dẫn đầu vào không tồn tại.")
            return

        if os.path.isdir(input_path):
            self.process_input_folder(input_path, output_folder)
        else:
            self.process_pdf(input_path, output_folder)
            
        self.input_path.set("")

    def process_pdf(self, file_path, output_folder):
        """Xử lý file PDF"""
        file_path = normalize_path(file_path)
        if not os.path.isfile(file_path):
            messagebox.showerror("Lỗi", f"File không tồn tại: {file_path}")
            return

        images = pdf_to_images(file_path)
        if not images:
            messagebox.showerror("Lỗi", "Không thể trích xuất ảnh từ PDF.")
            return

        cover_image = predict_cover(images, self.resnet_model)
        if cover_image is None:
            messagebox.showerror("Lỗi", "Không tìm thấy ảnh bìa.")
            return

        np_cover_image = convert_pil_to_opencv(cover_image)
        detected_seri = detect_seri(np_cover_image, self.yolo_model)

        serial_number = extract_seri(detected_seri).strip()
        if not serial_number:
            messagebox.showerror("Lỗi", "Không thể trích xuất số seri.")
            return

        new_filename = serial_number + ".pdf"
        os.makedirs(output_folder, exist_ok=True)
        new_path = normalize_path(os.path.join(output_folder, new_filename))

        if os.path.exists(new_path):
            user_choice = messagebox.askyesno("Xác nhận", f"File {new_filename} đã tồn tại! Bạn có muốn thay thế không?")
            if not user_choice:
                return

        try:
            shutil.move(file_path, new_path)
            messagebox.showinfo("Thành công", f"Đổi tên thành công: {new_filename}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi đổi tên file: {e}")

    def process_input_folder(self, folder_path, output_folder):
        """Xử lý tất cả file PDF trong thư mục"""
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            messagebox.showerror("Lỗi", "Không tìm thấy file PDF nào.")
            return

        for file_name in pdf_files:
            self.process_pdf(os.path.join(folder_path, file_name), output_folder)

if __name__ == "__main__":
    app = App()
    app.mainloop()
