import os
import threading
import tkinter as tk
import shutil

from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from Resnet_cover_classifier import load_Resnet_Classifier_model, pdf_to_images, predict_cover, convert_pil_to_opencv
from Serial_scanner import load_Serial_Scanner_model, detect_seri, extract_seri

OUTPUT_FILE = "output_folder_list.txt"  # File lưu danh sách thư mục đích
LOG_FILE = "processing_log.txt" #File ghi lại nhật ký xử lý
TREE_VIEW_FILE = "processing_file.txt" # File lưu danh sách các file đã xử lý và trạng thái 
LOG_LIMIT = 70  #Giới hạn số dòng trạng thái được lưu trong file log để tránh file quá lớn


'''Chỉnh sửa kích thước giao diện'''
APP_WIDTH  = 1000 
APP_HEIGHT = 600 


def normalize_path(path):
    """Chuẩn hóa đường dẫn file theo hệ điều hành."""
    return os.path.abspath(path)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Serial Filename Maker")
        self.geometry(str(APP_WIDTH)+"x"+str(APP_HEIGHT))

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
        """Tạo giao diện Tkinter với các tính năng bổ sung"""
        # Frame chọn thư mục OUTPUT
        frame_output = tk.Frame(self)
        frame_output.pack(pady=5, fill=tk.X, padx=10)

        tk.Label(frame_output, text="Chọn thư mục lưu kết quả:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.output_folder_path = tk.StringVar()
        self.combo_output_folder = ttk.Combobox(frame_output, textvariable=self.output_folder_path, width=50, state="readonly")
        self.combo_output_folder.pack(side=tk.LEFT, padx=5)
        btn_output = tk.Button(frame_output, text="Mở thư mục", command=self.select_output_folder)
        btn_output.pack(side=tk.LEFT, padx=5)

        # Frame chọn file/thư mục INPUT
        frame_input = tk.Frame(self)
        frame_input.pack(pady=5, fill=tk.X, padx=10)

        tk.Label(frame_input, text="Chọn thư mục chứa tệp PDF hoặc tệp PDF để xử lý:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.input_path = tk.StringVar()
        self.entry_input = tk.Entry(frame_input, textvariable=self.input_path, width=50)
        self.entry_input.pack(side=tk.LEFT, padx=5)
        btn_input_folder = tk.Button(frame_input, text="Mở thư mục", command=self.select_input_folder)
        btn_input_folder.pack(side=tk.LEFT, padx=5)
        btn_input_file = tk.Button(frame_input, text="Mở tệp", command=self.select_input_pdf)
        btn_input_file.pack(side=tk.LEFT, padx=5)
        btn_run = tk.Button(frame_input, text="Tiến hành xử lý", command=self.run_processing_thread)
        btn_run.pack(side=tk.LEFT, padx=5)

        # Thanh tiến trình
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10, padx=10)

        # Khung Treeview để hiển thị danh sách file đã xử lý
        frame_tree = tk.Frame(self)
        frame_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.tree = ttk.Treeview(frame_tree, columns=("File", "Trạng thái","Ngày thực hiện"), show="headings")
        self.tree.heading("File", text="Tên tệp")
        self.tree.heading("Trạng thái", text="Trạng thái")
        self.tree.heading("Ngày thực hiện", text="Ngày thực hiện")
        self.tree.column("File", width=50)
        self.tree.column("Trạng thái", width=50)
        self.tree.column("Ngày thực hiện", width=50)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Khung Text để hiển thị log
        frame_log = tk.Frame(self)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(frame_log, text="Tình trạng:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.txt_log = tk.Text(frame_log, height=8)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def select_output_folder(self):
        """Chọn thư mục OUTPUT và cập nhật ComboBox"""
        path = filedialog.askdirectory(title="Chọn thư mục OUTPUT")
        if path:
            self.output_folder_path.set(path)
            self.save_output_folder(path)
            self.combo_output_folder.set(path)
            self.log_message(f"Đã chọn thư mục OUTPUT: {path}")

    def select_input_folder(self):
        """Chọn thư mục INPUT"""
        path = filedialog.askdirectory(title="Chọn thư mục chứa file PDF")
        if path:
            self.input_path.set(path)
            self.log_message(f"Đã chọn thư mục INPUT: {path}")
        self.progress["value"] = 0

    def select_input_pdf(self):
        """Chọn file PDF"""
        path = filedialog.askopenfilename(title="Chọn file PDF", filetypes=[("PDF file", "*.pdf")])
        if path:
            self.input_path.set(path)
            self.log_message(f"Đã chọn file PDF: {path}")
        self.progress["value"] = 0

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
        """Tiến hành xử lý file PDF hoặc thư mục chứa file PDF"""
        input_path = self.input_path.get().strip()
        output_folder = self.output_folder_path.get().strip()

        if not os.path.exists(input_path):
            messagebox.showerror("Lỗi", "Đường dẫn đầu vào không tồn tại.")
            return

        # Nếu là thư mục thì xử lý tất cả file PDF (có thể mở rộng hỗ trợ định dạng khác)
        if os.path.isdir(input_path):
            pdf_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                         if f.lower().endswith(".pdf")]
            total = len(pdf_files)
            if total == 0:
                messagebox.showerror("Lỗi", "Không tìm thấy file PDF nào trong thư mục.")
                return
            self.progress["maximum"] = total
            for idx, file_path in enumerate(pdf_files, start=1):
                self.process_pdf(file_path, output_folder)
                self.progress["value"] = idx
        else:
            # Nếu là file đơn lẻ
            self.progress["maximum"] = 1
            self.process_pdf(input_path, output_folder)
            self.progress["value"] = 1

        self.log_message("Quá trình xử lý hoàn tất!")
        self.input_path.set("")

    def process_pdf(self, file_path, output_folder):
        """Xử lý file PDF"""
        file_path = normalize_path(file_path)
        file_name = os.path.basename(file_path)
        self.add_tree_item(file_name, "Đang xử lý...")
        self.log_message(f"Đang xử lý: {file_name}")

        if not os.path.isfile(file_path):
            self.log_message(f"[Lỗi] File không tồn tại: {file_path}")
            self.update_status(file_name, "Lỗi: File không tồn tại")
            return

        images = pdf_to_images(file_path)
        if not images:
            self.log_message(f"[Lỗi] Không thể trích xuất ảnh từ: {file_name}")
            self.update_status(file_name, "Lỗi: Trích xuất ảnh thất bại")
            return

        cover_image = predict_cover(images, self.resnet_model)
        if cover_image is None:
            self.log_message(f"[Lỗi] Không tìm thấy ảnh bìa trong: {file_name}")
            self.update_status(file_name, "Lỗi: Ảnh bìa không tìm thấy")
            return

        np_cover_image = convert_pil_to_opencv(cover_image)
        detected_seri = detect_seri(np_cover_image, self.yolo_model)

        serial_number = extract_seri(detected_seri).strip()
        if not serial_number:
            self.log_message(f"[Lỗi] Không thể trích xuất số seri từ: {file_name}")
            self.update_status(file_name, "Lỗi: Số seri không tìm thấy")
            return

        new_filename = serial_number + ".pdf"
        os.makedirs(output_folder, exist_ok=True)
        new_path = normalize_path(os.path.join(output_folder, new_filename))

        if os.path.exists(new_path):
            user_choice = messagebox.askyesno("Xác nhận", f"File {new_filename} đã tồn tại! Bạn có muốn thay thế không?")
            if not user_choice:
                self.log_message(f"Đã bỏ qua {file_name} vì tồn tại file trùng tên.")
                self.update_status(file_name, "Bỏ qua")
                return

        try:
            shutil.move(file_path, new_path)
            self.log_message(f"Đổi tên thành công: {new_filename}")
            self.update_status(file_name, "Thành công")
        except Exception as e:
            self.log_message(f"[Lỗi] Khi đổi tên {file_name}: {e}")
            self.update_status(file_name, f"Lỗi: {e}")

    def log_message(self, message):
        """Ghi thông điệp vào khung log và console"""
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        print(message)
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        self.trim_log_file(LOG_FILE)

    def add_tree_item(self, file_name, status):
        """Thêm mục vào Treeview danh sách file đã xử lý"""
        self.tree.insert("", tk.END, values=(file_name, status))
        self.trim_log_file(TREE_VIEW_FILE)
    
    def update_status(self, file_name, new_status):
        """Cập nhật trạng thái của file trong Treeview và file lưu trữ"""
    
        # Cập nhật trạng thái trong Treeview
        for item in self.tree.get_children():
            current_filename = self.tree.item(item, "values")[0]  # Lấy tên tệp trong cột đầu tiên
        
        # Kiểm tra nếu tên tệp là file_name và cập nhật trạng thái
            current_date = self.get_current_date()
            if current_filename == file_name:
                self.tree.item(item, values=(file_name, new_status, current_date))  # Cập nhật trạng thái và ngày thực hiện
                
        with open(TREE_VIEW_FILE, "a", encoding="utf-8") as f:
           f.write(f"{file_name}|{new_status}|{current_date}\n")
        #Giới hạn số dòng trong file
        self.trim_log_file(TREE_VIEW_FILE)
        
        
    def trim_log_file(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if len(lines) > LOG_LIMIT:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-LOG_LIMIT:])
                
    def get_current_date(self):
        """Lấy ngày tháng hiện tại theo định dạng YYYY-MM-DD"""
        return datetime.now().strftime("%Y-%m-%d")

if __name__ == "__main__":
    app = App()
    app.mainloop()