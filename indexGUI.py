print("Đang khởi động chương trình...")
import os
from tkinter import *
from tkinter import filedialog


# Khởi tạo cửa sổ Tkinter
Root = Tk()
print("Khởi động hoàn tất")
Root.title("Serial Filename Maker")
Root.geometry('500x600')

# Biến toàn cục để lưu đường dẫn thư mục
selected_folder = ""

def Open_Folder():
    global selected_folder  # Dùng biến toàn cục để lưu đường dẫn
    path = filedialog.askdirectory(title="Chọn thư mục")  # Mở hộp thoại chọn thư mục
    if path:  
        selected_folder = path  
        folder_path.set(path)  
       

# Biến Tkinter StringVar để lưu đường dẫn
folder_path = StringVar()

# Nút chọn thư mục
btn_Browser = Button(Root, text="Chọn thư mục", command=Open_Folder)
btn_Browser.pack(pady=10)

# Nhãn hiển thị đường dẫn thư mục đã chọn
label_result = Label(Root, text="", wraplength=400)
label_result.pack(pady=10)

# Bắt đầu vòng lặp giao diện
Root.mainloop()
# Sau khi đóng giao diện, biến selected_folder vẫn giữ giá trị đã chọn
print("Biến lưu thư mục sau khi thoát Tkinter:", selected_folder)
