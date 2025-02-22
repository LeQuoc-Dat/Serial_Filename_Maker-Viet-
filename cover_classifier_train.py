import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# Định nghĩa đường dẫn dữ liệu
DATA_DIR = "Dataset/Resnet_cover_classifier_dataset"

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset tuỳ chỉnh
class BookCoverDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Duyệt qua từng thư mục con (BookX)
        for book_folder in os.listdir(root_dir):
            book_path = os.path.join(root_dir, book_folder)
            input_path = os.path.join(book_path, "input")
            output_path = os.path.join(book_path, "output")
            
            # Thêm các trang sách (label 0)
            if os.path.exists(input_path):
                for img_name in os.listdir(input_path):
                    img_path = os.path.join(input_path, img_name)
                    self.data.append((img_path, 0))
            
            # Thêm ảnh bìa (label 1)
            if os.path.exists(output_path):
                for img_name in os.listdir(output_path):
                    img_path = os.path.join(output_path, img_name)
                    self.data.append((img_path, 1))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Tạo dataset và dataloader
dataset = BookCoverDataset(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Khởi tạo mô hình ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 lớp (không phải bìa / bìa sách)
model = model.to(device)

# Cấu hình huấn luyện
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Huấn luyện mô hình
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {accuracy:.2f}%")

# Lưu mô hình
torch.save(model.state_dict(), "models/resnet_cover_classifier.pth")
print("Huấn luyện hoàn tất! Mô hình đã được lưu.")
