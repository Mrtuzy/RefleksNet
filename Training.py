import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from norse.torch.module.lif import LIFCell

# --------------------------------------
# YOLO Formatlı Veri İçin Dataset Sınıfı
# --------------------------------------
class YOLOSNNElDataset(Dataset):
    def __init__(self, images_dir, labels_dir, el_class_id=0, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.el_class_id = str(el_class_id)
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = 0
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith(self.el_class_id):
                        label = 1
                        break

        return image, label

# -----------------------
# SNN Model: ReflexNet
# -----------------------
class ReflexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.lif1 = LIFCell()
        self.fc2 = nn.Linear(128, 1)
        self.lif2 = LIFCell()

    def forward(self, x, s1=None, s2=None):
        x = x.view(-1, 28*28)
        z1, s1 = self.lif1(self.fc1(x), s1)
        z2, s2 = self.lif2(self.fc2(z1), s2)
        return z2, (s1, s2)

# -------------------------
# Eğitim Fonksiyonu
# -------------------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Train loss: {total_loss:.4f}")

# -------------------------
# Test Fonksiyonu
# -------------------------
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.float().unsqueeze(1).to(device)
            output, _ = model(data)
            pred = (output > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# ------------------------
# Ana Program
# ------------------------
def main():
    # Dataset yolları (senin yapına göre)
    train_images_dir = "dataset/train/images"
    train_labels_dir = "dataset/train/labels/YOLO"
    test_images_dir = "dataset/test/images"
    test_labels_dir = "dataset/test/labels/YOLO"

    batch_size = 64
    epochs = 5
    el_class_id = 0  # YOLO class ID'si (el için)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    train_dataset = YOLOSNNElDataset(train_images_dir, train_labels_dir, el_class_id=el_class_id, transform=transform)
    test_dataset = YOLOSNNElDataset(test_images_dir, test_labels_dir, el_class_id=el_class_id, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReflexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/reflexnet_trained.pth")
    print("Model kaydedildi -> saved_models/reflexnet_trained.pth")

if __name__ == "__main__":
    main()
