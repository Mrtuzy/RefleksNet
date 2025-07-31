import os
from PIL import Image
import torch
import torch.nn as nn
from norse.torch import LIFParameters
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from norse.torch.module.lif import LIFCell

# ----------------------------
# Dataset Sınıfı
# ----------------------------
class HandsVsNonHandsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # El içeren örnekler (label = 1.0)
        hand_dir = os.path.join(root_dir, "hands")
        for fname in os.listdir(hand_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(hand_dir, fname), 1.0))

        # El içermeyen örnekler (label = 0.0)
        nonhand_dir = os.path.join(root_dir, "nonhands")
        for fname in os.listdir(nonhand_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(nonhand_dir, fname), 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ----------------------------
# ReflexNet SNN Modeli
# ----------------------------
class ReflexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

        # LIF hücreleri, threshold 0.2 olarak ayarlandı
        self.lif1 = LIFCell(p=LIFParameters(v_th=0.2))
        self.lif2 = LIFCell(p=LIFParameters(v_th=0.2))

    def forward(self, x, s1=None, s2=None):
        x = x.view(-1, 28 * 28)
        h1 = self.fc1(x)
        z1, s1 = self.lif1(h1, s1)
        h2 = self.fc2(z1)
        z2, s2 = self.lif2(h2, s2)

        # print(
        #     f"[Aktivasyonlar] fc1: {h1.mean():.4f}, lif1: {z1.mean():.4f}, fc2: {h2.mean():.4f}, lif2: {z2.mean():.4f}")
        return z2, (s1, s2)

# ----------------------------
# Eğitim Fonksiyonu
# ----------------------------
def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.unsqueeze(1).to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        probs = torch.sigmoid(output).detach().cpu().numpy()
        print(f"[Epoch {epoch} | Batch {batch_idx+1}] Loss: {loss.item():.4f} | Prob Mean: {probs.mean():.4f}")

    print(f"🔁 Epoch {epoch} tamamlandı | Toplam Kayıp: {total_loss:.4f}\n")

# ----------------------------
# Test Fonksiyonu
# ----------------------------
def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    print("[🧪 TEST SONUÇLARI]")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.unsqueeze(1).to(device)
            output, _ = model(data)
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    print(f"🎯 Test Doğruluğu: {acc:.2f}%")
    return acc

# ----------------------------
# Ana Eğitim Döngüsü
# ----------------------------
def main():
    dataset_path = "../dataset"
    batch_size = 32
    epochs = 10

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 10.0)
    ])

    dataset = HandsVsNonHandsDataset(dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReflexNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, test_loader)

    os.makedirs("../demo1/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "../demo1/saved_models/reflexnet_hands_vs_nonhands.pth")
    print("✅ Model kaydedildi: saved_models/reflexnet_hands_vs_nonhands.pth")

if __name__ == "__main__":
    main()
