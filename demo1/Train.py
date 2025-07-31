# train_autoencoder.py
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from Mdoel import HandAutoencoder
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = ImageFolder('dataset/', transform=transform)  # Sadece hands/ var
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = HandAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim
for epoch in range(10):
    total_loss = 0
    for img, _ in loader:
        img = img.to(device)
        out = model(img)
        loss = F.mse_loss(out, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# Kaydet
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/autoencoder.pth")
