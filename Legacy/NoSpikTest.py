import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import time

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model tanımı (eğitimdeki ile aynı)
class RefleksNet(nn.Module):
    def __init__(self):
        super(RefleksNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 16, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Modeli yükle
model = RefleksNet().to(device)
model.load_state_dict(torch.load('../saved_models/refleksnet_hand.pth', map_location=device))
model.eval()

# Görüntü dönüşümü
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Kamera başlat
cap = cv2.VideoCapture(0)
print("Webcam başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()

    elapsed_ms = (time.time() - start_time) * 1000

    label = "EL VAR ✅" if prob > 0.5 else "EL YOK ❌"
    cv2.putText(frame, f"{label} (p={prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prob > 0.5 else (0, 0, 255), 2)
    cv2.putText(frame, f"Süre: {elapsed_ms} ms", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("RefleksNet - El Algılama", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()