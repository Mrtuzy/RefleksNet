# test_parallel.py
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from Mdoel import HandAutoencoder
from ultralytics import YOLO
import time

# Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RefleksNet yükle
reflex_model = HandAutoencoder().to(device)
reflex_model.load_state_dict(torch.load("model/autoencoder.pth", map_location=device))
reflex_model.eval()

# YOLO modelini yükle (pretrained)
yolo = YOLO("yolov8n.pt")  # Küçük ve hızlı model

# Görüntü dönüştürücü (RefleksNet için)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

THRESHOLD = 0.02  # RefleksNet için yeniden yapılandırma eşik değeri

cap = cv2.VideoCapture(0)
reflexnet_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === YOLO için süre ölçümü ===
    yolo_start = time.time()
    frame_yolo = frame.copy()
    results = yolo(frame_yolo, verbose=False)[0]
    yolo_elapsed = (time.time() - yolo_start) * 1000  # ms

    for box in results.boxes:
        cls = yolo.model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # YOLO ms ekrana yaz
    cv2.putText(frame, f"YOLO: {yolo_elapsed:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # === YOLO için kopya al ===
    frame_yolo = frame.copy()
    results = yolo(frame_yolo, verbose=False)[0]  # YOLO tahmini
    for box in results.boxes:
        cls = yolo.model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # === RefleksNet testi ===
    reflex_start = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = reflex_model(img_tensor)
        loss = F.mse_loss(recon, img_tensor).item()
    reflex_elapsed = (time.time() - reflex_start) * 1000  # ms
    reflexnet_times.append(reflex_elapsed)

    print(f"RefleksNet loss: {loss:.5f} | Threshold: {THRESHOLD}")
    if loss < THRESHOLD:
        # EL GÖRÜLDÜ — ACİL DURDUR
        cv2.putText(frame, "REFLEKSNET: EL ALGILANDI!!!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("⚠️ RefleksNet devreye girdi! El algılandı!")
        cv2.imshow("YOLO + RefleksNet", frame)
        cv2.waitKey(0)  # Video durur
        break

    cv2.imshow("YOLO + RefleksNet", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# En son RefleksNet modelinin ortalama ms süresini konsola yaz
if reflexnet_times:
    ortalama_ms = sum(reflexnet_times) / len(reflexnet_times)
    print(f"RefleksNet ortalama tahmin süresi: {ortalama_ms:.1f} ms")
else:
    print("RefleksNet tahmin süresi ölçülemedi.")
