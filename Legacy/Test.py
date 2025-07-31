import cv2
import torch
import torchvision.transforms as T
import time
from Training import ReflexNet  # Eğittiğimiz modelin tanımı burada olmalı

# Cihaz ayarı (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükle
model = ReflexNet().to(device)
model.load_state_dict(torch.load("../demo1/saved_models/reflexnet_hands_vs_nonhands.pth", map_location=device))
model.eval()

# Görüntü ön işleme (modelin beklediği boyut ve normalize)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((28, 28)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Kamerayı aç
cap = cv2.VideoCapture(0)

print("El algılama testi başlıyor. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamıyor.")
        break

    # Model için uygun input hazırla
    img = transform(frame).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        output, activations = model(img)
        prob = torch.sigmoid(output).item()
    elapsed_ms = (time.time() - start_time) * 1000

    # Eşik değeri 0.5 olarak ayarlandı, değiştirebilirsin
    hand_detected = prob > 0.5

    # Sonucu ekrana yazdır
    text = f"El Algilandi: {'EVET' if hand_detected else 'HAYIR'} | Olasilik: {prob:.3f} | Sure: {elapsed_ms} ms"
    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if(hand_detected):
        print(f"{elapsed_ms}")
    # Görüntüyü göster
    cv2.imshow("RefleksNet El Algılama", frame)

    # 'q' ile çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
