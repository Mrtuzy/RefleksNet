import cv2
import time
import torch
import torch.nn as nn
from norse.torch.module.lif import LIFCell
from torchvision import transforms
import argparse
import numpy as np

# ---------
# ReflexNet SNN modeli tanımı (eğitilmiş .pth dosyasını yüklemek için)
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

# ---------
# Görüntüyü SNN girişine dönüştür
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# ---------
def snn_detect_hand(model, device, frame):
    # OpenCV BGR->PIL Image
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to(device)

    output, _ = model(input_tensor)
    spike_prob = output.item()
    # 0.5 threshold ile karar veriyoruz
    return spike_prob > 0.99999999

# ---------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SNN modelini yükle
    model = ReflexNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # YOLO modeli (PyTorch Hub kullanımı)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device).eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    print("Kamera açıldı. El algılama başlıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        reflex_detected = False
        reflex_time = None

        # Asenkron çalıştırma benzeri basit paralellik:
        # Önce SNN refleks
        if not args.no_reflex:
            reflex_start = time.time()
            if snn_detect_hand(model, device, frame):
                reflex_time = time.time() - reflex_start
                print(f"⚡ Refleks: El algılandı! Süre: {reflex_time*1000:.1f} ms")
                reflex_detected = True

        # Refleks yoksa YOLO devam eder
        if not reflex_detected:
            # YOLO için görüntü RGB formatına çevir ve modele ver
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo_model(img_rgb)

            # Sonuçları filtrele (sadece "person" ve "hand" classları vs)
            # YOLOv5 default modelinde 'hand' sınıfı olmayabilir,
            # El algılamak için özel model gerekebilir
            # Burada örnek olarak 'person' sınıfını kullanalım
            preds = results.xyxy[0]  # x1,y1,x2,y2,conf,class

            # Sadece 0 sınıf (person) var genelde
            for *box, conf, cls in preds:
                cls = int(cls)
                conf = float(conf)
                if conf > 0.5:
                    label = results.names[cls]
                    x1,y1,x2,y2 = map(int, box)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Frame gösterimi
        cv2.imshow("El Algılama - RefleksNet + YOLO", frame)

        # Çıkış ESC tuşu ile
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="RefleksNet + YOLO El Algılama")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Eğitilmiş SNN modelinin .pth dosya yolu")
    parser.add_argument("--no_reflex", action="store_true",
                        help="Refleks SNN'yi kapat, sadece YOLO çalışsın")
    args = parser.parse_args()

    main(args)
# Bu kod, ReflexNet SNN modelini kullanarak el algılama yapar.